import fs from "fs";
import crypto from "crypto";
import db from "./init_db.js";
import { v5 as uuidv5 } from 'uuid'

import { insertFileIndex, getFileIndexById, updateFileIndex, deleteFileIndex } from './fileIndexRepo.js'

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf"
import dotenv from 'dotenv'
import path from "path";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { QdrantClient } from "@qdrant/js-client-rest";
import { QdrantVectorStore } from "@langchain/qdrant";
import { fileURLToPath } from 'url';
import readline from 'readline'
import OpenAI from 'openai'


const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.resolve(__dirname, '../.env') });

const apiKey = process.env.GOOGLE_API_KEY;
if (!apiKey) throw new Error('api key not found');

const QdrantCollection = 'NCERT_Physics_2'
const QdrantURL = "http://localhost:6333"

// const NAMESPACE = '3d594650-3436-5d7c-9d8f-fdc0bce4c8d5';
const NAMESPACE = uuidv5.URL;

// if different file formats are uploaded then we need to route them to different loaders
function routeFile() {

}

async function llmCall(vectorStore) {
  let messages = []

  const client = new OpenAI({
    apiKey: apiKey,
    baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/"
  })

  // setting up readline
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  // helper function for asking questions
  const askQuestion = (query) => {
    return new Promise((resolve) => rl.question(query, resolve))
  }

  console.log('\n🤖 Chatbot ready! Type "exit" to quit.\n');
  while (true) {
    const userQuery = await askQuestion("🧠 Enter your query: ")

    if (userQuery.toLowerCase() === 'exit') {
      console.log('\n🤖 Chatbot exiting.\n')
      rl.close()
      break
    }

    messages.push({
      'role': 'user',
      'content': userQuery
    })

    // extracting relevant context from the retriever 
    console.log("🔍 Retrieving context... \n ");
    const retriever = vectorStore.asRetriever()
    let relevantContext;
    try {
      const relevantDocs = await retriever.invoke(userQuery)
      relevantContext = relevantDocs.map(doc => doc.pageContent).join("\n\n \n\n")
    } catch (error) {
      console.log('error in retrieving context : ', error)
      return
    }

    // writing system prompt
    const systemPrompt = `
        Answer the user query only based on the context available which is delimited in triple quotes :
        """${relevantContext}"""

        If the question does not relate to the available context then response with "no context available for this topic" 

        If you think you have the related context to answer the question then answer it by breaking down the answer in simple terms and make sure not to include any extra information that is not related to the question or the context.`

    const currentMessages = [
      { role: "system", content: systemPrompt },
      ...messages
    ];

    try {
      const response = await client.chat.completions.create({
        model: "gemini-2.5-flash-lite",
        // model: "gemini-2.0-flash",
        messages: currentMessages
      })

      const answer = response.choices[0].message.content;
      console.log(`\n🤖 Answer: ${answer}\n`);

      messages.push({
        role: "assistant",
        content: answer
      });

    } catch (error) {
      console.error("❌ Error generating response:", error);
    }

  }
}

async function getVectorStore() {
  const embedder = new GoogleGenerativeAIEmbeddings({
    apiKey: apiKey,
    model: "text-embedding-004"
  })

  return await QdrantVectorStore.fromExistingCollection(embedder, {
    url: QdrantURL,
    collectionName: QdrantCollection
  })
}

async function chunkAndVector(pdfPath, fileHash, vectorStore) {

  // adding the parent db entry before adding the chunks table

  const entry = {
    fileHash: fileHash,
    fileName: path.basename(pdfPath),
    fileSize: fs.statSync(pdfPath).size,
    uploadedAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    version: 1,
    chunkCount: 0
  }

  insertFileIndex(entry)

  const normalize = (text) =>
    text.replace(/\r\n/g, '\n').replace(/\s+/g, ' ').trim().toLowerCase();

  const hashChunk = (text) =>
    crypto.createHash('sha256').update(text, 'utf8').digest('hex');

  // 1. Load PDF
  const loader = new PDFLoader(pdfPath);
  const docs = await loader.load();

  // 2. Split
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 700,
    chunkOverlap: 400,
  });
  const splitDocs = await splitter.splitDocuments(docs);

  // 3. Normalize + hash
  const newChunks = splitDocs.map((doc, index) => {
    const normalized = normalize(doc.pageContent);
    const chunkHash = hashChunk(normalized);

    return {
      chunkHash,
      pageContent: normalized,
      metadata: JSON.stringify({
        chunkIndex: index,
        source: path.basename(pdfPath)
      })
    };
  });

  // 4. Fetch old chunk hashes from DB
  const oldChunks = db
    .prepare(`SELECT chunkHash FROM fileChunks WHERE fileHash = ?`)
    .all(fileHash)
    .map(r => r.chunkHash);

  const oldSet = new Set(oldChunks);
  const newSet = new Set(newChunks.map(c => c.chunkHash));

  const added = newChunks.filter(c => !oldSet.has(c.chunkHash));
  const removed = oldChunks.filter(hash => !newSet.has(hash));
  const removedUUID = removed.map(hash => uuidv5(hash, NAMESPACE));

  console.log(`🧩 Added: ${added.length}, Removed: ${removed.length}`);

  // 5. DELETE removed chunks (Qdrant + DB)
  if (removed.length > 0) {
    await vectorStore.client.delete(QdrantCollection, {
      points: removedUUID
    });

    const deleteStmt = db.prepare(
      `DELETE FROM fileChunks WHERE fileHash = ? AND chunkHash = ?`
    );

    for (const hash of removed) {
      deleteStmt.run(fileHash, hash);
    }
  }

  // 6. UPSERT added chunks
  if (added.length > 0) {
    const embedder = new GoogleGenerativeAIEmbeddings({
      apiKey,
      model: "text-embedding-004",
    });

    const vectors = await embedder.embedDocuments(
      added.map(c => c.pageContent)
    );

    const points = added.map((c, i) => ({
      id: uuidv5(c.chunkHash, NAMESPACE),               // ← Qdrant point ID
      vector: vectors[i],
      payload: JSON.parse(c.metadata)
    }));

    await vectorStore.client.upsert(QdrantCollection, { points });

    const insertStmt = db.prepare(
      `INSERT INTO fileChunks (fileHash, chunkHash, metadata)
      VALUES (?, ?, ?)`
    );

    for (const c of added) {
      insertStmt.run(fileHash, c.chunkHash, c.metadata);
    }
  }

  // 7. Update fileIndex
  updateFileIndex({
    fileHash,
    fileName: path.basename(pdfPath),
    fileSize: fs.statSync(pdfPath).size,
    uploadedAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    version: 1,
    chunkCount: newChunks.length,
  });

  console.log("✅ Incremental indexing complete");
}



function hashBuffer(buffer) {
  const fileHash = crypto.createHash('sha256').update(buffer).digest('hex')
  // console.log('unique file hash : ', fileHash, '\n \n')

  // check whether we have already indexed this file using it's hash (by querying the db)
  // if yes then the file is already indexed (duplicate file is uploaded), skipping all the heavy operations like normalizing, chunking etc...
  // if no then index the file and add an entry to db with all the metadata

  let fileIndex;
  try {
    fileIndex = getFileIndexById(fileHash)
  } catch (error) {
    console.log('error while checking file index : ', error, '\n')
    return false
  }

  if (fileIndex) {
    console.log('file is present in the index \n 🚀 skipping operations \n')
    return false
  } else {
    console.log('file is not present in the index \n 🪓 performing operations \n ')
    return fileHash
  }
}

(async () => {

  try {
    // we are getting this locally for now
    // in future, while writing backend, we will get this from the frontend

    // const pdfPath = "C:/Users/srikar/Desktop/RAG_Docs/leph1dd/leph102.pdf";
    const pdfPath = "C:/Users/srikar/OneDrive/Desktop/RAG_Docs/leph1dd/leph102.pdf";
    // 1) creating a buffer for the pdf file
    const pdfBuffer = fs.readFileSync(pdfPath);
    // 2) Hashing the buffer to avoid deduplication
    const result = hashBuffer(pdfBuffer)


    // result is the fileHash that is returned from the hashBuffer function
    // getting vector store and passing it to the chunkAndVector function
    let vectorStore = await getVectorStore()

    if (result) {
      await chunkAndVector(pdfPath, result, vectorStore)
    }

    if (vectorStore) {
      await llmCall(vectorStore)
    } else {
      console.log('💀 vector store not found')
    }

  } catch (error) {
    console.log('pipeline error : ', error)
  }
})()