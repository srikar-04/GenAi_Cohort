import fs from "fs";
import crypto from "crypto";

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

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) throw new Error('api key not found');

const QdrantCollection = 'NCERT_Physics_2'
const QdrantURL = "http://localhost:6333"

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
    while(true) {
        const userQuery = await askQuestion("🧠 Enter your query: ")

        if(userQuery.toLowerCase === 'exit') {
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
                model: "gemini-2.0-flash",
                messages: currentMessages
            })

            const answer = response.choices[0].message.content;
            console.log(`\n🤖 Answer: ${answer}\n`);

            messages.push({
                role: "assistant",
                content: answer
            });

        } catch (error) {
            console.error("❌ Error generating response:", error.message);
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
    // 1) load the pdf from local disk
    let docs;
    if(fs.existsSync(pdfPath)) {
        console.log('pdf found at : ', pdfPath, '\n', '🚀 loading document from local disk \n ')
        const loader = new PDFLoader(pdfPath)
        docs = await loader.load()
    }

    // 2) splitting the document (chunking)
    let splittedDocs;
    if(docs) {
        console.log('🚀 splitting docs \n')
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 700,
            chunkOverlap: 400
        })
        splittedDocs = await textSplitter.splitDocuments(docs)
    }

    // 3) embed the chunks
    const embedder = new GoogleGenerativeAIEmbeddings({
        apiKey: apiKey,
        model: "text-embedding-004"
    })

    const Qclient = new QdrantClient({host: 'localhost', port: 6333})

    // checking qdrant connection
    try {
        await Qclient.getCollections()
    } catch (error) {
        console.log('could not connect to qdrant client \n ')
        process.exit(1)
    }

    const existingCollections = await Qclient.getCollections()
    const collectionExists = existingCollections.collections?.some(col => col.name === QdrantCollection)

    // 4) store the chunks in vector db
    // getting vector store from from the place where we are calling function, in the main function we are getting it

    if(!collectionExists) {
        console.log('📊 storing data in new collection \n')
        vectorStore = await QdrantVectorStore.fromDocuments(splittedDocs, embedder, {
            url: QdrantURL,
            collectionName: QdrantCollection
        })     
    } else {
        const collectionInfo = await Qclient.getCollection(QdrantCollection)
        const isEmpty = collectionInfo.points_count === 0

        if(isEmpty) {
            // adding docs to existing empty collection
            vectorStore.addDocuments(splittedDocs)
        } 
    }


    // 5) updating the file index -->> final step
    const entry = {
        fileHash: fileHash,
        fileName: path.basename(pdfPath),
        fileSize: fs.statSync(pdfPath).size,
        uploadedAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        version: 1,
        chunkCount: splittedDocs.length
    }

    insertFileIndex(entry)

    console.log('✅ file indexed successfully \n')
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

    if(fileIndex) {
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

        const pdfPath = "C:/Users/srikar/Desktop/RAG_Docs/leph1dd/leph102.pdf";
        // 1) creating a buffer for the pdf file
        const pdfBuffer = fs.readFileSync(pdfPath);
        // 2) Hashing the buffer to avoid deduplication
        const result = hashBuffer(pdfBuffer)


        // result is the fileHash that is returned from the hashBuffer function
        // getting vector store and passing it to the chunkAndVector function
        let vectorStore = await getVectorStore()

        if(result) {
            await chunkAndVector(pdfPath, result, vectorStore)
        }

        if(vectorStore) {
            await llmCall(vectorStore)
        } else {
            console.log('💀 vector store not found')
        }

    } catch (error) {
        console.log('pipeline error : ', error)
    }
})()