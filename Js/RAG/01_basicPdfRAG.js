// load document by specifying the path
// if it is already loaded then it is present in the cache
// split the document using chunking method you want
// if it is already splitted then no need to do again
// embbed the splitted text
// store the embeddings in vector db

// embed the user query
// search vector db using the user's query embeddings
// feed it to the context of llm
// generate final output

import dotenv from 'dotenv'
import OpenAI from 'openai'
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf"
import path from 'path'
import fs from 'fs'
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
import { QdrantVectorStore } from "@langchain/qdrant"
import { QdrantClient } from '@qdrant/js-client-rest'
import readline from 'readline'
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load .env from the Js directory (parent of RAG)
dotenv.config({ path: path.resolve(__dirname, '../.env') });

async function main() {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) throw new Error('api key not found');

    console.log("🚀 Starting main function...");
    // NOTE: You might need to adjust these filenames or ensure the files exist
    const pdfPath = 'C:/Users/srikar/Desktop/RAG_Docs/leph1dd/leph101.pdf';
    const cachePath = 'C:/Users/srikar/Desktop/RAG_Docs/leph1dd/lenph1an_cache.json';
    const splittedCachePath = 'C:/Users/srikar/Desktop/RAG_Docs/leph1dd/lenph1an_split_cache.json';

    console.log("📂 Paths defined.");
    let docs;

    // (1) Loading docs either from cache or locally
    if (fs.existsSync(cachePath)) {
        console.log('📦 Getting Pdf from cache : ', cachePath)
        const raw = fs.readFileSync(cachePath, 'utf-8')
        docs = JSON.parse(raw)
    } else if (fs.existsSync(pdfPath)) {
        console.log('📄 pdf found at : ', pdfPath)
        const loader = new PDFLoader(pdfPath)
        docs = await loader.load()

        // caching the loaded documents
        fs.writeFileSync(cachePath, JSON.stringify(docs, null, 2))
        console.log('✅ Saved to cache:', cachePath);
    } else {
        console.error('❌ File not found:', pdfPath);
        console.error('Please ensure the PDF file exists at the specified path.');
        process.exit(1)
    }

    // (2) splitting text using recursive text splitter
    let splitDocs;

    if (fs.existsSync(splittedCachePath)) {
        console.log("📦 Getting splitted chunks from cache : ", splittedCachePath)
        const raw = fs.readFileSync(splittedCachePath, 'utf-8')
        splitDocs = JSON.parse(raw)
        console.log("splitted docs length :", splitDocs.length)
    } else {
        console.log('📄 Splitting the document ')
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 700,
            chunkOverlap: 400
        })

        splitDocs = await textSplitter.splitDocuments(docs)
        console.log("length of the splitted text chunks : ", splitDocs.length)

        // caching the splitted chunks
        fs.writeFileSync(splittedCachePath, JSON.stringify(splitDocs, null, 2))
        console.log('✅ Chunks saved to cache:', splittedCachePath);
    }

    // (3) embedding the splitted text and storing it in vector DB :
    const embedder = new GoogleGenerativeAIEmbeddings({
        apiKey: apiKey,
        model: "text-embedding-004"
    });

    const QdrantURL = "http://localhost:6333"
    const QdrantCollection = "NCERT_Physics_1"

    const Qclient = new QdrantClient({ host: "localhost", port: 6333 })

    // Check connection to Qdrant
    try {
        await Qclient.getCollections();
    } catch (e) {
        console.error("❌ Could not connect to Qdrant. Make sure it is running on port 6333.");
        process.exit(1);
    }

    const existingCollections = await Qclient.getCollections()
    const collectionExists = existingCollections.collections?.some(col => col.name === QdrantCollection)

    let vectorStore;

    if (!collectionExists) {
        console.log('📊 storing data in new collection')
        vectorStore = await QdrantVectorStore.fromDocuments(splitDocs, embedder, {
            url: QdrantURL,
            collectionName: QdrantCollection
        })
    } else {
        const collectionInfo = await Qclient.getCollection(QdrantCollection)
        const isEmpty = collectionInfo.points_count === 0

        if (isEmpty) {
            console.log('📊 collection exists but is empty, adding documents')
            vectorStore = await QdrantVectorStore.fromExistingCollection(embedder, {
                url: QdrantURL,
                collectionName: QdrantCollection
            })
            await vectorStore.addDocuments(splitDocs)
        } else {
            console.log('📊 collection exists with data, skipping documents addition')
            // Initialize vector store from existing collection
            vectorStore = await QdrantVectorStore.fromExistingCollection(embedder, {
                url: QdrantURL,
                collectionName: QdrantCollection
            })
        }
    }

    // (4) querying the vector store
    const retriever = vectorStore.asRetriever()

    const client = new OpenAI({
        apiKey: apiKey,
        baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/"
    })

    const messages = []

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    // Helper for async question
    const askQuestion = (query) => {
        return new Promise((resolve) => rl.question(query, resolve));
    };

    console.log('\n🤖 Chatbot ready! Type "exit" to quit.\n');

    while (true) {
        const query = await askQuestion('🧠 Enter your query: ');

        if (query.toLowerCase() === 'exit') {
            rl.close();
            break;
        }

        messages.push({
            role: "user",
            content: query
        })

        // retrieving related documents
        console.log("🔍 Retrieving context...");
        const relevantDocs = await retriever.invoke(query);

        // Correctly map docs to a string
        const relatedContext = relevantDocs.map(doc => doc.pageContent).join("\n\n   \n\n");

        const systemPrompt = `
    Answer the user query only based on the context available which is delimited in triple quotes :
    """${relatedContext}"""

    If the question does not relate to the available context then response with "no context available for this topic" 

    If you think you have the related context to answer the question then answer it by breaking down the answer in simple terms and make sure not to include any extra information that is not related to the question or the context.
        `

        // We don't want to keep pushing system prompts to history forever, 
        // but for this simple loop, we'll just send it as the latest system message 
        // or just prepend it to the current turn. 
        // The original code pushed it to 'messages' array which grows indefinitely.
        // Let's keep it simple but functional: use a temporary message list for the API call
        // or just push it. Pushing multiple system messages is generally fine in some APIs but 
        // usually you want one system message or context in the user message.
        // Let's follow the original intent but fix the structure.

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

main().catch(console.error);
