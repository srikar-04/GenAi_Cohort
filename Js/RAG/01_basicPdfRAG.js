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
dotenv.config()
import OpenAI from 'openai'
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf"
import path from 'path'
import fs from 'fs'
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
import { QdrantVectorStore } from "@langchain/qdrant"
import { QdrantClient } from '@qdrant/js-client-rest'
import readline from 'readline'


const userDir = 'C:/Users/srikar/Desktop/RAG_Docs/leph1dd'

const pdfPath = path.join(userDir, 'leph1an.pdf')
const cachePath = path.join(userDir, 'lenph1an_cache.json')

let docs;

// (1) Loading docs either from cache or locally

if (fs.existsSync(cachePath)) {
    console.log('📦 Getting Pdf from cache : ', cachePath)

    const raw = fs.readFileSync(cachePath, 'utf-8')

    docs = JSON.parse(raw)
    // console.log(docs[0].pageContent.slice(0, 40))

} else if(fs.existsSync(pdfPath)) {

    console.log('📄 pdf found at : ', pdfPath)

    const loader = new PDFLoader(pdfPath)

    docs = await loader.load()
    // console.log(docs[0].pageContent.slice(0, 40))

    // caching the loaded documents
    fs.writeFileSync(cachePath, JSON.stringify(docs, null, 2))
    console.log('✅ Saved to cache:', cachePath);

} else {
    console.error('File not found:', pdfPath);
    process.exit(1)
}

// (2) splitting text using recursive text splitter -->> splitting based on the pdf structure (headings, paragraphs and sentences)

let splitDocs;

const splittedCachePath = path.join(userDir, 'lenph1an_split_cache.json')

if(fs.existsSync(splittedCachePath)) {
    console.log("📦 Getting splitted chunks from cache : ", splittedCachePath)

    const raw = fs.readFileSync(splittedCachePath, 'utf-8')

    splitDocs = JSON.parse(raw)
    console.log("splitted docs lenght :", splitDocs.length)
} else {

    console.log('📄 Splitting the document ')

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 700,
        chunkOverlap: 400
    })
    
    splitDocs = await textSplitter.splitDocuments(docs)
    
    console.log("length of the splitted text chunks : ", splitDocs.length)
    // console.log("length of the splitted text chunks : ", texts[0].pageContent.slice(0, 100))

    // caching the splitted chunks

    fs.writeFileSync(splittedCachePath, JSON.stringify(splitDocs, null, 2))
    console.log('✅ Chunks saved to cache:', splittedCachePath);
}

// (3) embedding the splitted text and storing it in vector DB :

const embedder = new GoogleGenerativeAIEmbeddings({
    model: "text-embedding-004"
});

const QdrantURL = "http://localhost:6333"
const QdrantCollection = "NCERT_Physics_1"

// check if there are any exsisting collections with embeddings 

const Qclient = new QdrantClient({ host: "localhost", port: 6333 })

const exsistingCollections = await Qclient.getCollections()
const needIngestion = exsistingCollections.collections?.some(col => col.name === QdrantCollection)

let vectorStore;

if(!needIngestion) {
    console.log('📊 storing data in new collection')
    vectorStore = await QdrantVectorStore.fromDocuments(splitDocs, embedder, {
        url: QdrantURL,
        collectionName: QdrantCollection
    })
} else {
    const collectionInfo = await Qclient.getCollection(QdrantCollection)
    const isEmpty = collectionInfo.points_count === 0

    if(isEmpty) {
        console.log('📊 collection exists but is empty, adding documents')
        vectorStore = await QdrantVectorStore.fromExistingCollection(embedder, {
            url: QdrantURL,
            collectionName: QdrantCollection
        })
        await vectorStore.addDocuments(splitDocs)
    } else {
        console.log('📊 collection exists with data, skipping documents addition')
    }
    // else {
    //     console.log('📊 collection exists with data, adding more documents')
    //     vectorStore = await QdrantVectorStore.fromExistingCollection(embedder, {
    //         url: QdrantURL,
    //         collectionName: QdrantCollection
    //     })
    //     await vectorStore.addDocuments(splitDocs)
    // }
}

// (4) querying the vector store by turning it into a retireiver

const retireiver = vectorStore.asRetriever()

// const apiKey = process.env.GEMINI_API_KEY

// if(!apiKey) throw new Error('api key not found')

// const client = new OpenAI({
//     apiKey: apiKey,
//     baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/"
// })

// const messages = []

// const rl = readline.createInterface({
//     input: process.stdin,
//     output: process.stdout,
// });


// let realtedContext;

// while (true) {

//     realtedContext = ""
    
//     rl.question('🧠 Enter your qeury (or type "exit")', async (qeury) => {

//         messages.push({
//             role: "user",
//             content: qeury
//         })

//         if(qeury.toLowerCase === 'exit') {
//             rl.close()
//         }

//         // retireiving related docuemnts
//         const relevantDocs = retireiver.invoke(qeury)
//         relevantDocs.map(doc => realtedContext.concat("\n\n", doc.pageContent, "\n\n"))
//     })
// }

// const systemPrompt = `
//     Answert the use query only based on the context available which is delimented in triple quotes :
//     """${realtedContext}""""

//     If the questions does not relate to the available context then response with "no context available for this topic" 
// `
// messages.push({
//     role: "developer",
//     content: systemPrompt
// })

// const response = await client.chat.completions.create({
//     model: "gemini-2.0-flash",
//     messages: messages
// })

