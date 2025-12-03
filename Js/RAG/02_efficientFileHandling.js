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

dotenv.config({ path: path.resolve(__dirname, '../.env') });

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) throw new Error('api key not found');

function routeFile() {

}

async function chunkAndVector(pdfPath, fileHash) {
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

    const QdrantURL = "http://localhost:6333"
    const QdrantCollection = "NCERT_Physics_2"

    const Qclient = new QdrantClient({host: 'localhost', port: 6333})

    // checking qdrant connection
    try {
        await Qclient.getCollections()
    } catch (error) {
        console.log('could not connect to qdrant client ')
        process.exit(1)
    }

    const existingCollections = await Qclient.getCollections()
    const collectionExists = existingCollections.collections?.some(col => col.name === QdrantCollection)

    // 4) store the chunks in vector db
    let vectorStore;
    if(!collectionExists) {
        console.log('📊 storing data in new collection')
        vectorStore = await QdrantVectorStore.fromDocuments(splittedDocs, embedder, {
            url: QdrantURL,
            collectionName: QdrantCollection
        })
    } else {
        const collectionInfo = await Qclient.getCollection(QdrantCollection)
        const isEmpty = collectionInfo.points_count === 0

        if(isEmpty) {
            // creating an entirely new collection
            vectorStore = await QdrantVectorStore.fromExistingCollection(embedder, {
                url: QdrantURL,
                collectionName: QdrantCollection
            })
            vectorStore.addDocuments(splittedDocs)
        } else {
            // collection exists with data, skipping documents addition
            vectorStore = await QdrantVectorStore.fromExistingCollection(embedder, {
                url: QdrantURL,
                collectionName: QdrantCollection
            })
        }
    }

    // 5) updating the file index -->> final step
    const entry = {
        fileHash: fileHash,
        fileName: path.basename(pdfPath),
        fileSize: fs.statSync(pdfPath).size,
        uploadedAt: new Date(),
        updatedAt: new Date(),
        chunkCount: splittedDocs.length
    }

    insertFileIndex(entry)

    console.log('✅ file indexed successfully')
    return
}

async function hashBuffer(buffer) {
    const fileHash = crypto.createHash('sha256').update(buffer).digest('hex')
    console.log('unique file hash : ', fileHash, '\n \n')

    // check whether we have already indexed this file using it's hash (by querying the db)
    // if yes then the file is already indexed (duplicate file is uploaded), skipping all the heavy operations like normalizing, chunking etc...
    // if no then index the file and add an entry to db with all the metadata

    try {
        const fileIndex = getFileIndexById(fileHash)
    } catch (error) {
        console.log('error while checking file index : ', error)
        return false
    }

    if(fileIndex) {
        console.log('file is present in the index \n 🚀 skipping operations')
        return false
    } else {
        console.log('file is not present in the index \n 🪓 performing operations')
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
        if(result) {
            await chunkAndVector(pdfPath, result)
            console.log('🤖 move to the ai query part')
        } else {
            console.log('file is already indexed \n 🚀 skipping operations')
        }

    } catch (error) {
        console.log('pipeline error : ', error)
    }
})()