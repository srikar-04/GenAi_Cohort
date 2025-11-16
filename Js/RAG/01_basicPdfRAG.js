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

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf"
import path from 'path'
import fs from 'fs'
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"

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