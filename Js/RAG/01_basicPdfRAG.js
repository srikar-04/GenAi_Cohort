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

const userDir = 'C:/Users/srikar/Desktop/RAG_Docs/leph1dd'

const pdfPath = path.join(userDir, 'leph1an.pdf')

if (fs.existsSync(pdfPath)) {
    console.log('PDF found at:', pdfPath);

    const loader = new PDFLoader(pdfPath)

    const docs = await loader.load()

    // console.log(docs[0].pageContent)

} else {
    console.error('File not found:', pdfPath);
    process.exit(1)
}
