import fs from "fs";
import crypto from "crypto";

function routeFile() {

}

function hashBuffer(buffer) {
    const fileHash = crypto.createHash('sha256').update(buffer).digest('hex')
    console.log('unique file hash : ', fileHash)

    // check whether we have already indexed this file using it's hash
    // if yes then the file is already indexed (duplicate file is uploaded)
    // if no then index the file
}

(async () => {

    try {
        const pdfPath = "C:/Users/srikar/Desktop/RAG_Docs/leph1dd/leph102.pdf";
        // 1) creating a buffer for the pdf file
        const pdfBuffer = fs.readFileSync(pdfPath);
        // 2) Hashing the buffer to avoid deduplication
        hashBuffer(pdfBuffer)
    } catch (error) {
        console.log('pipeline error : ', error)
    }
})()