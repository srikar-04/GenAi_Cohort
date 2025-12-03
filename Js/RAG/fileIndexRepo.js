// writing functions to handle crud operations on the file index

import db from "./init_db.js";

const TableName = "fileIndex"

// -->> CRUD OPERATIONS <<--

export function insertFileIndex(entry) {
    const insert = db.prepare(`
            INSERT INTO 
            ${TableName} ( fileHash, fileName, fileSize, uploadedAt, updatedAt,chunkCount ) 
            VALUES (@fileHash, @fileName, @fileSize, @uploadedAt, @updatedAt, @chunkCount)
        `)
    return insert.run(entry)
}

export function getAllFileIndex() {
    const getStmt = db.prepare(`SELECT * FROM ${TableName}`)
    return getStmt.all()
}

export function getFileIndexById(fileHash) {
    const getByIdStmt = db.prepare(`SELECT * FROM ${TableName} WHERE fileHash = ?`)
    return getByIdStmt.get(fileHash)
}

export function updateFileIndex(entry) {
    const updateStmt = db.prepare(`
            UPDATE ${TableName}
            SET fileHash = @fileHash,
            fileName = @fileName, 
            fileSize = @fileSize, 
            uploadedAt = @uploadedAt,
            updatedAt = @updatedAt,
            chunkCount = @chunkCount
            WHERE fileHash = @fileHash
        `)
    return updateStmt.run(entry)
}

export function deleteFileIndex(fileHash) {
    const deleteStmt = db.prepare(`DELETE FROM ${TableName} WHERE fileHash = ?`)
    return deleteStmt.run(fileHash)
}