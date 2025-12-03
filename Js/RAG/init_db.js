// initialize the schema for the database for the 02_efficientFileHandling.js

// we are using better-sqlite3

// defining the schema

import Database from 'better-sqlite3'

const db = new Database('fileIndex.db')

// creating table for file indexing (schema)

db.prepare(`
        CREATE TABLE IF NOT EXISTS fileIndex (
            fileHash TEXT PRIMARY KEY,
            fileName TEXT,
            fileSize INTEGER,
            uploadedAt TEXT,
            updatedAt TEXT,
            version INTEGER,
            chunkCount INTEGER,
        )
    `).run()

console.log('database initialized successfully')

export default db