// initialize the schema for the database for the 02_efficientFileHandling.js

// we are using better-sqlite3

// defining the schema

import Database from 'better-sqlite3'

const db = new Database('fileIndex.db')

// creating table for file indexing (schema)

db.exec(`
        CREATE TABLE IF NOT EXISTS fileIndex (
            fileHash TEXT PRIMARY KEY,
            fileName TEXT,
            fileSize INTEGER,
            uploadedAt TEXT,
            updatedAt TEXT,
            version INTEGER,
            chunkCount INTEGER
        );

        CREATE TABLE IF NOT EXISTS fileChunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fileHash TEXT,
            chunkHash TEXT,
            metadata TEXT,
            FOREIGN KEY (fileHash) REFERENCES fileIndex(fileHash) ON DELETE CASCADE
        );
    `)

console.log('database initialized successfully')

export default db