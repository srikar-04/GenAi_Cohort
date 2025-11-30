import Database from 'better-sqlite3'

const db = new Database('sql_test.db')

db.prepare(`
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT
        )
    `).run()

// -->> WRITING TO DATABASE <<--

const insert = db.prepare(`
        INSERT INTO users (name, email) VALUES (?, ?)
    `)

// insert.run('konda', 'konda@gmail.com')

// -->> READING FROM THE DATABASE <<--

const rows = db.prepare(`
        SELECT * FROM users 
    `).all()

console.log('all rows : ', rows)

const specificRow = db.prepare(`
        SELECT * FROM users WHERE id = ?
    `).get(4)

console.log('row with specific id : ', specificRow)

// -->> UPDATING THE DATABASE <<--

// const update = db.prepare(`
//         UPDATE users SET name = ? WHERE id = ?
//     `)

// console.log(update.run('srikar', 1))


// -->> DELETING FROM THE DATABASE <<--

const deleteRow = db.prepare(`
        DELETE FROM users WHERE id = ?
    `)
console.log(deleteRow.run(4))