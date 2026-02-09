import sqlite3
import aiosqlite
from typing import List, Dict


DB_PATH = "chat_history.db"

class DatabaseManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db_sync()

    def _init_db_sync(self):
        """Initialize headers synchronously to ensure tables exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL;")
            
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            conn.commit()

    async def create_session(self, session_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES (?)", (session_id,))
            await db.commit()

    async def add_message(self, session_id: str, role: str, content: str):
        # ensure session exists
        await self.create_session(session_id)
        
        print(f"[DB] Adding message ({role}) for session {session_id}...")
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, role, content)
                )
                await db.commit()
            print(f"[DB] Message added successfully.")
        except Exception as e:
            print(f"[DB ERROR] Failed to add message: {e}")
            raise

    async def get_history(self, session_id: str) -> List[Dict]:
        print(f"[DB] Fetching history for session {session_id}...")
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                    (session_id,)
                ) as cursor:
                    rows = await cursor.fetchall()
                    history = [{"role": row["role"], "content": row["content"], "timestamp": row["timestamp"]} for row in rows]
                    print(f"[DB] Fetched {len(history)} messages.")
                    return history
        except Exception as e:
            print(f"[DB ERROR] Failed to fetch history: {e}")
            return []

    async def clear_history(self, session_id: str):
        print(f"[DB] Clearing history for session {session_id}...")
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            await db.commit()
        print(f"[DB] History cleared.")
