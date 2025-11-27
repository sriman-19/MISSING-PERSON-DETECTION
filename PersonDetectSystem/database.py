import sqlite3
import pickle
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

DATABASE_PATH = "database.sqlite"


# ---------------------------
# Initialize Database Tables
# ---------------------------
def init_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users_cases (
            case_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            contact TEXT NOT NULL,
            description TEXT,
            location TEXT,
            image_path TEXT NOT NULL,
            embedding BLOB,
            status TEXT DEFAULT 'QUEUED',
            match_place TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            frame_path TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            similarity REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (case_id) REFERENCES users_cases(case_id)
        )
    """)

    conn.commit()
    conn.close()


# ---------------------------
# Create a new user case
# ---------------------------
def create_case(name: str, age: int, contact: str, description: str,
                location: str, image_path: str) -> str:
    case_id = str(uuid.uuid4())[:8]  # Short unique case ID

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO users_cases (case_id, name, age, contact, description, location, image_path, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'QUEUED')
    """, (case_id, name, age, contact, description, location, image_path))

    conn.commit()
    conn.close()

    return case_id


# ---------------------------
# Retrieve a single case
# ---------------------------
def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users_cases WHERE case_id = ?", (case_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        case = dict(row)
        if case['embedding']:
            case['embedding'] = pickle.loads(case['embedding'])
        return case
    return None


# ---------------------------
# Update embedding for a case
# ---------------------------
def update_case_embedding(case_id: str, embedding):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    embedding_blob = pickle.dumps(embedding)
    cursor.execute(
        "UPDATE users_cases SET embedding = ? WHERE case_id = ?",
        (embedding_blob, case_id)
    )

    conn.commit()
    conn.close()


# ---------------------------
# Update status and optional match location
# ---------------------------
def update_case_status(case_id: str, status: str, match_place: Optional[str] = None):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    if match_place:
        cursor.execute(
            "UPDATE users_cases SET status = ?, match_place = ? WHERE case_id = ?",
            (status, match_place, case_id)
        )
    else:
        cursor.execute(
            "UPDATE users_cases SET status = ? WHERE case_id = ?",
            (status, case_id)
        )

    conn.commit()
    conn.close()


# ---------------------------
# Retrieve pending cases with embeddings
# ---------------------------
def get_pending_cases() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM users_cases 
        WHERE status NOT IN ('COMPLETED', 'PERSON FOUND')
        AND embedding IS NOT NULL
    """)

    rows = cursor.fetchall()
    conn.close()

    cases = []
    for row in rows:
        case = dict(row)
        if case['embedding']:
            case['embedding'] = pickle.loads(case['embedding'])
        cases.append(case)

    return cases


# ---------------------------
# Create a match record
# ---------------------------
def create_match(case_id: str, frame_path: str, timestamp: str, similarity: float):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO matches (case_id, frame_path, timestamp, similarity)
        VALUES (?, ?, ?, ?)
    """, (case_id, frame_path, timestamp, similarity))

    conn.commit()
    conn.close()


# ---------------------------
# Retrieve matches for a specific case
# ---------------------------
def get_matches_for_case(case_id: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM matches 
        WHERE case_id = ?
        ORDER BY similarity DESC
    """, (case_id,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


# ---------------------------
# Retrieve all cases
# ---------------------------
def get_all_cases() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users_cases ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


# ---------------------------
# Initialize database on module load
# ---------------------------
init_database()
