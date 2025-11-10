from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from backend.database import get_connection, init_db
import sqlite3
import os
import shutil
import secrets
import pandas as pd
import spacy
import io
import json

app = FastAPI()
init_db()

# --- Allow frontend access ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- REGISTER ----------------
@app.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return {"message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    finally:
        conn.close()


# ---------------- LOGIN ----------------
@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cur.fetchone()
    conn.close()

    if user:
        token = secrets.token_hex(16)
        return {"token": token, "username": username}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")


# ---------------- CREATE BOT ----------------
@app.post("/create_bot")
def create_bot(name: str = Form(...), file: UploadFile = File(...), username: str = Form(...)):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("INSERT INTO bots (name, owner_username) VALUES (?, ?)", (name, username))
    bot_id = cur.lastrowid

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cur.execute(
        "INSERT INTO datasets (bot_id, filename, owner_username) VALUES (?, ?, ?)",
        (bot_id, file.filename, username),
    )
    conn.commit()
    conn.close()

    return {"message": "Bot created successfully", "bot_id": bot_id}


# ---------------- FETCH DATASET PREVIEW ----------------
@app.get("/dataset_preview/{bot_id}")
def dataset_preview(bot_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT filename FROM datasets WHERE bot_id=?", (bot_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = os.path.join(UPLOAD_DIR, row[0])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    df = pd.read_csv(file_path)
    return df.head(10).to_dict(orient="records")


# ---------------- ANNOTATE SENTENCE ----------------
@app.post("/annotate")
def annotate(sentence: str = Form(...), bot_id: int = Form(...)):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT filename FROM datasets WHERE bot_id=?", (bot_id,))
        row = cur.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Dataset not found for this bot")

        dataset_path = os.path.join(UPLOAD_DIR, row[0])
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset file missing")

        
        try:
          df = pd.read_csv(dataset_path, encoding='utf-8')
        except Exception:
            raise HTTPException(status_code=400, detail="Could not read CSV — check file format or encoding.")

        if df.empty:
            raise HTTPException(status_code=400, detail="Empty dataset file uploaded")

        possible_cols = [col.lower() for col in df.columns]
        if "question" not in possible_cols and "sentence" not in possible_cols:
            raise HTTPException(status_code=400, detail="Dataset must contain 'question' or 'sentence' column")

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence)

        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        column_name = "question" if "question" in df.columns else "sentence"
        matching_intents = [
            q for q in df[column_name].astype(str).tolist() if any(word.lower() in q.lower() for word in sentence.split())
        ]
        intent = matching_intents[0] if matching_intents else "Unknown"

        return {"intent": intent, "entities": entities}

    except Exception as e:
        print("ANNOTATE ERROR:", e)
        raise HTTPException(status_code=500, detail=f"Error during annotation: {str(e)}")


# ---------------- SAVE ANNOTATION ----------------
@app.post("/save_annotation")
def save_annotation(workspace_name: str = Form(...), text: str = Form(...),
                    intent: str = Form(...), entities: str = Form(...)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_name TEXT,
            text TEXT,
            intent TEXT,
            entities TEXT
        )
    """)
    cur.execute("INSERT INTO annotations (workspace_name, text, intent, entities) VALUES (?, ?, ?, ?)",
                (workspace_name, text, intent, entities))
    conn.commit()
    conn.close()
    return {"message": "Annotation saved successfully"}
# ---------------- FETCH USER BOTS ----------------
@app.get("/bots")
def get_bots(username: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM bots WHERE owner_username=?", (username,))
    bots = [{"id": row[0], "name": row[1]} for row in cur.fetchall()]
    conn.close()
    return bots
@app.post("/save_annotation")
async def save_annotation(data: dict):
    import sqlite3
    conn = sqlite3.connect("chatbot.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER,
            sentence TEXT,
            intent TEXT,
            entities TEXT
        )
    """)
    
    bot_id = data.get("bot_id")
    sentence = data.get("sentence")
    intent = data.get("intent")
    entities = json.dumps(data.get("entities", []))  # Store as JSON

    c.execute(
        "INSERT INTO annotations (bot_id, sentence, intent, entities) VALUES (?, ?, ?, ?)",
        (bot_id, sentence, intent, entities)
    )
    conn.commit()
    conn.close()

    return {"message": "Annotation saved successfully!"}
