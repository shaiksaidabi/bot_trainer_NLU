import streamlit as st
import requests
import pandas as pd
import io
import sys
import os
import json

# Add backend folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
from database import get_connection, init_db

# Initialize database
init_db()

# ----------------------------
# 🌐 Backend URL
# ----------------------------
BACKEND_URL = "http://127.0.0.1:8000"

# ----------------------------
# 🔐 Login / Register Page
# ----------------------------
def login_page():
    st.title("🔑 Login / Register")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if not username or not password:
                st.warning("Please enter username and password")
            else:
                res = requests.post(f"{BACKEND_URL}/login", data={"username": username, "password": password})
                if res.status_code == 200:
                    st.session_state.token = res.json()["token"]
                    st.session_state.username = username
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error(res.json().get("detail", "Login failed"))

    with tab2:
        st.subheader("Register")
        username = st.text_input("New Username", key="reg_user")
        password = st.text_input("New Password", type="password", key="reg_pass")

        if st.button("Register"):
            res = requests.post(f"{BACKEND_URL}/register", data={"username": username, "password": password})
            if res.status_code == 200:
                st.success("✅ Registered successfully! You can now login.")
            else:
                st.error(res.json().get("detail", "Registration failed"))

# ----------------------------
# 🧠 Workspace Tabs
# ----------------------------
def workspace():
    st.title("💬 NLU Chatbot Trainer Workspace")
    st.write(f"👋 Welcome, **{st.session_state.username}**")
    tabs = st.tabs(["📂 Upload Dataset", "🧾 Annotate", "🤖 Train & Test"])

    # --- 1️⃣ Upload Dataset ---
    with tabs[0]:
        st.subheader("Upload Dataset")
        bot_name = st.text_input("Bot Name")
        dataset = st.file_uploader("Upload your dataset (CSV/JSON)", type=["csv", "json"])
        if st.button("Upload"):
            if not bot_name or not dataset:
                st.warning("Please provide bot name and dataset file.")
            else:
                data = {"name": bot_name, "username": st.session_state.username}
                res = requests.post(f"{BACKEND_URL}/create_bot", data=data, files={"file": dataset})
                if res.status_code == 200:
                    st.success("✅ Bot created successfully and dataset uploaded!")
                    st.session_state.dataset_name = dataset.name
                else:
                    st.error(res.json().get("detail", "Upload failed"))

    # --- 2️⃣ Annotate ---
    with tabs[1]:
        st.subheader("🧠 Annotation & Model Integration")
        res = requests.get(f"{BACKEND_URL}/bots", params={"username": st.session_state.username})
        if res.status_code != 200 or not res.json():
            st.warning("⚠️ No bots found. Please create one in the Upload tab first.")
            return

        bots = res.json()
        bot_names = [b["name"] for b in bots]
        selected_bot = st.selectbox("Select Bot", bot_names)
        bot_id = next(b["id"] for b in bots if b["name"] == selected_bot)
        st.session_state.bot_id = bot_id

        # Select sentence
        st.markdown("### ✍️ Text Annotation")
        dataset_preview = requests.get(f"{BACKEND_URL}/dataset_preview/{bot_id}")
        sentences = [row.get("question") or row.get("sentence") for row in dataset_preview.json()] if dataset_preview.status_code == 200 else []
        selected_sentence = st.selectbox("Select sentence from dataset:", sentences)
        sentence = st.text_area("Sentence", selected_sentence, height=70)

        # Annotate button
        if st.button("🔍 Annotate"):
            if not sentence.strip():
                st.warning("Please enter or select a sentence.")
            else:
                res = requests.post(f"{BACKEND_URL}/annotate", data={"sentence": sentence, "bot_id": bot_id})
                if res.status_code == 200:
                    result = res.json()
                    intent = result["intent"]
                    entities = result["entities"]

                    st.success(f"🎯 Intent: {intent}")
                    st.markdown("🏷️ Entities found:")
                    st.json(entities)

                    # --- Save annotation to DB ---
                    conn = get_connection()
                    cur = conn.cursor()
                    cur.execute(
                        "INSERT INTO annotations (bot_id, sentence, intent, entities) VALUES (?, ?, ?, ?)",
                        (bot_id, sentence, intent, json.dumps(entities))
                    )
                    conn.commit()
                    conn.close()
                    st.success("✅ Annotation saved to database!")

                    # --- Smart Intent Highlight ---
                    sentence_lower = sentence.lower()
                    intent_keywords = {
                        "book_flight": ["book", "flight", "ticket", "plane"],
                        "check_weather": ["weather", "temperature", "rain", "forecast"],
                        "find_restaurant": ["restaurant", "food", "eat", "dinner", "lunch"]
                    }
                    active_intents = [i for i, words in intent_keywords.items() if any(w in sentence_lower for w in words)]

                    # --- Normalize entity labels ---
                    entity_mapping = {"gpe": "location", "loc": "location", "org": "organization", "date": "date", "person": "person", "time": "date"}
                    entity_labels = [entity_mapping.get(e["label"].lower(), e["label"].lower()) for e in entities]

                    # --- Render Intent Buttons ---
                    st.markdown("### 🎯 Select Intent")
                    intent_cols = st.columns(3)
                    for idx, i in enumerate(["book_flight", "check_weather", "find_restaurant"]):
                        color = "#6ee7b7" if i in active_intents else "#f0f0f0"
                        intent_cols[idx].markdown(
                            f"<div style='background-color:{color};padding:10px;border-radius:8px;text-align:center;'>🧠 {i}</div>",
                            unsafe_allow_html=True
                        )

                    # --- Render Entity Buttons ---
                    st.markdown("### 🧩 Entity Tools")
                    entity_cols = st.columns(4)
                    for idx, e_type in enumerate(["location", "date", "person", "organization"]):
                        color = "#93c5fd" if e_type in entity_labels else "#f0f0f0"
                        entity_cols[idx].markdown(
                            f"<div style='background-color:{color};padding:10px;border-radius:8px;text-align:center;'>🔹 {e_type}</div>",
                            unsafe_allow_html=True
                        )

    # --- 3️⃣ Train & Test ---
    with tabs[2]:
        st.subheader("Train & Test Bot")
        st.info("🚀 Training placeholder – ML model integration comes later.")
        dataset = st.file_uploader("Upload dataset to train model", type=["csv"])

        if st.button("Train"):
            if not dataset:
                st.warning("Please upload a dataset file.")
            else:
                res = requests.post(f"{BACKEND_URL}/train_bot", files={"dataset": dataset})
                if res.status_code == 200:
                    st.success(f"✅ Model trained successfully! Accuracy: {res.json()['accuracy']}%")
                else:
                    st.error("Training failed.")

        st.write("---")
        st.subheader("Test Bot Response")
        message = st.text_input("💬 Enter a message for your bot")
        if st.button("Send Message"):
            if not message:
                st.warning("Please enter a message.")
            else:
                st.info("🤖 (Simulated) Bot Reply: *Response from model will appear here later.*")


# ----------------------------
# 🧭 App Entry
# ----------------------------
def main():
    st.set_page_config(page_title="Chatbot Trainer", layout="wide")
    if "token" not in st.session_state:
        login_page()
    else:
        workspace()

if __name__ == "__main__":
    main()
