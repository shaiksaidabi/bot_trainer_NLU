import pandas as pd
import spacy
import numpy as np

nlp = spacy.load("en_core_web_md")

class ChatBotModel:
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        self.questions = self.df['question'].tolist()
        self.answers = self.df['answer'].tolist()
        self.question_vecs = [nlp(q).vector for q in self.questions]

    def get_response(self, user_input):
        user_vec = nlp(user_input).vector
        sims = [self._cosine_similarity(user_vec, qv) for qv in self.question_vecs]
        best_idx = sims.index(max(sims))
        return self.answers[best_idx]

    def _cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

def train_bot(dataset_path):
    return ChatBotModel(dataset_path)

def get_bot_response(model, user_input):
    return model.get_response(user_input)
import random
from spacy.training import Example

# ---------- NEW FUNCTION: Annotate a sentence ----------
def annotate_sentence(sentence):
    """
    Analyzes a sentence using spaCy and returns detected entities.
    Example:
      Input: "Book flight from Delhi to Jaipur on Jan 15th"
      Output: {"text": ..., "entities": [{"text": "Delhi", "label": "GPE"}, ...]}
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return {"text": sentence, "entities": entities}


# ---------- NEW FUNCTION: Train spaCy model on annotated dataset ----------
def train_spacy_model(dataset_path):
    """
    Trains a simple spaCy NER model from an annotated dataset (CSV/JSON).
    The dataset must have columns: 'text' and 'entities',
    where entities is a list of tuples [(start, end, label), ...].
    """
    import os
    import pandas as pd

    # Load dataset
    df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_json(dataset_path)

    # Validate dataset
    if "text" not in df.columns or "entities" not in df.columns:
        raise ValueError("Dataset must have 'text' and 'entities' columns")

    TRAIN_DATA = []
    for _, row in df.iterrows():
        if isinstance(row["entities"], str):
            # Convert from string representation to list
            try:
                row["entities"] = eval(row["entities"])
            except:
                continue
        TRAIN_DATA.append((row["text"], {"entities": row["entities"]}))

    if not TRAIN_DATA:
        raise ValueError("No valid annotated data found for training")

    # Initialize model
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")

    # Add entity labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations["entities"]:
            ner.add_label(ent[2])

    # Train model
    optimizer = nlp.begin_training()
    for i in range(10):  # 10 epochs
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.2, sgd=optimizer, losses=losses)
        print(f"Epoch {i+1} Losses: {losses}")

    # Save model
    model_dir = "backend/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "trained_model")
    nlp.to_disk(model_path)

    # Dummy accuracy (for now)
    accuracy = round(100 - losses.get("ner", 0) / len(TRAIN_DATA), 2)
    return model_path, accuracy
