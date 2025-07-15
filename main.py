import re
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from starlette.middleware.cors import CORSMiddleware

model = joblib.load("logistic_model1.pkl")
# Actual prediction route
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://spamdetectorfrontend-git-main-rusira-dinujayas-projects.vercel.app",
"https://spamdetectorfrontend.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define input schema
class EmailInput(BaseModel):
    text: str



# --- WORDS from SpamBase (first 48 features)
word_list = [
    'make', 'all', 'our', 'over', 'remove', 'internet', 'order',
    'mail', 'receive', 'people', 'addresses', 'free', 'business',
    'email', 'you', 'credit', 'your', '000', 'money', 'hp', 'hpl', 'george',
    '650', 'lab', 'labs', 'telnet', '857', 'data', '415', '85', 'technology', '1999',
    'pm', 'meeting', 'original','re', 'edu'
]

# --- CHARACTERS from SpamBase (next 6 features)
char_list = ['!', '$']

# --- Function to calculate word frequency
def word_freq(text, word):
    words = re.findall(r'\b\w+\b', text.lower())
    return 100 * words.count(word) / len(words) if words else 0

# --- Function to calculate character frequency
def char_freq(text, char):
    return 100 * text.count(char) / len(text) if text else 0

# --- Capital letter statistics (last 3 features)
def capital_run_stats(text):
    runs = re.findall(r'[A-Z]+', text)
    lengths = [len(run) for run in runs]
    if lengths:
        return np.mean(lengths), np.max(lengths), sum(lengths)
    else:
        return 0, 0, 0

# --- Full extraction function
def extract_spambase_features(text):
    word_features = [word_freq(text, word) for word in word_list]
    char_features = [char_freq(text, char) for char in char_list]
    avg_cap_run, max_cap_run, total_cap_run = capital_run_stats(text)
    cap_features = [avg_cap_run, max_cap_run, total_cap_run]
    return word_features + char_features + cap_features

@app.post("/predict")
def predict_spam(email: EmailInput):
    features = [extract_spambase_features(email.text)]
    prob_spam = model.predict_proba(features)[0][1]  # Confidence
    threshold = 0.55
    is_spam = int(prob_spam >= threshold)

    return {
        "prediction": "spam" if is_spam else "not spam"
    }


