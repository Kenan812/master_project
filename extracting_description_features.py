# GETTING FEATURES FROM THE DESCRIPTION

import ast
import pandas as pd
import numpy as np
import torch
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Ensure stopwords and lemmatizer resources are available
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")  # for NER
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Downloading...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")  # try loading again

# Initialize summarization pipeline
device_num = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device_num)

# Set device for SentenceTransformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load metadata
metadata = pd.read_csv('datasets/metadata.csv')

def parse_description(desc):
    if isinstance(desc, str):
        desc = desc.strip()
        if desc == '[]':
            return ""
        try:
            parsed = ast.literal_eval(desc)
            if isinstance(parsed, list):
                full_text = " ".join(str(item) for item in parsed)
            else:
                full_text = str(parsed)
        except (ValueError, SyntaxError):
            full_text = desc
    else:
        return ""
    return full_text

# 1) Parse descriptions
print("Parsing descriptions...")
metadata['description_text'] = metadata['description'].apply(parse_description)

# 2) Basic cleaning: tokenization, remove duplicates, stopwords, lemmatization
def clean_text(text):
    if not text:
        return ""
    words = text.split()
    # Remove duplicates by converting to a set, then back to list (order lost)
    words = list(set(words))
    # Remove stopwords
    words = [w for w in words if w.lower() not in stop_words]
    # Lemmatize words
    words = [lemmatizer.lemmatize(w.lower()) for w in words]
    return " ".join(words)

print("Cleaning text...")
metadata['cleaned_text'] = metadata['description_text'].apply(clean_text)

# 3) Extract Keywords Using TF-IDF
print("Extracting keywords using TF-IDF...")
corpus = metadata['cleaned_text'].tolist()
tfidf = TfidfVectorizer(max_features=5000)  # limit to top 5000 features overall
tfidf_matrix = tfidf.fit_transform(corpus)
feature_names = np.array(tfidf.get_feature_names_out())

def extract_keywords(doc_idx, top_n=20):
    # For each document, get top N tf-idf terms
    row = tfidf_matrix[doc_idx].toarray().flatten()
    top_indices = row.argsort()[::-1][:top_n]
    return " ".join(feature_names[top_indices])

# Use tqdm to show progress
print("Applying TF-IDF keyword extraction...")
metadata['keywords_text'] = [
    extract_keywords(i, top_n=20) for i in tqdm(range(len(metadata)), desc="TF-IDF Keywords")
]

# 4) Summarization
print("Summarizing keywords...")

def summarize_batch(texts, max_length=50, min_length=10):
    if not texts:
        return []
    summaries = summarizer(texts, max_length=max_length, min_length=min_length, do_sample=False)
    return [summary['summary_text'] for summary in summaries]

# Split texts into batches for summarization
batch_size = 8  # Adjust based on your GPU memory
summarized_texts = []
keywords_texts = metadata['keywords_text'].tolist()

print("Applying summarization in batches...")
for i in tqdm(range(0, len(keywords_texts), batch_size), desc="Summarization"):
    batch_texts = keywords_texts[i:i+batch_size]
    # Split batch into texts to summarize and texts to keep
    texts_to_summarize = []
    indices_to_summarize = []
    for idx, text in enumerate(batch_texts):
        word_count = len(text.split())
        if word_count > 20:  # threshold for summarization
            texts_to_summarize.append(text)
            indices_to_summarize.append(idx)
    # Summarize texts_to_summarize
    if texts_to_summarize:
        summaries = summarize_batch(texts_to_summarize, max_length=50, min_length=10)
    else:
        summaries = []
    # Combine summaries and texts_to_keep
    batch_summarized = batch_texts.copy()
    for idx, summary in zip(indices_to_summarize, summaries):
        batch_summarized[idx] = summary
    summarized_texts.extend(batch_summarized)

metadata['summarized_text'] = summarized_texts

# 5) Named Entity Recognition (NER)
print("Applying Named Entity Recognition (NER)...")

def keep_named_entities(text):
    if not text:
        return text
    doc = nlp(text)
    # Extract all unique entity texts
    entities = list(set([ent.text for ent in doc.ents]))
    # If no entities found, return original text
    if not entities:
        return text
    # Otherwise, return just the named entities joined
    return " ".join(entities)

# Apply NER with progress bar
print("Filtering text based on named entities...")
metadata['entity_filtered_text'] = [
    keep_named_entities(text) for text in tqdm(metadata['summarized_text'], desc="NER Filtering")
]

# 6) Final Tokenization and Vectorization with Sentence-BERT
print("Generating Sentence-BERT embeddings...")
final_texts = metadata['entity_filtered_text'].tolist()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

# Encode in batches with progress bar
embeddings = model.encode(
    final_texts,
    batch_size=512,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Get parent_asins to map them
parent_asins = metadata['parent_asin'].values

# Create a dictionary {parent_asin: embedding_vector}
features_dict = {pid: emb for pid, emb in zip(parent_asins, embeddings)}

# Save to npz
np.savez_compressed('description_features.npz', **features_dict)

print(f"Saved embeddings for {len(features_dict)} products to 'description_features.npz'.")