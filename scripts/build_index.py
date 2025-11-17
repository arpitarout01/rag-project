# scripts/build_index.py

import pandas as pd
import os

# Correct imports for new LangChain versions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load cleaned CSV
df = pd.read_csv("data/cleaned.csv")

# Split docs into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

texts = []
metadatas = []

for _, row in df.iterrows():
    chunks = splitter.split_text(row["text"])
    for i, chunk in enumerate(chunks):
        texts.append(chunk)
        metadatas.append({
            "source": row["doc_id"],
            "chunk": i
        })

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS store
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# Save index
os.makedirs("vectorstore", exist_ok=True)
vectorstore.save_local("vectorstore/faiss_index")

print("FAISS index created and saved to vectorstore/faiss_index")
print("Total chunks stored:", len(texts))
