Retrieval-Augmented Generation (RAG) Project — Partial Submission

This repository contains a partial implementation of a Retrieval-Augmented Generation (RAG) system, developed as part of the AI/ML Intern assignment.
The objective is to demonstrate understanding of retrieval pipelines, vector search, embeddings, and integration with LLMs.

Project Overview

RAG (Retrieval-Augmented Generation) combines:

Retriever → fetches relevant text chunks from a knowledge base

Generator (LLM) → produces a final response using retrieved context

This project implements:

Data ingestion

Cleaning and preprocessing

Chunking

Embeddings

FAISS vector store

Retriever pipeline

Query CLI for testing

Completed Components (Included in This Submission)
1. Data Collection & Preprocessing

Created custom FAQ text files.

Cleaned and merged them into a single dataset (cleaned.csv).

Preprocessing includes: lowercasing, whitespace removal, dropping empty rows.

2. Document Chunking

Used RecursiveCharacterTextSplitter for chunking.

Ensures better embedding quality and retrieval accuracy.

3. Embeddings & Vectorstore

Embeddings generated using HuggingFace sentence-transformer model.

Built a FAISS vector index (vectorstore/faiss_index).

Successfully saved and loaded the vectorstore.

4. Retrieval Pipeline

Implemented retriever using FAISS + top-k similarity search.

Retrieval tested with sample queries.

5. Query CLI

Built query_cli.py for querying the system.

Supports:

Local HuggingFace models

OpenAI (working, but quota exhausted)

Groq (integrated but not fully functional)

Pending (Not Included in This Partial Submission)

Final working generation with cloud LLM (OpenAI quota & Groq endpoint issues)

Streamlit UI / Frontend chatbot

FastAPI-based RAG API

Full evaluation metrics (latency, accuracy, retrieval scoring)

Project Structure
rag-project/
│── app/
│   ├── __init__.py
│   ├── utils.py
│── data/
│   ├── raw/
│   ├── processed/
│   ├── cleaned.csv
│── vectorstore/
│   ├── faiss_index
│   ├── faiss_index.pkl
│── scripts/
│   ├── ingest.py
│   ├── build_index.py
│   ├── query_cli.py
│── README.md
│── requirements.txt

How to Run (Completed Parts)
1. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

2. Install dependencies
pip install -r requirements.txt

3. Ingest data
python scripts/ingest.py

4. Build index
python scripts/build_index.py

5. Query (Local model – no API needed)
python -m scripts.query_cli --model local --question "What is the return policy?"

Notes

OpenAI generator was tested and authenticated successfully, but the account has insufficient quota, so cloud generation is not included.

Groq generator integrated, but model-ID and endpoint returned 404, so left incomplete.

Retrieval pipeline is fully functional and demonstrated using local models.

Submission Status: Partial

This repository contains all completed components required for partial evaluation.