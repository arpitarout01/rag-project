# app/utils.py
import os
import time
from typing import List, Tuple
import requests

import numpy as np

# vectorstore loader (langchain-community wrappers)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# optional: transformers for local generation
from transformers import pipeline

# optional: OpenAI
from openai import OpenAI

# ---------- Vectorstore & Retriever ----------

def load_vectorstore(index_path: str = "vectorstore/faiss_index",
                     hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load saved FAISS index and return a vectorstore object and the embeddings model used.
    WARNING: we set allow_dangerous_deserialization=True because the index was created locally.
    Only enable this if you trust the index file (you created it).
    """
    embeddings = HuggingFaceEmbeddings(model_name=hf_model)
    # allow_dangerous_deserialization=True is required to unpickle the saved index metadata
    vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vs, embeddings


def get_retriever(vectorstore, k: int = 4):
    """
    Return a simple retriever function that returns top-k docs for a query.
    """
    # vectorstore.as_retriever exists in this wrapper
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

# ---------- Generators: OpenAI and Local HF ----------

# new OpenAIGenerator for openai>=1.0.0
import os
from openai import OpenAI

class OpenAIGenerator:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", temperature: float = 0.0):
        """
        Uses the new openai>=1.0.0 client.
        """
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var or pass api_key.")
        # create a client instance
        self.client = OpenAI(api_key=key)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        # call the chat completions API using the new client interface
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=512,
        )

        # extract content robustly (supports attribute and dict-like)
        try:
            content = resp.choices[0].message.content
        except Exception:
            content = resp["choices"][0]["message"]["content"]

        return content.strip()

class GroqGenerator:
    """
    Simple Groq API chat client using the OpenAI-compatible endpoint.
    Default model: 'mixtral-8x7b' (change if you prefer another model available in your Groq project).
    """
    def __init__(self, api_key: str = None, model: str = "mixtral-8x7b", temperature: float = 0.0):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not provided. Set GROQ_API_KEY env var or pass api_key.")
        self.model = model
        self.temperature = temperature
        # Groq exposes an OpenAI-compatible base URL
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 512,
        }
        resp = requests.post(self.url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        j = resp.json()
        # Groq uses the same response shape as OpenAI-compatible APIs
        return j["choices"][0]["message"]["content"].strip()

class LocalHFGenerator:
    def __init__(self, model_name: str = "google/flan-t5-small", device: int = -1):
        # device = -1 => CPU. Set device=0 for GPU if available.
        # Using pipeline with text2text-generation for T5/Flan
        self.pipe = pipeline("text2text-generation", model=model_name, device=device)

    def generate(self, prompt: str) -> str:
        out = self.pipe(prompt, max_length=256, do_sample=False)
        return out[0]["generated_text"].strip()

# ---------- RAG query pipeline ----------

def build_prompt(context_passages: List[str], question: str) -> str:
    """
    Build a simple prompt that instructs the generator to answer using only provided context.
    """
    ctx = "\n\n====\n\n".join(context_passages)
    prompt = (
        "You are a helpful assistant. Use ONLY the context below to answer the question. "
        "If the answer is not contained in the context, say \"I don't know\" or respond that the information is not available.\n\n"
        f"CONTEXT:\n{ctx}\n\nQUESTION: {question}\n\nAnswer:"
    )
    return prompt

def retrieve_and_generate(question: str,
                          retriever,
                          generator,
                          k: int = 4,
                          verbose: bool = True) -> Tuple[str, List[Tuple[float, dict]]]:
    """
    Retrieve top-k docs and generate an answer.
    Robust to different retriever implementations and signatures.
    """
    import inspect

    t0 = time.perf_counter()

    docs = None
    # 1) preferred: public API
    if hasattr(retriever, "get_relevant_documents"):
        try:
            docs = retriever.get_relevant_documents(question)
        except TypeError:
            # fallback to inspect below
            docs = None

    # 2) try _get_relevant_documents with/without run_manager
    if docs is None and hasattr(retriever, "_get_relevant_documents"):
        sig = inspect.signature(retriever._get_relevant_documents)
        params = list(sig.parameters.values())
        # if only single positional param (question)
        if len(params) == 1:
            docs = retriever._get_relevant_documents(question)
        else:
            # many implementations expect run_manager as kw-only arg
            try:
                docs = retriever._get_relevant_documents(question, run_manager=None)
            except TypeError:
                # safe final fallback: try calling with no args
                try:
                    docs = retriever._get_relevant_documents()
                except Exception as e:
                    raise RuntimeError("Unable to call _get_relevant_documents: " + str(e))

    # 3) try retrieve()
    if docs is None and hasattr(retriever, "retrieve"):
        try:
            docs = retriever.retrieve(question)
        except TypeError:
            try:
                docs = retriever.retrieve(question, top_k=k)
            except Exception as e:
                raise RuntimeError("Retriever.retrieve failed: " + str(e))

    if docs is None:
        raise AttributeError(
            "The retriever object does not have a known retrieval method. "
            "Expected one of: get_relevant_documents, _get_relevant_documents, retrieve."
        )

    retr_time = time.perf_counter() - t0

    # Extract passages and metadata — doc objects may have .page_content and .metadata
    passages = [getattr(d, "page_content", str(d)) for d in docs][:k]
    metadatas = [getattr(d, "metadata", {}) for d in docs][:k]

    # Build prompt from passages
    prompt = build_prompt(passages, question)

    t1 = time.perf_counter()
    answer = generator.generate(prompt)
    gen_time = time.perf_counter() - t1

    if verbose:
        print(f"[retrieval_time: {retr_time:.3f}s] [generation_time: {gen_time:.3f}s]")
        print("Top retrieved sources:")
        for i, (m, p) in enumerate(zip(metadatas, passages), start=1):
            src = m.get("source", "unknown") if isinstance(m, dict) else "unknown"
            chunk = m.get("chunk", "?") if isinstance(m, dict) else "?"
            print(f" {i}. {src} (chunk {chunk}) — {len(p)} chars")

    paired = [(None, {**(m if isinstance(m, dict) else {}), "text": p}) for m, p in zip(metadatas, passages)]
    return answer, paired


