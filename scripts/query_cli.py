# scripts/query_cli.py
"""
Query CLI for the RAG demo.

Usage examples:
  # Local HF generator (no API keys)
  python -m scripts.query_cli --model local --question "What is the return policy?"

  # OpenAI (make sure OPENAI_API_KEY is set in this session)
  python -m scripts.query_cli --model openai --question "What is the return policy?"

  # Groq (make sure GROQ_API_KEY is set in this session)
  python -m scripts.query_cli --model groq --question "What is the return policy?"
"""

import os
import argparse
import textwrap

# Import generators + utilities from app
from app.utils import (
    load_vectorstore,
    get_retriever,
    OpenAIGenerator,
    GroqGenerator,
    LocalHFGenerator,
    retrieve_and_generate,
)

def main():
    parser = argparse.ArgumentParser(
        description="Query the RAG system (supports openai, groq, local generators)."
    )
    parser.add_argument("--model", choices=["openai", "groq", "local"], default="local",
                        help="Generator to use")
    parser.add_argument("--question", type=str, required=True, help="Question to ask")
    parser.add_argument("--k", type=int, default=4, help="Number of passages to retrieve")
    parser.add_argument("--index", type=str, default="vectorstore/faiss_index", help="Path to FAISS index")
    args = parser.parse_args()

    # Load vectorstore
    print(f"Loading vectorstore from: {args.index} ...")
    vs, embeddings = load_vectorstore(args.index)
    retriever = get_retriever(vs, k=args.k)

    # Choose generator
    if args.model == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY not set. Set it in your session before running.")
        gen = OpenAIGenerator(api_key=api_key)
        print("Using OpenAI generator.")
    elif args.model == "groq":
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise SystemExit("GROQ_API_KEY not set. Set it in your session before running.")
        groq_model = os.getenv("GROQ_MODEL", "mixtral-8x7b")
        gen = GroqGenerator(api_key=groq_key, model=groq_model)
        print(f"Using Groq generator with model: {groq_model}")
    else:
        # Local HF generator (CPU by default)
        hf_model = os.getenv("LOCAL_HF_MODEL", "google/flan-t5-small")
        gen = LocalHFGenerator(model_name=hf_model, device=-1)
        print(f"Using Local HF generator: {hf_model} (device=CPU)")

    # Run retrieval + generation
    print("\nQuestion:")
    print(textwrap.fill(args.question, width=100))
    print("\nRetrieving and generating...\n")
    answer, retrieved = retrieve_and_generate(args.question, retriever, gen, k=args.k)

    print("\n\n=== GENERATED ANSWER ===\n")
    print(answer)
    print("\n\n=== RETRIEVED PASSAGES ===\n")
    for i, pair in enumerate(retrieved, 1):
        meta = pair[1]
        src = meta.get("source", "unknown")
        chunk = meta.get("chunk", "?")
        text = meta.get("text", "")
        print(f"--- Passage {i} (source: {src}, chunk: {chunk}) ---")
        print(textwrap.fill(text[:1200], width=100))
        print("\n")

if __name__ == "__main__":
    main()
