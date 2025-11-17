# scripts/ingest.py

import os
import glob
import pandas as pd
import pdfplumber

def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def clean(text):
    text = text.replace("\r", "\n")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()

def ingest():
    input_folder = "data/sample_faqs"
    rows = []

    for file in glob.glob(os.path.join(input_folder, "*")):
        name = os.path.basename(file)

        if file.lower().endswith(".txt"):
            raw = read_txt(file)
        elif file.lower().endswith(".pdf"):
            raw = read_pdf(file)
        else:
            continue

        cleaned = clean(raw)
        rows.append({"doc_id": name, "text": cleaned})

    df = pd.DataFrame(rows)
    df.to_csv("data/cleaned.csv", index=False)
    print(f"Saved {len(df)} documents to data/cleaned.csv")

if __name__ == "__main__":
    ingest()
