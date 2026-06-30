import json
from pathlib import Path

from pdf_reader import extract_text_from_pdf
PDF_DIR = Path("data/raw_pdfs")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "text_chunks.jsonl"

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start = start + chunk_size - overlap

    return chunks

def create_chunks_from_pdf(pdf_path):
    pages = extract_text_from_pdf(pdf_path)

    all_chunks = []

    for page in pages:
        page_chunks = chunk_text(page["text"])

        for chunk_index, chunk in enumerate(page_chunks):
            chunk_record = {
                "filename": page["filename"],
                "page_number": page["page_number"],
                "chunk_id": chunk_index,
                "text": chunk
            }

            all_chunks.append(chunk_record)

    return all_chunks

def save_chunks_to_jsonl(chunks, output_path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk) + "\n")

#test
if __name__ == "__main__":
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in data/raw_pdfs/")
    else:
        pdf_path = pdf_files[0]

        chunks = create_chunks_from_pdf(pdf_path)
        save_chunks_to_jsonl(chunks, OUTPUT_FILE)

        print(f"PDF: {pdf_path.name}")
        print(f"Chunks created: {len(chunks)}")
        print(f"Saved to: {OUTPUT_FILE}")

        if chunks:
            print("\nFirst chunk preview:")
            print(chunks[0]["text"][:500])