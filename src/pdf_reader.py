from pathlib import Path
import fitz
PDF_DIR = Path("data/raw_pdfs")
IMAGE_DIR = Path("data/extracted_images")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    pages = []

    for page_index, page in enumerate(doc):
        text = page.get_text()

        pages.append({
            "filename": pdf_path.name,
            "page_number": page_index + 1,
            "text": text
        })

    doc.close()
    return pages

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    images = []

    for page_index, page in enumerate(doc):
        image_list = page.get_images(full=True)

        for image_index, image_info in enumerate(image_list):
            xref = image_info[0]
            image_data = doc.extract_image(xref)

            image_bytes = image_data["image"]
            image_ext = image_data["ext"]

            image_filename = (
                f"{pdf_path.stem}_page_{page_index + 1}_img_{image_index + 1}.{image_ext}"
            )

            image_path = IMAGE_DIR / image_filename
            image_path.write_bytes(image_bytes)

            images.append({
                "filename": pdf_path.name,
                "page_number": page_index + 1,
                "image_number": image_index + 1,
                "image_path": str(image_path)
            })

    doc.close()
    return images

#test
if __name__ == "__main__":
    print("Current working folder:", Path.cwd())
    print("Looking for PDFs inside:", PDF_DIR.resolve())

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in data/raw_pdfs/")
    else:
        pdf_path = pdf_files[0]

        text_pages = extract_text_from_pdf(pdf_path)
        images = extract_images_from_pdf(pdf_path)

        print(f"PDF: {pdf_path.name}")
        print(f"Pages extracted: {len(text_pages)}")
        print(f"Images extracted: {len(images)}")

        if text_pages:
            print("\nFirst page text preview:")
            print(text_pages[0]["text"][:500])