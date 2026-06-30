import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
CHUNKS_FILE = Path("data/processed/text_chunks.jsonl")
OUTPUT_DIR = Path("data/processed")
EMBEDDINGS_FILE = OUTPUT_DIR / "text_embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "text_metadata.jsonl"
MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_DIR = Path("data/extracted_images")
IMAGE_EMBEDDINGS_FILE = OUTPUT_DIR / "image_embeddings.npy"
IMAGE_METADATA_FILE = OUTPUT_DIR / "image_metadata.jsonl"

def load_chunks(path):
    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks

def load_model():
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model

def embed_texts(chunks, processor, model):
    texts = [chunk["text"] for chunk in chunks]

    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    if hasattr(text_features, "pooler_output"):
        text_features = text_features.pooler_output

    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    return text_features.cpu().numpy()

def save_metadata(chunks, output_path):
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            metadata = {
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"]
            }
            file.write(json.dumps(metadata) + "\n")

def load_image_paths(image_dir):
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]

    image_paths = []

    for extension in image_extensions:
        image_paths.extend(image_dir.glob(extension))

    return image_paths

def embed_images(image_paths, processor, model):
    images = []

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        images.append(image)

    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    with torch.no_grad():
        image_features = model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )

    if hasattr(image_features, "pooler_output"):
        image_features = image_features.pooler_output

    image_features = image_features / image_features.norm(dim=1, keepdim=True)

    return image_features.cpu().numpy()

def save_image_metadata(image_paths, output_path):
    with output_path.open("w", encoding="utf-8") as file:
        for image_index, image_path in enumerate(image_paths):
            metadata = {
                "image_id": image_index,
                "image_path": str(image_path),
                "filename": image_path.name
            }

            file.write(json.dumps(metadata) + "\n")

#test
if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_FILE)

    print(f"Chunks loaded: {len(chunks)}")

    processor, model = load_model()

    text_embeddings = embed_texts(chunks, processor, model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDINGS_FILE, text_embeddings)
    save_metadata(chunks, METADATA_FILE)

    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Saved text embeddings to: {EMBEDDINGS_FILE}")
    print(f"Saved text metadata to: {METADATA_FILE}")

    image_paths = load_image_paths(IMAGE_DIR)

    if not image_paths:
        print("No extracted images found in data/extracted_images/")
    else:
        image_embeddings = embed_images(image_paths, processor, model)

        np.save(IMAGE_EMBEDDINGS_FILE, image_embeddings)
        save_image_metadata(image_paths, IMAGE_METADATA_FILE)

        print(f"Images loaded: {len(image_paths)}")
        print(f"Image embeddings shape: {image_embeddings.shape}")
        print(f"Saved image embeddings to: {IMAGE_EMBEDDINGS_FILE}")
        print(f"Saved image metadata to: {IMAGE_METADATA_FILE}")