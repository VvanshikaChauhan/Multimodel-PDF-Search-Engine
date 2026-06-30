import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPModel, CLIPProcessor
TEXT_EMBEDDINGS_FILE = Path("data/processed/text_embeddings.npy")
TEXT_METADATA_FILE = Path("data/processed/text_metadata.jsonl")

IMAGE_EMBEDDINGS_FILE = Path("data/processed/image_embeddings.npy")
IMAGE_METADATA_FILE = Path("data/processed/image_metadata.jsonl")

MODEL_NAME = "openai/clip-vit-base-patch32"

def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records

def load_model():
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model

def embed_query(query, processor, model):
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        query_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    if hasattr(query_features, "pooler_output"):
        query_features = query_features.pooler_output

    query_features = query_features / query_features.norm(dim=1, keepdim=True)

    return query_features.cpu().numpy()

def search_embeddings(query_embedding, embeddings, metadata, top_k=5):
    scores = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in top_indices:
        result = metadata[index].copy()
        result["score"] = float(scores[index])
        results.append(result)

    return results

#test
if __name__ == "__main__":
    query = "multimodal search engine"

    processor, model = load_model()
    query_embedding = embed_query(query, processor, model)

    text_embeddings = np.load(TEXT_EMBEDDINGS_FILE)
    text_metadata = load_jsonl(TEXT_METADATA_FILE)

    print(f"Query: {query}")
    print("\nTop text results:")

    text_results = search_embeddings(
        query_embedding,
        text_embeddings,
        text_metadata,
        top_k=5
    )

    for result in text_results:
        print("-" * 80)
        print(f"Score: {result['score']:.4f}")
        print(f"File: {result['filename']}")
        print(f"Page: {result['page_number']}")
        print(f"Chunk: {result['chunk_id']}")
        print(result["text"][:500])

    if IMAGE_EMBEDDINGS_FILE.exists() and IMAGE_METADATA_FILE.exists():
        image_embeddings = np.load(IMAGE_EMBEDDINGS_FILE)
        image_metadata = load_jsonl(IMAGE_METADATA_FILE)

        print("\nTop image results:")

        image_results = search_embeddings(
            query_embedding,
            image_embeddings,
            image_metadata,
            top_k=5
        )

        for result in image_results:
            print("-" * 80)
            print(f"Score: {result['score']:.4f}")
            print(f"Image: {result['image_path']}")