import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

TEXT_EMBEDDINGS_FILE = Path("data/processed/text_embeddings.npy")
TEXT_METADATA_FILE = Path("data/processed/text_metadata.jsonl")

IMAGE_EMBEDDINGS_FILE = Path("data/processed/image_embeddings.npy")
IMAGE_METADATA_FILE = Path("data/processed/image_metadata.jsonl")


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records

class MultimodalRetriever:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.model = CLIPModel.from_pretrained(MODEL_NAME)
        self.model.eval()

        self.text_embeddings = np.load(TEXT_EMBEDDINGS_FILE)
        self.text_metadata = load_jsonl(TEXT_METADATA_FILE)

        self.image_embeddings = None
        self.image_metadata = None

        if IMAGE_EMBEDDINGS_FILE.exists() and IMAGE_METADATA_FILE.exists():
            self.image_embeddings = np.load(IMAGE_EMBEDDINGS_FILE)
            self.image_metadata = load_jsonl(IMAGE_METADATA_FILE)

    def embed_query(self, query):
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            query_features = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        if hasattr(query_features, "pooler_output"):
            query_features = query_features.pooler_output

        query_features = query_features / query_features.norm(dim=1, keepdim=True)

        return query_features.cpu().numpy()
    
    def search_embeddings(self, query_embedding, embeddings, metadata, top_k):
        scores = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in top_indices:
            result = metadata[index].copy()
            result["score"] = float(scores[index])
            results.append(result)

        return results
    
    def search(self, query, top_k=5):
        query_embedding = self.embed_query(query)

        text_results = self.search_embeddings(
            query_embedding,
            self.text_embeddings,
            self.text_metadata,
            top_k
        )

        image_results = []

        if self.image_embeddings is not None:
            image_results = self.search_embeddings(
                query_embedding,
                self.image_embeddings,
                self.image_metadata,
                top_k
            )

        return {
            "query": query,
            "text_results": text_results,
            "image_results": image_results
        }
    
#test
if __name__ == "__main__":
    retriever = MultimodalRetriever()

    query = input("Enter your search query: ")

    results = retriever.search(query, top_k=5)

    print(f"\nQuery: {results['query']}")

    print("\nTop text results:")
    for result in results["text_results"]:
        print("-" * 80)
        print(f"Score: {result['score']:.4f}")
        print(f"File: {result['filename']}")
        print(f"Page: {result['page_number']}")
        print(f"Chunk: {result['chunk_id']}")
        print(result["text"][:500])

    print("\nTop image results:")
    for result in results["image_results"]:
        print("-" * 80)
        print(f"Score: {result['score']:.4f}")
        print(f"Image: {result['image_path']}")