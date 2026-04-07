#!/usr/bin/env python3
import argparse
import json
from PIL import Image
from sentence_transformers import SentenceTransformer, util

class MultimodalSearch:
    def __init__(self, documents=None, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents if documents is not None else []
        self.texts = [f"{doc.get('title', '')}: {doc.get('description', '')}" for doc in self.documents]
        if self.texts:
            self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        else:
            self.text_embeddings = []

    def embed_image(self, image_path):
        image = Image.open(image_path)
        embedding = self.model.encode([image])[0]
        return embedding

    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)
        
        results = []
        for i, doc in enumerate(self.documents):
            sim = util.cos_sim(image_embedding, self.text_embeddings[i])[0][0].item()
            results.append({
                "id": doc.get("id"),
                "title": doc.get("title"),
                "description": doc.get("description"),
                "similarity": sim
            })
            
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:5]

def verify_image_embedding(image_path):
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path):
    try:
        with open("data/movies.json", "r", encoding="utf-8") as f:
            documents = json.load(f)["movies"]
    except FileNotFoundError:
        print("data/movies.json not found!")
        return

    search = MultimodalSearch(documents)
    results = search.search_with_image(image_path)
    
    for i, res in enumerate(results, 1):
        desc = res.get("description", "")
        if len(desc) > 100:
            desc = desc[:100] + "..."
        print(f"{i}. {res['title']} (similarity: {res['similarity']:.3f})")
        print(f"   {desc}\n")

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding generation")
    verify_parser.add_argument("image_path", help="Path to the image")

    search_parser = subparsers.add_parser("image_search", help="Search movies using an image")
    search_parser.add_argument("image_path", help="Path to the image")

    args = parser.parse_args()

    if args.command == "verify_image_embedding":
        verify_image_embedding(args.image_path)
    elif args.command == "image_search":
        image_search_command(args.image_path)

if __name__ == "__main__":
    main()
