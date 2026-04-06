#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

  
    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or just whitespace.")
        
        # We wrap 'text' in a list as the model expects a list of sequences
        embeddings = self.model.encode([text])
        
        # Return the first (and only) embedding
        return embeddings[0]
    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        
        # Create string representations for each movie
        movie_strings = [
            f"{doc['title']}: {doc['description']}" for doc in documents
        ]
        
        # Keep track of progress during encoding
        print("Building embeddings... this might take a minute.")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        
        # Ensure the cache directory exists before saving
        os.makedirs("cache", exist_ok=True)
        
        # Save embeddings
        np.save("cache/movie_embeddings.npy", self.embeddings)
        
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        
        cache_path = "cache/movie_embeddings.npy"
        
        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path)
            # Make sure we didn't add/remove any docs locally
            if len(self.embeddings) == len(documents):
                print("Loaded embeddings from cache.")
                return self.embeddings
                
        # If cache invalid or non-existent, rebuild
        return self.build_embeddings(documents)

    def search(self, query, limit=5):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
            
        query_embedding = self.generate_embedding(query)
        
        results = []
        for doc_index, doc in enumerate(self.documents):
            doc_embedding = self.embeddings[doc_index]
            score = cosine_similarity(query_embedding, doc_embedding)
            results.append((score, doc))
            
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x[0], reverse=True)
        
        top_results = []
        for score, doc in results[:limit]:
            top_results.append({
                "score": float(score),
                "title": doc["title"],
                "description": doc["description"]
            })
            
        return top_results


def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def verify_embeddings():
    search = SemanticSearch()
    
    # Update the path to 'movies.json' if yours is in a different directory 
    # (e.g., 'data/movies.json')
    try:
        with open("data/movies.json", "r", encoding="utf-8") as f:
            documents = json.load(f)["movies"]
    except FileNotFoundError:
        print("data/movies.json not found! Please check your file paths.")
        return
        
    embeddings = search.load_or_create_embeddings(documents)
    
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_text(text):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def run_semantic_search(query, limit):
    search = SemanticSearch()
    try:
        with open("data/movies.json", "r", encoding="utf-8") as f:
            documents = json.load(f)["movies"]
    except FileNotFoundError:
        print("data/movies.json not found! Please check your file paths.")
        return
        
    search.load_or_create_embeddings(documents)
    results = search.search(query, limit)
    
    for i, res in enumerate(results, 1):
        desc = res["description"]
        if len(desc) > 85: 
            desc = desc[:85] + "..."
        print(f"{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"  {desc}\n")


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("verify", help="Verify the semantic search model")
    
    # NEW: embed_text command
    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for the given text")
    embed_parser.add_argument("text", type=str, help="The string of text to embed")

    # NEW: verify_embeddings command (NO arguments needed)
    subparsers.add_parser("verify_embeddings", help="Verify document embeddings over our dataset")

    # NEW: embedquery command
    embedquery_parser = subparsers.add_parser("embedquery", help="Embed a search query")
    embedquery_parser.add_argument("query", type=str, help="The search query string")

    # NEW: search command
    search_parser = subparsers.add_parser("search", help="Search movies by meaning")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            # Handle the embed_text command
            embed_text(args.text)
        case "verify_embeddings":
            # NEW: Handle the verify_embeddings command
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            run_semantic_search(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()