#!/usr/bin/env python3
import argparse
import json
import os
import re
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
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        
        all_chunks = []
        chunk_metadata = []
        
        for doc_idx, doc in enumerate(documents):
            desc = doc.get("description", "")
            if not desc:
                continue
                
            chunks = get_semantic_chunks(desc, max_chunk_size=4, overlap=1)
            total_chunks = len(chunks)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks
                })
                
        print("Building chunk embeddings... this might take a minute.")
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        
        os.makedirs("cache", exist_ok=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
            
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        
        chunk_emb_path = "cache/chunk_embeddings.npy"
        chunk_meta_path = "cache/chunk_metadata.json"
        
        if os.path.exists(chunk_emb_path) and os.path.exists(chunk_meta_path):
            self.chunk_embeddings = np.load(chunk_emb_path)
            with open(chunk_meta_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
                self.chunk_metadata = meta_data["chunks"]
                total_chunks = meta_data["total_chunks"]
                
            if len(self.chunk_embeddings) == total_chunks:
                print("Loaded chunk embeddings from cache.")
                return self.chunk_embeddings
                
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_metadata_item = self.chunk_metadata[i]
            
            chunk_scores.append({
                "chunk_idx": chunk_metadata_item["chunk_idx"],
                "movie_idx": chunk_metadata_item["movie_idx"],
                "score": score
            })
            
        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]["score"]:
                movie_scores[movie_idx] = chunk_score
                
        sorted_movie_scores = sorted(movie_scores.values(), key=lambda x: x["score"], reverse=True)
        top_movie_scores = sorted_movie_scores[:limit]
        
        results = []
        for ms in top_movie_scores:
            movie_idx = ms["movie_idx"]
            doc = self.documents[movie_idx]
            desc = doc.get("description", "")
            
            results.append({
                "id": doc.get("id"),
                "title": doc.get("title"),
                "document": desc[:100],
                "score": float(ms["score"]),
                "metadata": {
                    "chunk_idx": ms["chunk_idx"],
                    "movie_idx": movie_idx
                }
            })
            
        return results


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


def run_chunking(text: str, chunk_size: int, overlap: int = 0):
    words = text.split()
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be strictly greater than overlap")
        
    step = chunk_size - overlap
    chunks = []
    
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + chunk_size]))
        if i + chunk_size >= len(words):
            break
            
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def get_semantic_chunks(text: str, max_chunk_size: int, overlap: int = 0) -> list[str]:
    text = text.strip()
    if not text:
        return []

    split_sentences = re.split(r"(?<=[.!?])\s+", text)
    
    if len(split_sentences) == 1 and not split_sentences[0].endswith((".", "!", "?")):
        sentences = [text]
    else:
        sentences = split_sentences
        
    cleaned_sentences = []
    for s in sentences:
        s_stripped = s.strip()
        if s_stripped:
            cleaned_sentences.append(s_stripped)
            
    if not cleaned_sentences:
        return []

    if max_chunk_size <= overlap:
        raise ValueError("max_chunk_size must be strictly greater than overlap")
        
    step = max_chunk_size - overlap
    chunks = []
    
    for i in range(0, len(cleaned_sentences), step):
        chunks.append(" ".join(cleaned_sentences[i:i + max_chunk_size]))
        if i + max_chunk_size >= len(cleaned_sentences):
            break
            
    return chunks


def run_semantic_chunking(text: str, max_chunk_size: int, overlap: int = 0):
    chunks = get_semantic_chunks(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def run_embed_chunks():
    try:
        with open("data/movies.json", "r", encoding="utf-8") as f:
            documents = json.load(f)["movies"]
    except FileNotFoundError:
        print("data/movies.json not found! Please check your file paths.")
        return
        
    search = ChunkedSemanticSearch()
    embeddings = search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def run_search_chunked(query: str, limit: int = 5):
    try:
        with open("data/movies.json", "r", encoding="utf-8") as f:
            documents = json.load(f)["movies"]
    except FileNotFoundError:
        print("data/movies.json not found! Please check your file paths.")
        return
        
    search = ChunkedSemanticSearch()
    search.load_or_create_chunk_embeddings(documents)
    
    results = search.search_chunks(query, limit)
    
    for i, res in enumerate(results, 1):
        print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['document']}...")


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

    subparsers.add_parser("embed_chunks", help="Embed all document chunks")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text into fixed size pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words per chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of words to overlap between chunks")

    sem_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk text semantically into sentences")
    sem_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    sem_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Max sentences per chunk")
    sem_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of sentences to overlap")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search movies using chunked semantic embeddings")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

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
        case "embed_chunks":
            run_embed_chunks()
        case "chunk":
            run_chunking(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            run_semantic_chunking(args.text, args.max_chunk_size, args.overlap)
        case "search_chunked":
            run_search_chunked(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()