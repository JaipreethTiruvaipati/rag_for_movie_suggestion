import argparse
import json
import string
import os
import math
import pickle
from nltk.stem import PorterStemmer
from collections import Counter

BM25_K1 = 1.5
BM25_B = 0.75 

class InvertedIndex:
    def __init__(self):
        self.index = {}  # maps token -> set of doc_ids
        self.docmap = {} # maps doc_id -> full doc object
        self.term_frequencies = {} # maps doc_id -> Counter(token -> count)
        self.doc_lengths = {}                                          # ← ADD
        self.doc_lengths_path = "cache/doc_lengths.pkl"
        # Pre-load stopwords and our stemmer so __add_document can use them
        with open("data/stopwords.txt", "r", encoding="utf-8") as f:
            self.stopwords = set(f.read().splitlines())
        self.stemmer = PorterStemmer()

    def __add_document(self, doc_id: int, text: str):
        raw_tokens = get_tokens(text)
        
        processed_tokens = [
            self.stemmer.stem(t) for t in raw_tokens if t not in self.stopwords
        ]
        
        # Initialize a counter for this document
        self.term_frequencies[doc_id] = Counter()
        self.doc_lengths[doc_id] = len(processed_tokens) 
        for token in processed_tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            
            # Increment the term frequency cache for this token
            self.term_frequencies[doc_id][token] += 1


    def get_documents(self, term: str) -> list[int]:
        # Clean and stem the input term to match how we indexed it
        term = term.lower()
        term = self.stemmer.stem(term)
        
        # Return sorted list of matching IDs
        if term in self.index:
            return sorted(list(self.index[term]))
        return []

    def build(self):
        with open("data/movies.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for movie in data["movies"]:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            
            # Combine the title and description for indexing
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)

    def save(self):
        # Create cache directory if it doesn't exist
        os.makedirs("cache", exist_ok=True)
        
        # Save attributes using pickle
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open(self.doc_lengths_path, "wb") as f:    
            pickle.dump(self.doc_lengths, f)             
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        if not os.path.exists("cache/index.pkl") or not os.path.exists("cache/docmap.pkl"):
            raise FileNotFoundError("Cache files not found. Please run the 'build' command first.")
        
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:     
            self.doc_lengths = pickle.load(f)    
                    
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:        # Edge case: no documents
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)


    def get_tf(self, doc_id: int, term: str) -> int:
        # Tokenize the term exactly like we tokenize documents
        raw_tokens = get_tokens(term)
        if len(raw_tokens) > 1:
            raise Exception("Term must be a single string/token.")
            
        term_tokens = [self.stemmer.stem(t) for t in raw_tokens if t not in self.stopwords]
        if not term_tokens:
            return 0 # Edge case: the search term was a stopword
            
        token = term_tokens[0]
        
        if doc_id in self.term_frequencies:
            return self.term_frequencies[doc_id].get(token, 0)
        return 0
    def get_bm25_idf(self, term: str) -> float:
        # Tokenize the term exactly like we tokenize documents
        raw_tokens = get_tokens(term)
        if len(raw_tokens) > 1:
            raise Exception("Term must be a single string/token.")
            
        term_tokens = [self.stemmer.stem(t) for t in raw_tokens if t not in self.stopwords]
        if not term_tokens:
            return 0.0 # Edge case: the search term was a stopword
            
        token = term_tokens[0]
        
        # Calculate N (Total number of documents)
        N = len(self.docmap)
        
        # Calculate df (Document frequency: how many docs contain the token)
        df = len(self.index.get(token, set()))
        
        # BM25 IDF formula
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return idf
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()

        length_norm = 1 - b + b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1.0
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        """Return the full BM25 score (TF component * IDF component) for a term in a document."""
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int = 5) -> list[tuple[dict, float]]:
        """Score every document against the query using full BM25 and return the top results."""
        # Tokenize + stem + stopword-filter the query, same as indexing
        query_tokens = [
            self.stemmer.stem(t)
            for t in get_tokens(query)
            if t not in self.stopwords
        ]

        # Accumulate BM25 scores across all documents for each query token
        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            total = 0.0
            for token in query_tokens:
                total += self.bm25(doc_id, token)
            scores[doc_id] = total

        # Sort by score descending and return the top `limit` (movie, score) pairs
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.docmap[doc_id], score) for doc_id, score in ranked[:limit]]


def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_tf(doc_id, term, k1, b)



def get_tokens(text: str) -> list[str]:
    # Remove punctuation and lowercase
    remove_punct = str.maketrans("", "", string.punctuation)
    clean_text = text.lower().translate(remove_punct)
    # .split() inherently splits by all whitespace and ignores empty strings
    return clean_text.split()

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build and cache the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")
    
    # Add the idf subparser
    idf_parser = subparsers.add_parser("idf", help="Calculate Inverse Document Frequency (IDF) for a term")
    idf_parser.add_argument("term", type=str, help="Search term")
    # Add the tfidf subparser
    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search term")

    # Add the bm25idf subparser
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
      "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")

    args = parser.parse_args()
    match args.command:
        case "build":
            # Build and cache the index 
            index = InvertedIndex()
            index.build()
            index.save()
            print("Index built successfully!")

        case "tf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                return
            
            tf = index.get_tf(args.doc_id, args.term)
            print(tf)
        case "idf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                return
            
            # 1. Total documents in the dataset
            total_doc_count = len(index.docmap)
            
            # 2. Total documents that contain the searched term
            term_match_doc_count = len(index.get_documents(args.term))
            
            # 3. Calculate IDF score with +1 smoothing
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            
            # Print the formatted output
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        
        case "tfidf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                return
            
            # 1. Calculate the Term Frequency (TF)
            tf = index.get_tf(args.doc_id, args.term)
            
            # 2. Calculate the Inverse Document Frequency (IDF)
            total_doc_count = len(index.docmap)
            term_match_doc_count = len(index.get_documents(args.term))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            
            # 3. Multiply them to get the final TF-IDF score
            tf_idf = tf * idf
            
            # Print the formatted output
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        
        case "bm25idf":
            try:
                bm25idf = bm25_idf_command(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            except Exception as e:
                print(e)
                
        case "bm25tf":
            try:
                bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)   
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            except Exception as e:
                print(e)


        case "bm25search":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                return

            results = index.bm25_search(args.query, args.limit)
            for i, (movie, score) in enumerate(results, 1):
                print(f"{i}. ({movie['id']}) {movie['title']} - Score: {score:.2f}")

        case "search":
            print(f"Searching for: {args.query}")
            
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                return
            
            # Use the stemmer and stopwords we already loaded on the InvertedIndex class!
            query_tokens = [
                index.stemmer.stem(t) for t in get_tokens(args.query) if t not in index.stopwords
            ]

            results = []
            seen_ids = set() # To ensure we don't return duplicate movies
            
            # Iterate through our query tokens
            for token in query_tokens:
                doc_ids = index.get_documents(token)
                
                # Add each matching document until we hit 5
                for doc_id in doc_ids:
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        results.append(index.docmap[doc_id])
                        
                    if len(results) == 5:
                        break
                        
                if len(results) == 5:
                    break
            
            # Print the results along with the document IDs
            for i, movie in enumerate(results, 1):
                print(f"{i}. {movie['title']} (ID: {movie['id']})")

                
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
