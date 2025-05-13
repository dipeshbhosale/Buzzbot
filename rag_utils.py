from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np
import os
import json
from pathlib import Path
import faiss # Import FAISS
from concurrent.futures import ThreadPoolExecutor # Added for parallel encoding

class RAGManager:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents: List[str] = []
        self.embeddings: np.ndarray = np.array([]) # Initialize as empty numpy array
        self.index: Optional[faiss.Index] = None # FAISS index

        self.storage_dir = Path("knowledge_base")
        self.storage_dir.mkdir(exist_ok=True)
        self.docs_file = self.storage_dir / "documents.json"
        self.embeddings_file = self.storage_dir / "embeddings.npy"
        self.index_file = self.storage_dir / "knowledge.faiss" # FAISS index file

        # Clear any persisted knowledge files before attempting to load.
        # This ensures that each new RAGManager instance (when not from cache)
        # starts with a fresh, empty knowledge base from disk.
        self._clear_persisted_knowledge_files()

        self.load_knowledge_base()

    def _clear_persisted_knowledge_files(self):
        """
        Deletes the persisted documents, embeddings, and FAISS index files
        from the storage directory. This is called during initialization to ensure
        the RAGManager starts with a clean slate from disk.
        """
        print("Attempting to clear persisted knowledge base files...")
        files_to_clear = [self.docs_file, self.embeddings_file, self.index_file]
        for file_path in files_to_clear:
            if file_path.exists():
                try:
                    os.remove(file_path)
                    print(f"Successfully removed: {file_path}")
                except OSError as e: # Catch OS-level errors during file removal
                    print(f"Error removing file {file_path}: {e}")
            else:
                print(f"File not found, no need to remove: {file_path}")

    def load_knowledge_base(self):
        """Loads documents, embeddings, and FAISS index from storage."""
        # 1. Load documents
        if self.docs_file.exists():
            try:
                with open(self.docs_file, 'r') as f:
                    self.documents = json.load(f)
                print(f"Loaded {len(self.documents)} documents from {self.docs_file}.")
            except Exception as e:
                print(f"Error loading documents from {self.docs_file}: {e}. Starting with empty document list.")
                self.documents = []
        else:
            print(f"Documents file {self.docs_file} not found. Starting with empty document list.")
            self.documents = []

        # 2. Load or compute embeddings
        if self.documents: # Only proceed if there are documents
            if self.embeddings_file.exists():
                try:
                    loaded_embeddings = np.load(self.embeddings_file)
                    if len(loaded_embeddings) == len(self.documents):
                        self.embeddings = loaded_embeddings
                        print(f"Loaded {len(self.embeddings)} pre-computed embeddings from {self.embeddings_file}.")
                    else:
                        print(f"Embeddings file {self.embeddings_file} out of sync with documents. Re-computing.")
                        self._compute_and_save_embeddings()
                except Exception as e:
                    print(f"Error loading embeddings from {self.embeddings_file}: {e}. Re-computing.")
                    self._compute_and_save_embeddings()
            else: # No embeddings file, compute them
                print(f"Embeddings file {self.embeddings_file} not found. Computing embeddings.")
                self._compute_and_save_embeddings()
        else: # No documents, so no embeddings
            self.embeddings = np.array([])
            print("No documents loaded, so no embeddings to load or compute.")

        # 3. Load or build FAISS index
        if self.embeddings.size > 0: # Check if embeddings array is not empty
            if self.index_file.exists():
                try:
                    self.index = faiss.read_index(str(self.index_file))
                    if self.index.ntotal == len(self.embeddings):
                        print(f"Loaded FAISS index with {self.index.ntotal} vectors from {self.index_file}.")
                    else:
                        print(f"FAISS index {self.index_file} out of sync with embeddings ({self.index.ntotal} vs {len(self.embeddings)}). Re-building.")
                        self._build_and_save_index()
                except Exception as e:
                    print(f"Error loading FAISS index from {self.index_file}: {e}. Re-building.")
                    self._build_and_save_index()
            else: # No index file, build it
                print(f"FAISS index file {self.index_file} not found. Building index.")
                self._build_and_save_index()
        else: # No embeddings, so no index
            self.index = None
            print("No embeddings available, so no FAISS index to load or build.")

    def _embed_single_document(self, doc_text: str) -> np.ndarray:
        """Helper to embed a single document string. For use with ThreadPoolExecutor."""
        # self.model.encode expects a list, even for a single sentence.
        # It returns a list of embeddings; we take the first (and only) one.
        return self.model.encode([doc_text])[0]

    def _encode_documents_parallel(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a list of documents in parallel using ThreadPoolExecutor.
        Note: sentence-transformers' model.encode() on a list is already batch-optimized.
        This method provides an alternative parallelization strategy.
        """
        if not texts:
            return np.array([])
        
        print(f"Computing embeddings for {len(texts)} documents (using ThreadPoolExecutor)...")
        # Using max_workers=None will default to os.cpu_count() * 5 for ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=None) as executor:
            embeddings_list = list(executor.map(self._embed_single_document, texts))
        
        return np.array(embeddings_list).astype(np.float32)

    def _compute_and_save_embeddings(self):
        """Computes embeddings for self.documents and saves them."""
        if not self.documents:
            self.embeddings = np.array([])
            print("No documents to compute embeddings for.")
            return
        print(f"Computing embeddings for {len(self.documents)} documents...")
        # Current efficient batch encoding:
        self.embeddings = self.model.encode(self.documents, show_progress_bar=True)
        # Alternative using ThreadPoolExecutor (generally not needed for sentence-transformers):
        # self.embeddings = self._encode_documents_parallel(self.documents)

        self._save_embeddings()

    def save_documents(self):
        """Save documents to disk."""
        with open(self.docs_file, 'w') as f:
            json.dump(self.documents, f)
        print(f"Saved {len(self.documents)} documents to {self.docs_file}")

    def _save_embeddings(self):
        """Save embeddings to disk."""
        if self.embeddings.size > 0: # Check if there are embeddings to save
            np.save(self.embeddings_file, self.embeddings)
            print(f"Saved {len(self.embeddings)} embeddings to {self.embeddings_file}")

    def _build_and_save_index(self):
        """Builds FAISS index from self.embeddings and saves it."""
        if self.embeddings.size == 0:
            print("No embeddings available to build FAISS index.")
            self.index = None
            return

        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d) # Using Inner Product (cosine similarity for normalized vectors)
        try:
            self.index.add(self.embeddings.astype(np.float32))
            print(f"FAISS index built with {self.index.ntotal} vectors.")
            self._save_index()
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            self.index = None

    def _save_index(self):
        """Saves the current FAISS index to disk."""
        if self.index:
            try:
                faiss.write_index(self.index, str(self.index_file))
                print(f"FAISS index saved to {self.index_file}")
            except Exception as e:
                print(f"Error saving FAISS index to {self.index_file}: {e}")

    def add_documents(self, texts: List[str]):
        """Adds new documents to the collection."""
        if not texts:
            return

        new_doc_count = len(texts)
        print(f"Adding {new_doc_count} new documents.")
        self.documents.extend(texts)
        self.save_documents()

        print(f"Computing embeddings for {new_doc_count} new documents...")
        # Current efficient batch encoding:
        new_embeddings = self.model.encode(texts, show_progress_bar=True).astype(np.float32)
        # Alternative using ThreadPoolExecutor:
        # new_embeddings = self._encode_documents_parallel(texts)

        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
            # Ensure new_embeddings is float32 if it came from _encode_documents_parallel
        else:
            if self.embeddings.shape[1] != new_embeddings.shape[1]:
                print(f"Dimension mismatch: existing embeddings {self.embeddings.shape[1]}, new embeddings {new_embeddings.shape[1]}. Re-building all embeddings and index.")
                self._compute_and_save_embeddings() # Re-encodes all current self.documents
                self._build_and_save_index()
                return
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self._save_embeddings()

        # Update or build FAISS index
        if self.index and self.index.d == new_embeddings.shape[1] and self.index.ntotal == (len(self.embeddings) - new_doc_count):
            try:
                self.index.add(new_embeddings)
                print(f"Added {new_doc_count} vectors to existing FAISS index. Total vectors: {self.index.ntotal}")
                self._save_index() # Persist changes
            except Exception as e:
                print(f"Error adding to existing FAISS index: {e}. Re-building.")
                self._build_and_save_index()
        else:
            print("Re-building FAISS index as it's missing, outdated, or dimensions changed.")
            self._build_and_save_index()

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieves k most relevant documents for the query."""
        if self.index is None or self.index.ntotal == 0:
            print("FAISS index is not available or empty for retrieval.")
            return []
        if not self.documents: # Should ideally not happen if index exists and is non-empty
            print("No documents available for retrieval, though index exists.")
            return []

        query_embedding = self.model.encode([query])[0].astype(np.float32).reshape(1, -1)

        actual_k = min(k, self.index.ntotal)
        if actual_k == 0:
            return []

        print(f"Searching FAISS index for top {actual_k} results for query: '{query[:50]}...'")
        try:
            distances, indices = self.index.search(query_embedding, actual_k)
        except Exception as e:
            print(f"Error during FAISS search: {e}")
            return []

        retrieved_docs = []
        for i in indices[0]:
            if i != -1 and i < len(self.documents): # FAISS can return -1 if not enough results
                retrieved_docs.append(self.documents[i])
            elif i == -1:
                print("FAISS search returned -1 index, indicating fewer than k results found.")
            else: # i >= len(self.documents)
                print(f"Warning: FAISS returned index {i} which is out of bounds for {len(self.documents)} documents. Index might be stale.")

        print(f"Retrieved {len(retrieved_docs)} documents. Indices: {indices[0]}, Distances: {distances[0]}")
        return retrieved_docs

    def get_augmented_prompt(self, query: str) -> str:
        """Creates an augmented prompt with retrieved context."""
        relevant_docs = self.retrieve(query)
        if not relevant_docs:
            return query
            
        context = "\n".join(relevant_docs)
        return f"""Context information is below.
---------------------
{context}
---------------------
Based on the above context and your knowledge, please respond to: {query}"""
