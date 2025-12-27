from pathlib import Path
import json
import librosa
import numpy as np
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except ImportError:
    chromadb = None
    SentenceTransformer = None

class RAGSystem:
    """
    Retrieval-Augmented Generation for Music.
    - Indexes your local music library.
    - Retrieves 'similar' tracks to use as reference/inspiration.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.db_dir = data_dir / "vectordb"
        self.collection_name = "music_collection"
        self.client = None
        self.collection = None
        self.text_model = None
        
        if chromadb is None:
            print("⚠️ RAG dependencies missing. Install chromadb and sentence-transformers.")
            return

        self._init_db()
        self._load_models()

    def _init_db(self):
        """Initialize ChromaDB"""
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print(f"📚 Music VectorDB Initialized at {self.db_dir}")

    def _load_models(self):
        """Load embedding models"""
        # We use a lightweight model for text-to-text similarity (style descriptions)
        print("🧠 Loading RAG models...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        # In the future, we would load CLAP for audio-to-text similarity here
        print("✅ RAG models loaded")

    def index_library(self, metadata_dir: Path):
        """
        Read all metadata JSONs and index them.
        """
        if not self.collection: return
        
        embeddings = []
        documents = []
        metadatas = []
        ids = []
        
        json_files = list(metadata_dir.glob("*.json"))
        print(f"Scanning {len(json_files)} metadata files...")
        
        for f in json_files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    # Create a rich textual representation for embedding
                    # "Upbeat rock song with distorted guitar and fast tempo"
                    desc = data.get("description", "")
                    tags = f"{data.get('genre', '')} {data.get('mood', '')} {data.get('key', '')}"
                    full_text = f"{desc} {tags}"
                    
                    vec = self.text_model.encode(full_text).tolist()
                    
                    embeddings.append(vec)
                    documents.append(full_text)
                    metadatas.append(data)
                    ids.append(f.stem)
            except Exception as e:
                print(f"Skipping {f.name}: {e}")
                
        if ids:
            self.collection.upsert(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ Indexed {len(ids)} tracks into VectorDB")

    def query_similar(self, query_text: str, n_results: int = 3):
        """
        Find tracks similar to the user's prompt.
        """
        if not self.collection: return []
        
        query_vec = self.text_model.encode(query_text).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_results
        )
        
        # Flatten results
        hits = []
        if results['metadatas']:
            for i, meta in enumerate(results['metadatas'][0]):
                hits.append({
                    "score": results['distances'][0][i], # distance score
                    "metadata": meta
                })
        return hits
