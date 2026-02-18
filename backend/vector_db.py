"""
FAISS Vector Database Integration for Scalable Video Retrieval
Supports multi-video indexing and efficient similarity search
"""

import numpy as np
import faiss
import json
import os
from typing import List, Dict, Tuple, Optional
import pickle


class VideoVectorDatabase:
    """
    FAISS-based vector database for storing and searching video chunk embeddings.
    Supports multiple videos with metadata tracking.
    """
    
    def __init__(self, dimension: int = 1024, index_type: str = 'flat'):
        """
        Initialize the vector database.
        
        Args:
            dimension: Vector dimension (default 1024 for CLIP text+visual)
            index_type: 'flat' for exact search, 'ivf' for approximate (faster for large scale)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.metadata = []  # List of {video_id, chunk_index, timestamp, ...}
        self.video_registry = {}  # video_id -> {start_idx, end_idx, ...}
        
        self._create_index()
    
    def _create_index(self):
        """Create the FAISS index based on type."""
        if self.index_type == 'flat':
            # Exact search - good for up to ~100k vectors
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine sim for normalized vectors)
        elif self.index_type == 'ivf':
            # IVF index for larger scale - requires training
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
            self.index.nprobe = 10  # Search 10 clusters
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def add_video(
        self, 
        video_id: str,
        vectors: np.ndarray,
        chunk_metadata: List[Dict],
        video_info: Dict = None
    ):
        """
        Add a video's chunk vectors to the database.
        
        Args:
            video_id: Unique video identifier
            vectors: Numpy array of shape (num_chunks, dimension)
            chunk_metadata: List of metadata dicts for each chunk
            video_info: Optional video-level metadata (title, duration, etc.)
        """
        if video_id in self.video_registry:
            print(f"Warning: Video {video_id} already exists. Skipping.")
            return
        
        # Ensure vectors are float32 and normalized
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)
        
        # Track position in index
        start_idx = len(self.metadata)
        
        # Add to index
        if self.index_type == 'ivf' and not self.index.is_trained:
            print("Training IVF index...")
            self.index.train(vectors)
        
        self.index.add(vectors)
        
        # Store metadata
        for i, chunk_meta in enumerate(chunk_metadata):
            self.metadata.append({
                'video_id': video_id,
                'chunk_index': chunk_meta.get('index', i),
                'timestamp': chunk_meta.get('timestamp', 0),
                'ocr_text': chunk_meta.get('ocr_text', ''),
                'audio_context': chunk_meta.get('audio_context', ''),
                'frame_path': chunk_meta.get('frame_path', '')
            })
        
        # Register video
        self.video_registry[video_id] = {
            'start_idx': start_idx,
            'end_idx': start_idx + len(vectors),
            'num_chunks': len(vectors),
            'info': video_info or {}
        }
        
        print(f"Added video {video_id} with {len(vectors)} chunks")
    
    def remove_video(self, video_id: str):
        """
        Remove a video from the database.
        Note: FAISS doesn't support efficient removal, so we rebuild the index.
        """
        if video_id not in self.video_registry:
            print(f"Video {video_id} not found.")
            return
        
        # Get indices to keep
        keep_indices = []
        new_metadata = []
        
        for i, meta in enumerate(self.metadata):
            if meta['video_id'] != video_id:
                keep_indices.append(i)
                new_metadata.append(meta)
        
        if len(keep_indices) == 0:
            # Reset to empty
            self._create_index()
            self.metadata = []
            self.video_registry = {}
            return
        
        # Reconstruct vectors (this requires IndexFlat or similar)
        old_vectors = np.zeros((len(self.metadata), self.dimension), dtype=np.float32)
        for i in range(len(self.metadata)):
            old_vectors[i] = self.index.reconstruct(i)
        
        new_vectors = old_vectors[keep_indices]
        
        # Rebuild index
        self._create_index()
        self.index.add(new_vectors)
        self.metadata = new_metadata
        
        # Update registry
        del self.video_registry[video_id]
        self._rebuild_registry()
        
        print(f"Removed video {video_id}")
    
    def _rebuild_registry(self):
        """Rebuild video registry from metadata."""
        self.video_registry = {}
        current_video = None
        start_idx = 0
        
        for i, meta in enumerate(self.metadata):
            vid = meta['video_id']
            if vid != current_video:
                if current_video is not None:
                    self.video_registry[current_video]['end_idx'] = i
                self.video_registry[vid] = {'start_idx': i, 'end_idx': i + 1}
                current_video = vid
        
        if current_video is not None:
            self.video_registry[current_video]['end_idx'] = len(self.metadata)
        
        # Update num_chunks
        for vid, info in self.video_registry.items():
            info['num_chunks'] = info['end_idx'] - info['start_idx']
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        video_id: Optional[str] = None,
        search_mode: str = 'text'
    ) -> List[Dict]:
        """
        Search for similar chunks.
        
        Args:
            query_vector: Query embedding (can be 512D for text-only or 1024D for full)
            top_k: Number of results to return
            video_id: Optional - filter to specific video
            search_mode: 'text' (use first 512D), 'visual' (use last 512D), or 'full' (use all 1024D)
        
        Returns:
            List of result dicts with metadata and similarity scores
        """
        # Handle different query dimensions
        query = query_vector.astype(np.float32).reshape(1, -1)
        
        if search_mode == 'text' and query.shape[1] == 512:
            # Pad with zeros for visual portion
            query = np.hstack([query, np.zeros((1, 512), dtype=np.float32)])
        elif search_mode == 'visual' and query.shape[1] == 512:
            # Pad with zeros for text portion
            query = np.hstack([np.zeros((1, 512), dtype=np.float32), query])
        
        faiss.normalize_L2(query)
        
        # Search
        if video_id and video_id in self.video_registry:
            # Filter to specific video
            info = self.video_registry[video_id]
            
            # Create ID selector for video range
            id_selector = faiss.IDSelectorRange(info['start_idx'], info['end_idx'])
            params = faiss.SearchParametersIVF() if self.index_type == 'ivf' else faiss.SearchParameters()
            params.sel = id_selector
            
            distances, indices = self.index.search(query, min(top_k, info['num_chunks']), params=params)
        else:
            # Search all videos
            distances, indices = self.index.search(query, top_k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            meta = self.metadata[idx]
            results.append({
                'similarity': float(dist),
                'video_id': meta['video_id'],
                'chunk_index': meta['chunk_index'],
                'timestamp': meta['timestamp'],
                'ocr_text': meta['ocr_text'],
                'audio_context': meta['audio_context'],
                'frame_path': meta['frame_path']
            })
        
        return results
    
    def get_video_list(self) -> List[Dict]:
        """Get list of all indexed videos."""
        return [
            {
                'video_id': vid,
                'num_chunks': info['num_chunks'],
                'info': info.get('info', {})
            }
            for vid, info in self.video_registry.items()
        ]
    
    def get_video_chunks(self, video_id: str) -> List[Dict]:
        """Get all chunks for a specific video."""
        if video_id not in self.video_registry:
            return []
        
        info = self.video_registry[video_id]
        return self.metadata[info['start_idx']:info['end_idx']]
    
    def save(self, filepath: str):
        """Save the database to disk."""
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save metadata and registry
        with open(f"{filepath}.meta", 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'video_registry': self.video_registry,
                'dimension': self.dimension,
                'index_type': self.index_type
            }, f)
        
        print(f"Database saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the database from disk."""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load metadata and registry
        with open(f"{filepath}.meta", 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.video_registry = data['video_registry']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
        
        print(f"Database loaded from {filepath}")
        print(f"  Videos: {len(self.video_registry)}")
        print(f"  Total chunks: {len(self.metadata)}")
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        return {
            'total_videos': len(self.video_registry),
            'total_chunks': len(self.metadata),
            'dimension': self.dimension,
            'index_type': self.index_type,
            'videos': {
                vid: info['num_chunks'] 
                for vid, info in self.video_registry.items()
            }
        }


class VideoLibrary:
    """
    High-level video library manager that combines vector search with metadata.
    """
    
    def __init__(self, db_path: str = 'video_library', output_folder: str = 'outputs'):
        self.db_path = db_path
        self.output_folder = output_folder
        self.vector_db = VideoVectorDatabase()
        
        # Load existing database if available
        if os.path.exists(f"{db_path}.faiss"):
            self.vector_db.load(db_path)
    
    def index_video(self, video_id: str, video_info: Dict = None):
        """
        Index a processed video into the library.
        
        Args:
            video_id: Video ID (must already be processed by main app)
            video_info: Optional metadata (title, url, etc.)
        """
        # Load vectors
        chunks_file = os.path.join(self.output_folder, f"{video_id}_chunks.npz")
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"Video {video_id} not processed. Run transcription first.")
        
        data = np.load(chunks_file)
        vectors = data['vectors']
        
        # Load metadata
        metadata_file = os.path.join(self.output_folder, f"{video_id}_chunks_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Add to database
        self.vector_db.add_video(
            video_id=video_id,
            vectors=vectors,
            chunk_metadata=metadata['chunks'],
            video_info=video_info
        )
        
        # Save database
        self.vector_db.save(self.db_path)
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        video_id: Optional[str] = None
    ) -> List[Dict]:
        """Search across the video library."""
        return self.vector_db.search(query_vector, top_k, video_id)
    
    def get_videos(self) -> List[Dict]:
        """Get all indexed videos."""
        return self.vector_db.get_video_list()
    
    def remove_video(self, video_id: str):
        """Remove a video from the library."""
        self.vector_db.remove_video(video_id)
        self.vector_db.save(self.db_path)


# Singleton instance for app integration
_library_instance = None

def get_library(output_folder: str = 'outputs') -> VideoLibrary:
    """Get or create the video library singleton."""
    global _library_instance
    if _library_instance is None:
        _library_instance = VideoLibrary(
            db_path=os.path.join(output_folder, 'video_library'),
            output_folder=output_folder
        )
    return _library_instance


if __name__ == '__main__':
    # Example usage
    print("Testing VideoVectorDatabase...")
    
    db = VideoVectorDatabase()
    
    # Create dummy data
    vectors = np.random.randn(10, 1024).astype(np.float32)
    metadata = [{'index': i, 'timestamp': i * 2.0, 'ocr_text': f'Text {i}'} for i in range(10)]
    
    db.add_video('test_video', vectors, metadata)
    
    # Search
    query = np.random.randn(1024).astype(np.float32)
    results = db.search(query, top_k=3)
    
    print(f"\nSearch results: {len(results)}")
    for r in results:
        print(f"  Chunk {r['chunk_index']}: similarity={r['similarity']:.4f}")
    
    print(f"\nStats: {db.get_stats()}")
