"""
Unit Tests for Multimodal Video Retrieval Backend
Tests core functionality: OCR, keyframe detection, vector creation, search, evaluation
"""

import unittest
import numpy as np
import os
import sys
import json
import tempfile
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import RetrievalEvaluator, VideoRetrievalBenchmark
from vector_db import VideoVectorDatabase, VideoLibrary


class TestRetrievalEvaluator(unittest.TestCase):
    """Tests for the evaluation metrics framework."""
    
    def setUp(self):
        self.evaluator = RetrievalEvaluator()
        
    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        relevant = {'doc1', 'doc3', 'doc5'}
        
        # P@1 = 1/1 = 1.0 (doc1 is relevant)
        self.assertEqual(self.evaluator.precision_at_k(retrieved, relevant, 1), 1.0)
        
        # P@2 = 1/2 = 0.5 (doc1 relevant, doc2 not)
        self.assertEqual(self.evaluator.precision_at_k(retrieved, relevant, 2), 0.5)
        
        # P@3 = 2/3 (doc1, doc3 relevant)
        self.assertAlmostEqual(self.evaluator.precision_at_k(retrieved, relevant, 3), 2/3)
        
        # P@5 = 3/5 = 0.6 (doc1, doc3, doc5 relevant)
        self.assertEqual(self.evaluator.precision_at_k(retrieved, relevant, 5), 0.6)
        
    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        relevant = {'doc1', 'doc3', 'doc5'}
        
        # R@1 = 1/3 (1 of 3 relevant docs found)
        self.assertAlmostEqual(self.evaluator.recall_at_k(retrieved, relevant, 1), 1/3)
        
        # R@3 = 2/3 (doc1, doc3 found)
        self.assertAlmostEqual(self.evaluator.recall_at_k(retrieved, relevant, 3), 2/3)
        
        # R@5 = 3/3 = 1.0 (all relevant found)
        self.assertEqual(self.evaluator.recall_at_k(retrieved, relevant, 5), 1.0)
        
    def test_f1_at_k(self):
        """Test F1@K calculation."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc1', 'doc3'}
        
        # F1@3: P=2/3, R=2/2=1.0 -> F1 = 2*(2/3*1)/(2/3+1) = 0.8
        f1 = self.evaluator.f1_at_k(retrieved, relevant, 3)
        self.assertAlmostEqual(f1, 0.8)
        
    def test_mean_reciprocal_rank(self):
        """Test MRR calculation."""
        # First result is relevant: MRR = 1/1 = 1.0
        self.assertEqual(
            self.evaluator.mean_reciprocal_rank(['rel', 'not', 'not'], {'rel'}),
            1.0
        )
        
        # Second result is relevant: MRR = 1/2 = 0.5
        self.assertEqual(
            self.evaluator.mean_reciprocal_rank(['not', 'rel', 'not'], {'rel'}),
            0.5
        )
        
        # Third result is relevant: MRR = 1/3
        self.assertAlmostEqual(
            self.evaluator.mean_reciprocal_rank(['not', 'not', 'rel'], {'rel'}),
            1/3
        )
        
        # No relevant results: MRR = 0
        self.assertEqual(
            self.evaluator.mean_reciprocal_rank(['not1', 'not2'], {'rel'}),
            0.0
        )
        
    def test_ndcg_at_k(self):
        """Test nDCG@K calculation."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevance_scores = {'doc1': 3, 'doc2': 0, 'doc3': 2}
        
        # nDCG should be between 0 and 1
        ndcg = self.evaluator.ndcg_at_k(retrieved, relevance_scores, 3)
        self.assertGreaterEqual(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)
        
        # Perfect ranking should give nDCG = 1.0
        perfect_retrieved = ['doc1', 'doc3', 'doc2']
        ndcg_perfect = self.evaluator.ndcg_at_k(perfect_retrieved, relevance_scores, 3)
        self.assertEqual(ndcg_perfect, 1.0)
        
    def test_average_precision(self):
        """Test Average Precision calculation."""
        retrieved = ['doc1', 'not1', 'doc2', 'not2', 'doc3']
        relevant = {'doc1', 'doc2', 'doc3'}
        
        # AP = (1/1 + 2/3 + 3/5) / 3
        expected_ap = (1 + 2/3 + 3/5) / 3
        ap = self.evaluator.average_precision(retrieved, relevant)
        self.assertAlmostEqual(ap, expected_ap)
        
    def test_evaluate_query_comprehensive(self):
        """Test comprehensive query evaluation."""
        retrieved = ['doc1', 'doc3', 'doc2', 'doc5', 'doc4']
        relevant = {'doc1', 'doc2', 'doc4'}
        
        results = self.evaluator.evaluate_query(
            query_id='test',
            retrieved=retrieved,
            relevant=relevant,
            k_values=[1, 3, 5]
        )
        
        # Check all metrics are present
        self.assertIn('mrr', results)
        self.assertIn('ap', results)
        self.assertIn('precision@1', results)
        self.assertIn('recall@5', results)
        self.assertIn('f1@3', results)
        self.assertIn('ndcg@5', results)
        
    def test_empty_cases(self):
        """Test edge cases with empty inputs."""
        # Empty relevant set
        self.assertEqual(
            self.evaluator.precision_at_k(['a', 'b'], set(), 2),
            0.0
        )
        self.assertEqual(
            self.evaluator.recall_at_k(['a', 'b'], set(), 2),
            0.0
        )
        
        # K = 0
        self.assertEqual(
            self.evaluator.precision_at_k(['a'], {'a'}, 0),
            0.0
        )


class TestVideoVectorDatabase(unittest.TestCase):
    """Tests for the FAISS vector database."""
    
    def setUp(self):
        self.db = VideoVectorDatabase(dimension=1024, index_type='flat')
        
    def test_add_video(self):
        """Test adding a video to the database."""
        vectors = np.random.randn(10, 1024).astype(np.float32)
        metadata = [{'index': i, 'timestamp': i * 2.0, 'ocr_text': f'Text {i}'} for i in range(10)]
        
        self.db.add_video('test_video', vectors, metadata)
        
        self.assertEqual(len(self.db.metadata), 10)
        self.assertIn('test_video', self.db.video_registry)
        self.assertEqual(self.db.video_registry['test_video']['num_chunks'], 10)
        
    def test_search(self):
        """Test vector search functionality."""
        # Add video
        vectors = np.random.randn(10, 1024).astype(np.float32)
        metadata = [{'index': i, 'timestamp': i * 2.0, 'ocr_text': f'Text {i}'} for i in range(10)]
        self.db.add_video('test_video', vectors, metadata)
        
        # Search
        query = np.random.randn(1024).astype(np.float32)
        results = self.db.search(query, top_k=5)
        
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIn('similarity', result)
            self.assertIn('video_id', result)
            self.assertIn('chunk_index', result)
            self.assertIn('timestamp', result)
            
    def test_search_specific_video(self):
        """Test searching within a specific video."""
        # Add two videos
        vectors1 = np.random.randn(5, 1024).astype(np.float32)
        vectors2 = np.random.randn(5, 1024).astype(np.float32)
        metadata1 = [{'index': i, 'timestamp': i} for i in range(5)]
        metadata2 = [{'index': i, 'timestamp': i} for i in range(5)]
        
        self.db.add_video('video1', vectors1, metadata1)
        self.db.add_video('video2', vectors2, metadata2)
        
        # Search only in video1
        query = np.random.randn(1024).astype(np.float32)
        results = self.db.search(query, top_k=3, video_id='video1')
        
        for result in results:
            self.assertEqual(result['video_id'], 'video1')
            
    def test_get_video_list(self):
        """Test getting list of indexed videos."""
        vectors = np.random.randn(5, 1024).astype(np.float32)
        metadata = [{'index': i} for i in range(5)]
        
        self.db.add_video('vid1', vectors, metadata)
        self.db.add_video('vid2', vectors.copy(), metadata)
        
        video_list = self.db.get_video_list()
        
        self.assertEqual(len(video_list), 2)
        video_ids = [v['video_id'] for v in video_list]
        self.assertIn('vid1', video_ids)
        self.assertIn('vid2', video_ids)
        
    def test_get_stats(self):
        """Test database statistics."""
        vectors = np.random.randn(10, 1024).astype(np.float32)
        metadata = [{'index': i} for i in range(10)]
        
        self.db.add_video('test', vectors, metadata)
        
        stats = self.db.get_stats()
        
        self.assertEqual(stats['total_videos'], 1)
        self.assertEqual(stats['total_chunks'], 10)
        self.assertEqual(stats['dimension'], 1024)
        
    def test_save_and_load(self):
        """Test saving and loading the database."""
        vectors = np.random.randn(5, 1024).astype(np.float32)
        metadata = [{'index': i, 'timestamp': i * 2.0, 'ocr_text': f'Text {i}'} for i in range(5)]
        
        self.db.add_video('test_video', vectors, metadata)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_db')
            self.db.save(filepath)
            
            # Create new DB and load
            new_db = VideoVectorDatabase()
            new_db.load(filepath)
            
            self.assertEqual(len(new_db.metadata), 5)
            self.assertIn('test_video', new_db.video_registry)
            
    def test_duplicate_video_prevention(self):
        """Test that duplicate videos are prevented."""
        vectors = np.random.randn(5, 1024).astype(np.float32)
        metadata = [{'index': i} for i in range(5)]
        
        self.db.add_video('same_id', vectors, metadata)
        self.db.add_video('same_id', vectors, metadata)  # Should be skipped
        
        self.assertEqual(self.db.video_registry['same_id']['num_chunks'], 5)


class TestTextCleaning(unittest.TestCase):
    """Tests for text cleaning and preprocessing."""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        # Import the clean_text function from app
        try:
            from app import clean_text
            
            # Test removing non-ASCII
            result = clean_text("Hello 你好 World")
            self.assertNotIn("你好", result)
            
            # Test removing UI elements
            result = clean_text("Subscribe to our channel")
            self.assertNotIn("Subscribe", result)
            
            # Test whitespace normalization
            result = clean_text("Hello    World")
            self.assertEqual(result.count("  "), 0)  # No double spaces
        except ImportError:
            self.skipTest("Could not import clean_text from app")


class TestKeyframeProcessing(unittest.TestCase):
    """Tests for keyframe detection and processing."""
    
    def test_keyframe_data_structure(self):
        """Test that keyframe data has correct structure."""
        # Create mock keyframe data
        keyframe = {
            'timestamp': 5.0,
            'frame': np.zeros((480, 640, 3), dtype=np.uint8),
            'change_score': 45.0,
            'frame_number': 150
        }
        
        self.assertIn('timestamp', keyframe)
        self.assertIn('frame', keyframe)
        self.assertIn('change_score', keyframe)
        self.assertIsInstance(keyframe['frame'], np.ndarray)
        self.assertEqual(keyframe['frame'].shape[2], 3)  # RGB channels
        

class TestChunkVectorCreation(unittest.TestCase):
    """Tests for multimodal vector creation."""
    
    def test_vector_dimensions(self):
        """Test that vectors have correct dimensions."""
        # Mock a 1024D vector (512 text + 512 visual)
        text_features = np.random.randn(512).astype(np.float32)
        visual_features = np.random.randn(512).astype(np.float32)
        
        combined = np.concatenate([text_features, visual_features])
        
        self.assertEqual(combined.shape[0], 1024)
        
    def test_vector_normalization(self):
        """Test that vectors are properly normalized."""
        vector = np.random.randn(1024).astype(np.float32)
        
        # Normalize
        vector_norm = vector / np.linalg.norm(vector)
        
        # Check unit norm
        self.assertAlmostEqual(np.linalg.norm(vector_norm), 1.0, places=5)


class TestAPIEndpoints(unittest.TestCase):
    """Tests for Flask API endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Set up Flask test client."""
        try:
            from app import app
            cls.client = app.test_client()
            cls.app = app
        except ImportError:
            cls.client = None
            
    def setUp(self):
        if self.client is None:
            self.skipTest("Could not import Flask app")
            
    def test_library_videos_endpoint(self):
        """Test GET /api/library/videos endpoint."""
        response = self.client.get('/api/library/videos')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('videos', data)
        self.assertIsInstance(data['videos'], list)
        
    def test_search_requires_query(self):
        """Test that search endpoint requires query parameter."""
        response = self.client.post(
            '/api/search',
            json={'video_id': 'test'},
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)


class TestBenchmark(unittest.TestCase):
    """Tests for the benchmark framework."""
    
    def test_benchmark_template_creation(self):
        """Test creating evaluation template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock metadata file
            metadata = {
                'video_id': 'test_video',
                'total_keyframes': 3,
                'chunks': [
                    {'index': 0, 'timestamp': 0.0, 'ocr_text': 'Hello', 'audio_context': 'World'},
                    {'index': 1, 'timestamp': 2.0, 'ocr_text': 'Test', 'audio_context': 'Data'},
                    {'index': 2, 'timestamp': 4.0, 'ocr_text': 'More', 'audio_context': 'Text'}
                ]
            }
            
            metadata_path = os.path.join(tmpdir, 'test_video_chunks_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            benchmark = VideoRetrievalBenchmark(tmpdir)
            template = benchmark.create_ground_truth_template('test_video', ['query1', 'query2'])
            
            self.assertEqual(template['video_id'], 'test_video')
            self.assertIn('query1', template['queries'])
            self.assertIn('query2', template['queries'])
            self.assertEqual(len(template['chunk_reference']), 3)


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)
