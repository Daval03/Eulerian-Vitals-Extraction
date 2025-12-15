# test_evm_core_simple.py
import unittest
import numpy as np

from src.evm.evm_core import EVMProcessor

class TestEVMCoreSimple(unittest.TestCase):
    
    def test_basic_functionality(self):
        """Basic test of EVMProcessor functionality"""
        # Create test video frames
        video_frames = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            for _ in range(35)  # More than minimum 30 frames
        ]
        
        # Create processor
        processor = EVMProcessor()
        
        # Process video
        hr_signal, rr_signal = processor.process_dual_band(video_frames)
        
        # Check results
        self.assertIsNotNone(hr_signal)
        self.assertIsNotNone(rr_signal)
        self.assertEqual(len(hr_signal), len(video_frames))
        self.assertEqual(len(rr_signal), len(video_frames))
    
    def test_insufficient_frames(self):
        """Test with too few frames"""
        video_frames = [
            np.zeros((64, 64, 3), dtype=np.uint8)
            for _ in range(10)  # Less than minimum 30 frames
        ]
        
        processor = EVMProcessor()
        hr_signal, rr_signal = processor.process_dual_band(video_frames)
        
        self.assertIsNone(hr_signal)
        self.assertIsNone(rr_signal)
    
    def test_custom_parameters(self):
        """Test with custom parameters"""
        video_frames = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            for _ in range(40)
        ]
        
        # Custom parameters
        processor = EVMProcessor(
            levels=4,
            alpha_hr=75.0,
            alpha_rr=150.0
        )
        
        hr_signal, rr_signal = processor.process_dual_band(video_frames)
        
        # Should still process successfully
        if hr_signal is not None and rr_signal is not None:
            self.assertEqual(len(hr_signal), len(video_frames))
            self.assertEqual(len(rr_signal), len(video_frames))
    
    def test_invalid_input(self):
        """Test with invalid input"""
        processor = EVMProcessor()
        
        # Test with None
        hr_signal, rr_signal = processor.process_dual_band(None)
        self.assertIsNone(hr_signal)
        self.assertIsNone(rr_signal)
        
        # Test with empty list
        hr_signal, rr_signal = processor.process_dual_band([])
        self.assertIsNone(hr_signal)
        self.assertIsNone(rr_signal)
    
    def test_signal_characteristics(self):
        """Test that signals have reasonable characteristics"""
        video_frames = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            for _ in range(50)
        ]
        
        processor = EVMProcessor()
        hr_signal, rr_signal = processor.process_dual_band(video_frames)
        
        if hr_signal is not None and rr_signal is not None:
            # Signals should not contain NaN or Inf
            self.assertFalse(np.any(np.isnan(hr_signal)))
            self.assertFalse(np.any(np.isnan(rr_signal)))
            self.assertFalse(np.any(np.isinf(hr_signal)))
            self.assertFalse(np.any(np.isinf(rr_signal)))
            
            # Signals should have some variation (not constant)
            self.assertGreater(np.std(hr_signal), 0)
            self.assertGreater(np.std(rr_signal), 0)
