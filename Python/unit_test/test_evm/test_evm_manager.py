import unittest
import numpy as np
from src.evm.evm_manager import process_video_evm_vital_signs

class TestEVMManager(unittest.TestCase):
    """
    Test suite for the EVM manager's core functionality.
    
    This class tests various scenarios including valid inputs, edge cases,
    and error conditions for the EVM-based vital signs extraction.
    """
    
    def setUp(self):
        """
        Set up common test fixtures before each test method.
        
        Creates synthetic video frames for testing:
        - valid_video_frames: 50 random frames (sufficient for processing)
        - insufficient_frames: 20 zero frames (below minimum threshold)
        """
        # Create synthetic video frames for testing
        self.valid_video_frames = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            for _ in range(50)  # Enough frames (>30 minimum)
        ]
        
        self.insufficient_frames = [
            np.zeros((100, 100, 3), dtype=np.uint8)
            for _ in range(20)  # Insufficient frames (<30 minimum)
        ]
    
    def test_basic_functionality(self):
        """
        Test basic functionality with valid video input.
        
        Verifies:
        - Function returns a dictionary
        - Dictionary contains expected keys ('heart_rate', 'respiratory_rate')
        - Results may be None if signals aren't detected (acceptable behavior)
        """
        # Process valid video
        results = process_video_evm_vital_signs(self.valid_video_frames)
        
        # Verify result structure
        self.assertIsInstance(results, dict)
        self.assertIn('heart_rate', results)
        self.assertIn('respiratory_rate', results)
        
    def test_insufficient_frames(self):
        """
        Test with insufficient number of frames.
        
        Verifies that both heart rate and respiratory rate return None
        when provided with fewer than the minimum required frames.
        """
        results = process_video_evm_vital_signs(self.insufficient_frames)
        
        # Both results should be None
        self.assertIsNone(results['heart_rate'])
        self.assertIsNone(results['respiratory_rate'])
    
    def test_empty_video(self):
        """
        Test with empty video input.
        
        Tests edge cases:
        - Empty list input
        - None input
        Both should return None for both vital signs.
        """
        # Test with empty list
        results = process_video_evm_vital_signs([])
        self.assertIsNone(results['heart_rate'])
        self.assertIsNone(results['respiratory_rate'])
        
        # Test with None input
        results = process_video_evm_vital_signs(None)
        self.assertIsNone(results['heart_rate'])
        self.assertIsNone(results['respiratory_rate'])
    
    def test_verbose_mode(self):
        """
        Test with verbose mode enabled.
        
        Verifies that the function doesn't raise exceptions when verbose=True.
        This test doesn't check console output, only ensures no crashes.
        """
        # This test verifies there are no errors with verbose=True
        # We can capture output or simply verify no exception is raised
        try:
            results = process_video_evm_vital_signs(self.valid_video_frames, verbose=True)
            self.assertIsInstance(results, dict)
        except Exception as e:
            self.fail(f"verbose=True caused an exception: {e}")
    
    def test_silent_mode(self):
        """
        Test with verbose mode disabled (default).
        
        Verifies normal operation with the default silent mode.
        """
        results = process_video_evm_vital_signs(self.valid_video_frames, verbose=False)
        self.assertIsInstance(results, dict)
    
    def test_minimum_frames(self):
        """
        Test with exactly the minimum number of frames (30).
        
        Verifies that processing is attempted with the minimum required frames,
        though results may still be None if signals aren't detected.
        """
        minimum_frames = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            for _ in range(30)  # Exactly the minimum
        ]
        
        results = process_video_evm_vital_signs(minimum_frames)
        # Processing should be attempted, but results may be None
        self.assertIsNotNone(results)
    
    def test_result_types(self):
        """
        Test that result types are correct.
        
        Verifies:
        - heart_rate is either float/int or None
        - respiratory_rate is either float/int or None
        Also prints actual values for debugging/inspection.
        """
        results = process_video_evm_vital_signs(self.valid_video_frames)
        
        # heart_rate can be float or None
        if results['heart_rate'] is not None:
            self.assertIsInstance(results['heart_rate'], (float, int))

        # respiratory_rate can be float or None
        if results['respiratory_rate'] is not None:
            self.assertIsInstance(results['respiratory_rate'], (float, int))
    
    def test_different_frame_sizes(self):
        """
        Test with different frame sizes.
        
        Verifies that the function handles various frame dimensions correctly.
        Tests multiple resolutions to ensure robustness.
        """
        sizes = [(64, 64), (128, 128), (256, 256)]
        
        for height, width in sizes:
            with self.subTest(size=f"{height}x{width}"):
                frames = [
                    np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                    for _ in range(40)
                ]
                
                results = process_video_evm_vital_signs(frames)
                self.assertIsInstance(results, dict)
    
    def test_consistent_results(self):
        """
        Test consistency across multiple executions.
        
        Runs the same video through the processor multiple times and
        verifies that the result structure remains consistent.
        Prints results to show variability for debugging.
        """
        # Execute multiple times with the same frames
        all_results = []
        for i in range(3):
            results = process_video_evm_vital_signs(self.valid_video_frames)
            all_results.append(results)
        
        # Results should have the same structure
        for results in all_results:
            self.assertIn('heart_rate', results)
            self.assertIn('respiratory_rate', results)
            
    def test_exception_handling(self):
        """
        Test that exceptions are properly handled.
        
        Provides invalid input (non-numpy arrays) and verifies that
        the function gracefully handles errors by returning empty results.
        """
        # Create invalid frames (but with sufficient length)
        invalid_frames = [
            "not a numpy array"  # This will cause processing errors
            for _ in range(40)
        ]
        
        # The handler should catch exceptions and return empty results
        results = process_video_evm_vital_signs(invalid_frames)
        self.assertIsNone(results['heart_rate'])
        self.assertIsNone(results['respiratory_rate'])


# Optional integration tests
class TestEVMManagerIntegration(unittest.TestCase):
    """
    Integration test suite for more realistic video patterns.
    
    These tests simulate more realistic video scenarios with
    periodic variations that might contain actual signal patterns.
    """
    
    def test_with_realistic_video_pattern(self):
        """
        Test with a more realistic video pattern.
        
        Creates frames with small periodic variations to simulate
        potential vital signs signals (e.g., color changes from blood flow).
        
        Verifies that processing completes without errors when presented
        with patterned input similar to real video data.
        """
        # Create frames with slight temporal variation to simulate signals
        frames = []
        base_frame = np.random.randint(100, 150, (100, 100, 3), dtype=np.uint8)
        
        for i in range(60):  # 60 frames at 30 FPS = 2 seconds
            # Add small periodic variation
            variation = np.sin(i * 0.1) * 5  # Sinusoidal variation
            frame = np.clip(base_frame + variation, 0, 255).astype(np.uint8)
            frames.append(frame)
        
        results = process_video_evm_vital_signs(frames)
        
        # Only verify no errors occurred
        self.assertIsInstance(results, dict)