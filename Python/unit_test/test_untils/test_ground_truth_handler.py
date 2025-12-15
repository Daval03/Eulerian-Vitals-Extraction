import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import the class to test
from src.utils.ground_truth_handler import GroundTruthHandler, setup_ground_truth

class TestGroundTruthHandler(unittest.TestCase):
    """Test cases for GroundTruthHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gt_path = "dummy_path.txt"
        self.frame_chunk_size = 10
        self.handler = GroundTruthHandler(self.gt_path, self.frame_chunk_size)
        
        # Sample ground truth data for testing
        self.sample_gt_data = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            [70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        ])
        
        # Mock the numpy loadtxt to return sample data
        self.loadtxt_patcher = patch('numpy.loadtxt')
        self.mock_loadtxt = self.loadtxt_patcher.start()
        self.mock_loadtxt.return_value = self.sample_gt_data
        
    def tearDown(self):
        """Clean up after tests."""
        self.loadtxt_patcher.stop()
    
    def test_initialization(self):
        """Test initialization of GroundTruthHandler."""
        self.assertEqual(self.handler.gt_path, self.gt_path)
        self.assertEqual(self.handler.frame_chunk_size, self.frame_chunk_size)
        self.assertIsNone(self.handler.gt_data)
        self.assertIsNone(self.handler.gt_hr)
        self.assertIsNone(self.handler.gt_chunk_hrs)
    
    def test_load_ground_truth_success(self):
        """Test successful loading of ground truth data."""
        # Call the method
        result = self.handler.load_ground_truth()
        
        # Verify results
        self.assertTrue(result)
        self.assertTrue(np.array_equal(self.handler.gt_data, self.sample_gt_data))
        self.assertTrue(np.array_equal(self.handler.gt_hr, self.sample_gt_data[1, :]))
        
        # Verify chunk averages were calculated
        # With 15 data points and chunk size 10, we should have 1 complete chunk
        self.assertEqual(len(self.handler.gt_chunk_hrs), 1)
        expected_chunk_avg = np.mean(self.sample_gt_data[1, :10])  # Mean of first 10 HR values
        self.assertAlmostEqual(self.handler.gt_chunk_hrs[0], expected_chunk_avg)
    
    def test_load_ground_truth_failure(self):
        """Test handling of file loading failure."""
        # Make loadtxt raise an exception
        self.mock_loadtxt.side_effect = Exception("File not found")
        
        # Call the method
        result = self.handler.load_ground_truth()
        
        # Verify failure
        self.assertFalse(result)
        self.assertIsNone(self.handler.gt_data)
        self.assertIsNone(self.handler.gt_hr)
        self.assertIsNone(self.handler.gt_chunk_hrs)
    
    def test_calculate_chunk_averages(self):
        """Test calculation of chunk averages."""
        # Manually set up the data
        self.handler.gt_hr = np.array([60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115])
        self.handler.frame_chunk_size = 3
        
        # Call the private method (we'll use reflection to test it)
        self.handler._calculate_chunk_averages()
        
        # Verify chunk averages
        # With 12 data points and chunk size 3, we should have 4 chunks
        self.assertEqual(len(self.handler.gt_chunk_hrs), 4)
        
        # Calculate expected averages
        expected_chunk1 = np.mean([60, 65, 70])  # 65.0
        expected_chunk2 = np.mean([75, 80, 85])  # 80.0
        expected_chunk3 = np.mean([90, 95, 100])  # 95.0
        expected_chunk4 = np.mean([105, 110, 115])  # 110.0
        
        self.assertAlmostEqual(self.handler.gt_chunk_hrs[0], expected_chunk1)
        self.assertAlmostEqual(self.handler.gt_chunk_hrs[1], expected_chunk2)
        self.assertAlmostEqual(self.handler.gt_chunk_hrs[2], expected_chunk3)
        self.assertAlmostEqual(self.handler.gt_chunk_hrs[3], expected_chunk4)
    
    def test_calculate_chunk_averages_edge_case(self):
        """Test chunk average calculation with edge cases."""
        # Test with fewer points than chunk size
        self.handler.gt_hr = np.array([60, 65, 70])  # Only 3 points
        self.handler.frame_chunk_size = 5  # Chunk size larger than data
        
        self.handler._calculate_chunk_averages()
        
        # Should have 0 complete chunks
        self.assertEqual(len(self.handler.gt_chunk_hrs), 0)
    
    def test_get_hr_for_chunk_valid(self):
        """Test getting HR for a valid chunk index."""
        # Set up mock data
        self.handler.gt_chunk_hrs = [65.0, 75.0, 85.0, 95.0]
        
        # Test valid indices
        self.assertEqual(self.handler.get_hr_for_chunk(0), 65.0)
        self.assertEqual(self.handler.get_hr_for_chunk(2), 85.0)
        self.assertEqual(self.handler.get_hr_for_chunk(3), 95.0)
    
    def test_get_hr_for_chunk_invalid(self):
        """Test getting HR for invalid chunk indices."""
        # Test with None data
        self.assertIsNone(self.handler.get_hr_for_chunk(0))
        
        # Test with data but invalid index
        self.handler.gt_chunk_hrs = [65.0, 75.0]
        self.assertIsNone(self.handler.get_hr_for_chunk(2))  # Index out of bounds
        self.assertIsNone(self.handler.get_hr_for_chunk(-1))  # Negative index
    
    def test_calculate_error_valid(self):
        """Test error calculation with valid data."""
        # Set up mock data
        self.handler.gt_chunk_hrs = [70.0, 75.0, 80.0]
        
        # Test error calculation
        error, true_hr = self.handler.calculate_error(72.5, 0)  # Estimated 72.5, chunk 0 true=70.0
        
        self.assertAlmostEqual(error, 2.5)  # |72.5 - 70.0| = 2.5
        self.assertAlmostEqual(true_hr, 70.0)
        
        # Test another chunk
        error, true_hr = self.handler.calculate_error(78.0, 2)  # Estimated 78.0, chunk 2 true=80.0
        
        self.assertAlmostEqual(error, 2.0)  # |78.0 - 80.0| = 2.0
        self.assertAlmostEqual(true_hr, 80.0)
    
    def test_calculate_error_invalid(self):
        """Test error calculation with invalid data."""
        # Test with None chunk data
        error, true_hr = self.handler.calculate_error(72.5, 0)
        self.assertIsNone(error)
        self.assertIsNone(true_hr)
        
        # Test with valid chunk data but invalid index
        self.handler.gt_chunk_hrs = [70.0, 75.0]
        error, true_hr = self.handler.calculate_error(72.5, 5)  # Invalid index
        self.assertIsNone(error)
        self.assertIsNone(true_hr)
        
        # Test with None estimated HR
        error, true_hr = self.handler.calculate_error(None, 0)
        self.assertIsNone(error)
        self.assertIsNone(true_hr)
    
    def test_get_available_chunks_count(self):
        """Test getting the count of available chunks."""
        # Test with no data
        self.assertEqual(self.handler.get_available_chunks_count(), 0)
        
        # Test with data
        self.handler.gt_chunk_hrs = [70.0, 75.0, 80.0, 85.0]
        self.assertEqual(self.handler.get_available_chunks_count(), 4)
        
        # Test with empty list
        self.handler.gt_chunk_hrs = []
        self.assertEqual(self.handler.get_available_chunks_count(), 0)
    
    def test_get_summary_stats_with_data(self):
        """Test getting summary statistics with data."""
        # Load data first
        self.handler.load_ground_truth()
        
        # Get summary stats
        stats = self.handler.get_summary_stats()
        
        # Verify stats
        self.assertEqual(stats['total_points'], len(self.sample_gt_data[1, :]))
        self.assertEqual(stats['total_chunks'], 1)  # 15 points / 10 chunk size = 1 chunk
        self.assertAlmostEqual(stats['hr_min'], np.min(self.sample_gt_data[1, :]))
        self.assertAlmostEqual(stats['hr_max'], np.max(self.sample_gt_data[1, :]))
        self.assertAlmostEqual(stats['hr_mean'], np.mean(self.sample_gt_data[1, :]))
        self.assertAlmostEqual(stats['hr_std'], np.std(self.sample_gt_data[1, :]))
    
    def test_get_summary_stats_no_data(self):
        """Test getting summary statistics without data."""
        stats = self.handler.get_summary_stats()
        
        # Should return empty dict when no data
        self.assertEqual(stats, {})


class TestSetupGroundTruthUtility(unittest.TestCase):
    """Test cases for the setup_ground_truth utility function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.patcher = patch('src.utils.ground_truth_handler.GroundTruthHandler')
        self.mock_handler_class = self.patcher.start()
        
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
    
    def test_setup_ground_truth_success(self):
        """Test successful setup of ground truth."""
        
        # Mock the handler instance
        mock_handler = MagicMock()
        mock_handler.load_ground_truth.return_value = True
        self.mock_handler_class.return_value = mock_handler
        
        # Call the function
        subject_num = 1
        dataset_path = "/test/dataset"
        buffer_size = 30
        
        gt_handler, video_path = setup_ground_truth(subject_num, dataset_path, buffer_size)
        
        # Verify results
        self.assertEqual(gt_handler, mock_handler)
        expected_video_path = f"{dataset_path}/subject{subject_num}/vid.mp4"
        self.assertEqual(video_path, expected_video_path)
        
        # Verify handler was initialized with correct arguments
        expected_gt_path = f"{dataset_path}/subject{subject_num}/ground_truth.txt"
        self.mock_handler_class.assert_called_once_with(expected_gt_path, buffer_size)
        
        # Verify load_ground_truth was called
        mock_handler.load_ground_truth.assert_called_once()
    
    def test_setup_ground_truth_failure(self):
        """Test setup when ground truth loading fails."""
        
        # Mock the handler instance
        mock_handler = MagicMock()
        mock_handler.load_ground_truth.return_value = False
        self.mock_handler_class.return_value = mock_handler
        
        # Call the function
        gt_handler, video_path = setup_ground_truth(2, "/test/dataset", 30)
        
        # Should still return the handler even if loading failed
        self.assertEqual(gt_handler, mock_handler)
        # Warning message would be printed but we can't easily test that in unit test


class TestGroundTruthHandlerWithRealFile(unittest.TestCase):
    """Test GroundTruthHandler with a real temporary file."""
    
    def setUp(self):
        """Create a temporary ground truth file for testing."""
        # Create a temporary file with ground truth data
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        
        # Write sample data to the file
        # Format: 3 rows, 10 columns each
        data = ""
        for row in range(3):
            row_data = [str(i + row * 100) for i in range(10)]
            data += "   ".join(row_data) + "\n"
        self.temp_file.write(data)
        self.temp_file.close()
        
        # Initialize handler with the temp file
        self.handler = GroundTruthHandler(self.temp_file.name, 5)
    
    def tearDown(self):
        """Clean up temporary file."""
        os.unlink(self.temp_file.name)
    
    def test_load_real_file(self):
        """Test loading from a real file."""
        result = self.handler.load_ground_truth()
        
        self.assertTrue(result)
        self.assertIsNotNone(self.handler.gt_data)
        self.assertIsNotNone(self.handler.gt_hr)
        self.assertIsNotNone(self.handler.gt_chunk_hrs)
        
        # Verify data shape (3 rows, 10 columns)
        self.assertEqual(self.handler.gt_data.shape, (3, 10))
        
        # Verify HR data (second row)
        expected_hr = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0])
        np.testing.assert_array_almost_equal(self.handler.gt_hr, expected_hr)
        
        # Verify chunk averages (10 points / chunk size 5 = 2 chunks)
        self.assertEqual(len(self.handler.gt_chunk_hrs), 2)
        expected_chunk1 = np.mean([100.0, 101.0, 102.0, 103.0, 104.0])  # 102.0
        expected_chunk2 = np.mean([105.0, 106.0, 107.0, 108.0, 109.0])  # 107.0
        self.assertAlmostEqual(self.handler.gt_chunk_hrs[0], expected_chunk1)
        self.assertAlmostEqual(self.handler.gt_chunk_hrs[1], expected_chunk2)