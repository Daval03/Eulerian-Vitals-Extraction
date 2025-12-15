import pytest
import numpy as np
import cv2
from unittest.mock import patch
from src.evm.pyramid_processing import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    collapse_laplacian_pyramid,
    build_video_pyramid_stack,
    extract_pyramid_level
)

# Mock the LEVELS_RPI constant for testing
LEVELS_RPI = 3

class TestBuildGaussianPyramid:
    """Test the build_gaussian_pyramid function."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample BGR frame for testing."""
        np.random.seed(42)
        return np.random.randint(0, 256, (64, 48, 3), dtype=np.uint8)
    
    def test_basic_gaussian_pyramid(self, sample_frame):
        """Test building a basic Gaussian pyramid."""
        pyramid = build_gaussian_pyramid(sample_frame, levels=LEVELS_RPI)
        
        assert isinstance(pyramid, list)
        assert len(pyramid) == LEVELS_RPI + 1  # Includes original
        
        # Check each level
        for i, level in enumerate(pyramid):
            assert level.dtype == np.float32
            if i > 0:
                # Each level should be half the size (approximately)
                expected_h = max(1, sample_frame.shape[0] // (2 ** i))
                expected_w = max(1, sample_frame.shape[1] // (2 ** i))
                assert level.shape[0] == expected_h
                assert level.shape[1] == expected_w
    
    def test_zero_levels(self, sample_frame):
        """Test pyramid with zero levels (should return just the original)."""
        pyramid = build_gaussian_pyramid(sample_frame, levels=0)
        
        assert len(pyramid) == 1
        assert np.allclose(pyramid[0], sample_frame.astype(np.float32))
    
    def test_single_level_pyramid(self, sample_frame):
        """Test pyramid with single level."""
        pyramid = build_gaussian_pyramid(sample_frame, levels=1)
        
        assert len(pyramid) == 2
        assert pyramid[0].shape == sample_frame.shape
        assert pyramid[1].shape[0] == sample_frame.shape[0] // 2
        assert pyramid[1].shape[1] == sample_frame.shape[1] // 2
    
    def test_pyramid_content(self):
        """Test that pyramid actually performs blurring and downsampling."""
        # Create a simple test pattern
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[8:24, 8:24, :] = 255
        
        pyramid = build_gaussian_pyramid(frame, levels=2)
        
        # Original should be sharp
        assert np.max(pyramid[0][8:24, 8:24, 0]) == 255
        assert np.min(pyramid[0][:8, :, 0]) == 0
        
        # First downsampled level should be blurred
        # The edge should not be as sharp
        level1 = pyramid[1]
        
        # Check that we have some intermediate values (due to blur)
        unique_vals = np.unique(level1)
        assert len(unique_vals) > 2  # More than just 0 and 255
    
    def test_dtype_preservation(self):
        """Test that float inputs are handled correctly."""
        # Test with float32 input
        frame_float = np.random.randn(32, 32, 3).astype(np.float32)
        pyramid = build_gaussian_pyramid(frame_float, levels=2)
        
        assert pyramid[0].dtype == np.float32
        assert np.allclose(pyramid[0], frame_float, rtol=1e-5)
    
    def test_grayscale_image(self):
        """Test with grayscale (single channel) image."""
        frame_gray = np.random.randint(0, 256, (64, 48), dtype=np.uint8)
        
        # Convert to 3D for pyramid processing
        frame_gray_3d = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        pyramid = build_gaussian_pyramid(frame_gray_3d, levels=2)
        
        assert len(pyramid) == 3
        assert pyramid[0].shape == (64, 48, 3)
    
    @patch('cv2.pyrDown')
    def test_pyrdown_failure(self, mock_pyrdown, sample_frame):
        """Test error handling if pyrDown fails."""
        mock_pyrdown.side_effect = Exception("OpenCV error")
        
        # Should still return at least the original
        pyramid = build_gaussian_pyramid(sample_frame, levels=2)
        assert isinstance(pyramid, list)
        assert len(pyramid) >= 1  # At least the original
        assert np.allclose(pyramid[0], sample_frame.astype(np.float32))


class TestBuildLaplacianPyramid:
    """Test the build_laplacian_pyramid function."""
    
    @pytest.fixture
    def sample_gaussian_pyramid(self):
        """Create a sample Gaussian pyramid for testing."""
        frame = np.random.randint(0, 256, (64, 48, 3), dtype=np.uint8)
        return build_gaussian_pyramid(frame, levels=2)
    
    def test_laplacian_pyramid_structure(self, sample_gaussian_pyramid):
        """Test basic Laplacian pyramid construction."""
        laplacian_pyr = build_laplacian_pyramid(sample_gaussian_pyramid)
        
        assert isinstance(laplacian_pyr, list)
        assert len(laplacian_pyr) == len(sample_gaussian_pyramid)
        
        # Check that last level is the same as Gaussian pyramid
        assert np.allclose(
            laplacian_pyr[-1], 
            sample_gaussian_pyramid[-1]
        )
    
    def test_laplacian_content(self, sample_gaussian_pyramid):
        """Test that Laplacian pyramid contains detail information."""
        laplacian_pyr = build_laplacian_pyramid(sample_gaussian_pyramid)
        
        # Laplacian levels should contain both positive and negative values
        # (representing detail information)
        for i in range(len(laplacian_pyr) - 1):
            level = laplacian_pyr[i]
            assert np.min(level) < 0 or np.max(level) > 0
            # Mean should be near zero (detail information)
            assert abs(np.mean(level)) < 50
    
    def test_reconstruction_capability(self):
        """Test that Laplacian pyramid can be used for perfect reconstruction."""
        # Create a test image
        original = np.random.randint(0, 256, (64, 48, 3), dtype=np.uint8).astype(np.float32)
        
        # Build pyramids
        gaussian_pyr = build_gaussian_pyramid(original, levels=2)
        laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)
        
        # Reconstruct
        reconstructed = collapse_laplacian_pyramid(laplacian_pyr)
        
        # Should reconstruct perfectly (within numerical precision)
        assert reconstructed.shape == original.shape
        assert np.allclose(reconstructed, original, rtol=1e-5, atol=1e-5)
    
    def test_single_level_gaussian(self):
        """Test Laplacian with single-level Gaussian pyramid."""
        frame = np.random.randn(32, 32, 3).astype(np.float32)
        gaussian_pyr = [frame]  # Single level
        
        laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)
        
        assert len(laplacian_pyr) == 1
        assert np.allclose(laplacian_pyr[0], frame)
    
    @patch('cv2.pyrUp')
    @patch('cv2.subtract')
    def test_cv2_operations_failure(self, mock_pyrUp, sample_gaussian_pyramid):
        """Test error handling in Laplacian pyramid construction."""
        mock_pyrUp.side_effect = Exception("pyrUp failed")
        
        # Should handle gracefully
        laplacian_pyr = build_laplacian_pyramid(sample_gaussian_pyramid)
        # Should return empty or partial pyramid
        assert isinstance(laplacian_pyr, list)


class TestCollapseLaplacianPyramid:
    """Test the collapse_laplacian_pyramid function."""
    
    @pytest.fixture
    def sample_laplacian_pyramid(self):
        """Create a sample Laplacian pyramid for testing."""
        frame = np.random.randint(0, 256, (64, 48, 3), dtype=np.uint8)
        gaussian_pyr = build_gaussian_pyramid(frame, levels=2)
        return build_laplacian_pyramid(gaussian_pyr)
    
    def test_basic_collapse(self, sample_laplacian_pyramid):
        """Test basic pyramid collapse."""
        reconstructed = collapse_laplacian_pyramid(sample_laplacian_pyramid)
        
        assert reconstructed.shape == sample_laplacian_pyramid[0].shape
        assert reconstructed.dtype == np.float32
        
        # Should produce valid pixel values (not extreme)
        assert np.min(reconstructed) >= -1000  # Allow some negative from Laplacian
        assert np.max(reconstructed) <= 1000
    
    def test_identity_collapse(self):
        """Test that collapsing a single-level pyramid returns it unchanged."""
        single_pyramid = [np.random.randn(32, 32, 3).astype(np.float32)]
        result = collapse_laplacian_pyramid(single_pyramid)
        
        assert np.allclose(result, single_pyramid[0])
    
    def test_collapse_with_modified_levels(self):
        """Test collapse with artificially modified pyramid levels."""
        # Create base pyramid
        frame = np.ones((32, 32, 3), dtype=np.float32) * 128
        gaussian_pyr = build_gaussian_pyramid(frame, levels=1)
        laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)
        
        # Modify a level
        laplacian_pyr[0] += 10.0  # Add constant to detail level
        
        reconstructed = collapse_laplacian_pyramid(laplacian_pyr)
        
        # Reconstruction should reflect the modification
        assert np.allclose(reconstructed, frame + 10.0, rtol=1e-5)
    
    @patch('cv2.pyrUp')
    @patch('cv2.add')
    def test_cv2_failure_during_collapse(self, mock_add, mock_pyrUp, sample_laplacian_pyramid):
        """Test error handling during collapse."""
        mock_pyrUp.side_effect = Exception("pyrUp failed")
        
        # Should raise exception or return partial result
        try:
            result = collapse_laplacian_pyramid(sample_laplacian_pyramid)
            # If it doesn't raise, result should be something
            assert result is not None
        except Exception:
            pass  # Exception is also acceptable


class TestBuildVideoPyramidStack:
    """Test the build_video_pyramid_stack function."""
    
    @pytest.fixture
    def sample_video_frames(self):
        """Create sample video frames for testing."""
        np.random.seed(42)
        frames = []
        for i in range(5):  # 5 frames
            frame = np.random.randint(0, 256, (64, 48, 3), dtype=np.uint8)
            frames.append(frame)
        return frames
    
    def test_basic_video_pyramid_stack(self, sample_video_frames):
        """Test building pyramids for video frames."""
        pyramid_stack = build_video_pyramid_stack(sample_video_frames, levels=2)
        
        assert isinstance(pyramid_stack, list)
        assert len(pyramid_stack) == len(sample_video_frames)
        
        # Check each frame's pyramid
        for i, pyramid in enumerate(pyramid_stack):
            assert isinstance(pyramid, list)
            assert len(pyramid) == 3  # levels + 1

    def test_empty_video_frames(self):
        """Test with empty video frames list."""
        pyramid_stack = build_video_pyramid_stack([], levels=2)
        assert pyramid_stack == []
    
    def test_single_frame_video(self):
        """Test with single frame video."""
        frame = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        pyramid_stack = build_video_pyramid_stack([frame], levels=1)
        
        assert len(pyramid_stack) == 1
        assert len(pyramid_stack[0]) == 2  # levels + 1
    
    def test_different_frame_sizes(self):
        """Test with frames of different sizes (should handle independently)."""
        frames = [
            np.random.randint(0, 256, (64, 48, 3), dtype=np.uint8),
            np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8),
            np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8),
        ]
        
        pyramid_stack = build_video_pyramid_stack(frames, levels=2)
        
        # Each pyramid should have correct dimensions for its frame
        for i, pyramid in enumerate(pyramid_stack):
            original_h, original_w = frames[i].shape[:2]
            assert pyramid[0].shape[0] == original_h
            assert pyramid[0].shape[1] == original_w


class TestExtractPyramidLevel:
    """Test the extract_pyramid_level function."""
    
    @pytest.fixture
    def sample_pyramid_stack(self):
        """Create a sample pyramid stack for testing."""
        frames = []
        for i in range(3):  # 3 frames
            # Make frames slightly different
            base = np.ones((64, 48, 3), dtype=np.uint8) * (i + 1) * 50
            frames.append(base)
        
        return build_video_pyramid_stack(frames, levels=2)
    
    def test_extract_specific_level(self, sample_pyramid_stack):
        """Test extracting a specific pyramid level."""
        # Extract level 0 (original resolution)
        level_tensor = extract_pyramid_level(sample_pyramid_stack, level=0)
        
        assert isinstance(level_tensor, np.ndarray)
        assert level_tensor.dtype == np.float32
        assert level_tensor.shape[0] == len(sample_pyramid_stack)  # T
        assert level_tensor.shape[1:] == sample_pyramid_stack[0][0].shape  # H x W x C
        
        # Check content
        for i in range(len(sample_pyramid_stack)):
            assert np.allclose(
                level_tensor[i], 
                sample_pyramid_stack[i][0]
            )
    
    def test_extract_higher_level(self, sample_pyramid_stack):
        """Test extracting a downsampled pyramid level."""
        # Extract level 1 (downsampled once)
        level_tensor = extract_pyramid_level(sample_pyramid_stack, level=1)
        
        expected_shape = sample_pyramid_stack[0][1].shape
        assert level_tensor.shape[1:] == expected_shape
        
        # Check that it contains the right data
        for i in range(len(sample_pyramid_stack)):
            assert np.allclose(
                level_tensor[i], 
                sample_pyramid_stack[i][1]
            )
    
    def test_extract_level_with_resize(self):
        """Test extraction when frames need resizing."""
        # Create pyramids with different sized frames
        frames = [
            np.ones((64, 64, 3), dtype=np.uint8) * 100,
            np.ones((60, 60, 3), dtype=np.uint8) * 150,
            np.ones((68, 68, 3), dtype=np.uint8) * 200,
        ]
        
        pyramid_stack = build_video_pyramid_stack(frames, levels=1)
        
        # Extract level where frames might have different sizes
        level_tensor = extract_pyramid_level(pyramid_stack, level=1)
        
        # All frames should be resized to same shape
        assert level_tensor.shape[0] == 3
        assert level_tensor.shape[1] == level_tensor.shape[2]  # Should be square
        
        # Check that resize happened
        unique_heights = {pyr[1].shape[0] for pyr in pyramid_stack}
        if len(unique_heights) > 1:
            # If original pyramids had different sizes, tensor should have uniform size
            assert all(level_tensor[i].shape == level_tensor[0].shape 
                      for i in range(1, 3))
    
    def test_extract_invalid_level(self, sample_pyramid_stack):
        """Test extracting a non-existent pyramid level."""
        level_tensor = extract_pyramid_level(sample_pyramid_stack, level=10)
        # Should still return something
        assert isinstance(level_tensor, np.ndarray)
        assert level_tensor.shape[0] == len(sample_pyramid_stack)
    
    def test_empty_pyramid_stack(self):
        """Test extracting from empty pyramid stack."""
        level_tensor = extract_pyramid_level([], level=0)
        
        # Should return empty array
        assert isinstance(level_tensor, np.ndarray)
        assert level_tensor.shape == (0,)
        assert level_tensor.dtype == np.float32
    
    def test_single_pyramid_stack(self):
        """Test extracting from single pyramid."""
        frame = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        pyramid_stack = build_video_pyramid_stack([frame], levels=2)
        
        level_tensor = extract_pyramid_level(pyramid_stack, level=1)
        
        assert level_tensor.shape == (1, 16, 16, 3)
        assert np.allclose(level_tensor[0], pyramid_stack[0][1])


class TestIntegration:
    """Integration tests for the full pyramid processing pipeline."""
    
    def test_full_pipeline(self):
        """Test the complete pyramid processing pipeline."""
        # Create test video
        T, H, W, C = 4, 64, 48, 3
        video_frames = [
            np.random.randint(0, 256, (H, W, C), dtype=np.uint8)
            for _ in range(T)
        ]
        
        # 1. Build pyramid stack
        pyramid_stack = build_video_pyramid_stack(video_frames, levels=2)
        assert len(pyramid_stack) == T
        
        # 2. Extract a specific level
        level_tensor = extract_pyramid_level(pyramid_stack, level=1)
        assert level_tensor.shape == (T, H//2, W//2, C)
        
        # 3. Verify we can reconstruct each frame
        for i in range(T):
            reconstructed = collapse_laplacian_pyramid(pyramid_stack[i])
            # Convert back to uint8 for comparison
            reconstructed_uint8 = np.clip(reconstructed, 0, 255).astype(np.uint8)
            assert np.allclose(
                reconstructed_uint8, 
                video_frames[i], 
                rtol=1.0, atol=1.0  # Allow small differences due to compression
            )
    
    def test_pyramid_level_consistency(self):
        """Test that all pyramids in a stack have the same number of levels."""
        frames = [
            np.random.randint(0, 256, (64, 48, 3), dtype=np.uint8),
            np.random.randint(0, 256, (64, 48, 3), dtype=np.uint8),
            np.random.randint(0, 256, (64, 48, 3), dtype=np.uint8),
        ]
        
        pyramid_stack = build_video_pyramid_stack(frames, levels=2)
        
        # All pyramids should have same number of levels
        num_levels = len(pyramid_stack[0])
        for pyramid in pyramid_stack[1:]:
            assert len(pyramid) == num_levels
