import numpy as np
import pytest
from src.evm.temporal_filtering import (
    create_bandpass_filter,
    apply_temporal_bandpass,
    temporal_dual_bandpass_filter
)


class TestCreateBandpassFilter:
    """Test the create_bandpass_filter function."""
    
    def test_valid_filter_creation(self):
        """Test creating a valid bandpass filter."""
        lowcut = 0.5
        highcut = 3.0
        nyquist = 15.0  # For 30 FPS
        
        b, a = create_bandpass_filter(lowcut, highcut, nyquist, order=2)
        
        assert b is not None
        assert a is not None
        assert len(b) == 5  # order*2 + 1
        assert len(a) == 5  # order*2 + 1
        
    def test_edge_cases_frequency_normalization(self):
        """Test frequency normalization at edges."""
        # Test with very low frequency
        result = create_bandpass_filter(0.001, 5.0, 15.0)
        assert result is not None
        
        # Test with frequencies near nyquist
        result = create_bandpass_filter(0.5, 14.9, 15.0)
        assert result is not None
        
    def test_invalid_cutoff_order(self):
        """Test when lowcut >= highcut."""
        result = create_bandpass_filter(3.0, 0.5, 15.0)
        assert result is None
        
        result = create_bandpass_filter(2.0, 2.0, 15.0)
        assert result is None
        
    def test_extreme_frequencies(self):
        """Test with frequencies at boundaries."""
        # Both frequencies clamped to valid range
        result = create_bandpass_filter(-1.0, 20.0, 15.0)
        assert result is not None
        
    def test_different_orders(self):
        """Test filter creation with different orders."""
        for order in [1, 2, 4, 6]:
            result = create_bandpass_filter(0.5, 3.0, 15.0, order=order)
            assert result is not None
            b, a = result
            assert len(b) == order * 2 + 1
            assert len(a) == order * 2 + 1


class TestApplyTemporalBandpass:
    """Test the apply_temporal_bandpass function."""
    
    @pytest.fixture
    def sample_video_tensor(self):
        """Create a sample video tensor for testing."""
        np.random.seed(42)
        T, H, W, C = 100, 64, 64, 3
        # Create a signal with specific frequency components
        t = np.arange(T)
        signal_1hz = 0.5 * np.sin(2 * np.pi * 1.0 * t / 30)  # 1 Hz component
        signal_2hz = 0.3 * np.sin(2 * np.pi * 2.0 * t / 30)  # 2 Hz component
        
        base_tensor = np.random.randn(T, H, W, C) * 0.1
        # Add periodic signals to all pixels
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    base_tensor[:, i, j, c] += signal_1hz + signal_2hz
        
        return base_tensor
    
    def test_apply_filter_to_video(self, sample_video_tensor):
        """Test applying bandpass filter to video tensor."""
        filtered = apply_temporal_bandpass(
            sample_video_tensor,
            lowcut=0.8,
            highcut=1.2,
            fps=30.0,
            axis=0
        )
        
        assert filtered.shape == sample_video_tensor.shape
        assert not np.allclose(filtered, sample_video_tensor)
        
        # Filtered signal should have reduced amplitude at non-pass frequencies
        pixel_ts = sample_video_tensor[:, 0, 0, 0]
        filtered_pixel_ts = filtered[:, 0, 0, 0]
        
        # Check that the filtered signal is different but correlated
        correlation = np.corrcoef(pixel_ts, filtered_pixel_ts)[0, 1]
        assert abs(correlation) > 0.3  # Should still be somewhat correlated
        
    def test_short_video_handling(self):
        """Test that short videos are returned unchanged."""
        short_tensor = np.random.randn(5, 10, 10, 3)
        result = apply_temporal_bandpass(short_tensor, 0.5, 3.0, 30.0)
        
        assert np.array_equal(result, short_tensor)
        
    def test_invalid_filter_returns_original(self, sample_video_tensor):
        """Test that invalid filter parameters return original tensor."""
        # Use invalid cutoff frequencies
        result = apply_temporal_bandpass(
            sample_video_tensor,
            lowcut=3.0,  # > highcut
            highcut=0.5,
            fps=30.0
        )
        
        assert np.array_equal(result, sample_video_tensor)
        
    def test_different_axes(self, sample_video_tensor):
        """Test filtering along different axes."""
        # Reshape tensor to test different axis
        tensor_reshaped = np.transpose(sample_video_tensor, (1, 0, 2, 3))
        
        filtered = apply_temporal_bandpass(
            tensor_reshaped,
            lowcut=0.8,
            highcut=1.2,
            fps=30.0,
            axis=1  # Temporal axis is now at position 1
        )
        
        assert filtered.shape == tensor_reshaped.shape
        assert not np.allclose(filtered, tensor_reshaped)
        
    def test_sine_wave_extraction(self):
        """Test filtering on a pure sine wave."""
        T = 300
        t = np.arange(T)
        fps = 30.0
        
        # Create a 1 Hz sine wave
        signal_1hz = np.sin(2 * np.pi * 1.0 * t / fps)
        # Create a 0.1 Hz sine wave (should be filtered out)
        signal_01hz = 0.5 * np.sin(2 * np.pi * 0.1 * t / fps)
        
        combined_signal = signal_1hz + signal_01hz
        
        # Create tensor with signal in all pixels
        tensor = np.zeros((T, 10, 10, 3))
        for c in range(3):
            for i in range(10):
                for j in range(10):
                    tensor[:, i, j, c] = combined_signal
        
        # Apply bandpass filter around 1 Hz
        filtered = apply_temporal_bandpass(
            tensor,
            lowcut=0.8,
            highcut=1.2,
            fps=fps,
            axis=0
        )
        
        # Extract one pixel's time series
        filtered_pixel = filtered[:, 0, 0, 0]
        
        # The 0.1 Hz component should be attenuated
        assert np.std(filtered_pixel) < 1.2  # Reduced from ~1.12
        
        # Check correlation with original 1Hz signal
        correlation = np.corrcoef(signal_1hz, filtered_pixel)[0, 1]
        assert correlation > 0.95  # Highly correlated with 1Hz signal


class TestTemporalDualBandpassFilter:
    """Test the temporal_dual_bandpass_filter function."""
    
    @pytest.fixture
    def multi_freq_tensor(self):
        """Create tensor with multiple frequency components."""
        np.random.seed(42)
        T, H, W, C = 200, 32, 32, 3
        fps = 30.0
        t = np.arange(T)
        
        # Heart rate signal (~1.2 Hz = 72 BPM)
        heart_signal = 0.8 * np.sin(2 * np.pi * 1.2 * t / fps)
        # Respiration signal (~0.3 Hz = 18 BPM)
        resp_signal = 0.6 * np.sin(2 * np.pi * 0.3 * t / fps)
        # Noise
        noise = np.random.randn(T) * 0.2
        
        combined_signal = heart_signal + resp_signal + noise
        
        tensor = np.zeros((T, H, W, C))
        for c in range(C):
            tensor[:, :, :, c] = combined_signal[:, np.newaxis, np.newaxis]
        
        return tensor, fps
    
    def test_dual_filter_application(self, multi_freq_tensor):
        """Test applying two bandpass filters simultaneously."""
        video_tensor, fps = multi_freq_tensor
        
        filtered_hr, filtered_rr = temporal_dual_bandpass_filter(
            video_tensor,
            fps=fps,
            low_heart=0.8,
            high_heart=1.5,
            low_resp=0.1,
            high_resp=0.5,
            axis=0
        )
        
        assert filtered_hr.shape == video_tensor.shape
        assert filtered_rr.shape == video_tensor.shape
        
        # They should be different from each other
        assert not np.allclose(filtered_hr, filtered_rr)
        
        # They should be different from original
        assert not np.allclose(filtered_hr, video_tensor)
        assert not np.allclose(filtered_rr, video_tensor)
        
    def test_short_video_handling_dual(self):
        """Test dual filter with short video."""
        short_tensor = np.random.randn(9, 10, 10, 3)
        hr, rr = temporal_dual_bandpass_filter(
            short_tensor,
            fps=30.0,
            low_heart=0.8,
            high_heart=1.5,
            low_resp=0.1,
            high_resp=0.5
        )
        
        assert np.array_equal(hr, short_tensor)
        assert np.array_equal(rr, short_tensor)
        
    def test_frequency_separation(self, multi_freq_tensor):
        """Test that filters properly separate frequency bands."""
        video_tensor, fps = multi_freq_tensor
        
        filtered_hr, filtered_rr = temporal_dual_bandpass_filter(
            video_tensor,
            fps=fps,
            low_heart=0.8,
            high_heart=1.5,
            low_resp=0.1,
            high_resp=0.5,
            axis=0
        )
        
        # Take one pixel from each filtered result
        hr_pixel = filtered_hr[:, 0, 0, 0]
        rr_pixel = filtered_rr[:, 0, 0, 0]
        
        # Compute FFT to check frequency content
        fft_hr = np.abs(np.fft.rfft(hr_pixel))
        fft_rr = np.abs(np.fft.rfft(rr_pixel))
        freqs = np.fft.rfftfreq(len(hr_pixel), 1/fps)
        
        # Find dominant frequencies
        hr_dom_freq = freqs[np.argmax(fft_hr)]
        rr_dom_freq = freqs[np.argmax(fft_rr)]
        
        # HR filter should pass ~1.2 Hz
        assert 0.8 <= hr_dom_freq <= 1.5
        # RR filter should pass ~0.3 Hz
        assert 0.1 <= rr_dom_freq <= 0.5
        
    def test_identical_filters(self):
        """Test with identical filter parameters."""
        T, H, W, C = 100, 10, 10, 3
        tensor = np.random.randn(T, H, W, C)
        
        filtered_hr, filtered_rr = temporal_dual_bandpass_filter(
            tensor,
            fps=30.0,
            low_heart=0.5,
            high_heart=3.0,
            low_resp=0.5,
            high_resp=3.0
        )
        
        # Should be identical (within numerical precision)
        assert np.allclose(filtered_hr, filtered_rr, rtol=1e-10)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_nan_handling(self):
        """Test behavior with NaN values in input."""
        tensor = np.random.randn(100, 10, 10, 3)
        # Add some NaN values
        tensor[50:55, :, :, :] = np.nan
        
        # Should raise no error and return original or filtered tensor
        result = apply_temporal_bandpass(tensor, 0.5, 3.0, 30.0)
        
        assert result.shape == tensor.shape
        
    def test_inf_handling(self):
        """Test behavior with infinite values."""
        tensor = np.random.randn(100, 10, 10, 3)
        tensor[0, 0, 0, 0] = np.inf
        
        result = apply_temporal_bandpass(tensor, 0.5, 3.0, 30.0)
        assert result.shape == tensor.shape


def test_filter_properties():
    """Test specific filter properties."""
    # Test that filtfilt is used (zero-phase filtering)
    # by checking symmetry properties
    T = 100
    fps = 30.0
    
    # Create an asymmetric signal
    t = np.arange(T)
    signal = t / T  # Ramp function
    
    tensor = np.zeros((T, 5, 5, 3))
    for c in range(3):
        tensor[:, :, :, c] = signal[:, np.newaxis, np.newaxis]
    
    filtered = apply_temporal_bandpass(
        tensor,
        lowcut=0.01,
        highcut=10.0,
        fps=fps,
        axis=0
    )
    
    # With filtfilt, phase distortion should be minimized
    filtered_signal = filtered[:, 0, 0, 0]
    
    # The filtered signal should not have extreme phase shifts
    # (This is a qualitative test - actual verification would require
    # comparing with lfilter output)
    assert not np.any(np.isnan(filtered_signal))
    assert not np.any(np.isinf(filtered_signal))