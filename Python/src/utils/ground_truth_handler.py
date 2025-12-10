import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))

class GroundTruthHandler:
    """
    Handles loading and processing of Ground Truth data for comparison with EVM measurements.
    
    This class loads ground truth heart rate data from files, calculates chunk averages,
    and provides methods for error calculation and statistical analysis.
    """
    
    def __init__(self, gt_path, frame_chunk_size):
        """
        Initializes the Ground Truth handler.
        
        Args:
            gt_path (str): Path to the ground truth file
            frame_chunk_size (int): Size of frame chunks for averaging
        """
        self.gt_path = gt_path
        self.frame_chunk_size = frame_chunk_size
        self.gt_data = None  # Raw ground truth data
        self.gt_hr = None    # Heart rate values (second row of data)
        self.gt_chunk_hrs = None  # Chunk-averaged heart rate values
        
    def load_ground_truth(self):
        """
        Loads and processes ground truth data from file.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            # Load ground truth data from text file
            self.gt_data = np.loadtxt(self.gt_path)
            
            # Second row contains heart rate values (assumed format)
            self.gt_hr = self.gt_data[1, :]
            
            # Calculate chunk averages
            self._calculate_chunk_averages()
            
            print(f"✅ Ground Truth loaded: {len(self.gt_hr)} points, {len(self.gt_chunk_hrs)} chunks")
            return True
        except Exception as e:
            print(f"❌ Error loading Ground Truth: {e}")
            return False
    
    def _calculate_chunk_averages(self):
        """Calculates average heart rate values for each frame chunk."""
        # Determine number of complete chunks
        num_chunks = len(self.gt_hr) // self.frame_chunk_size
        
        # Calculate mean heart rate for each chunk
        self.gt_chunk_hrs = [
            np.mean(self.gt_hr[i * self.frame_chunk_size:(i + 1) * self.frame_chunk_size]) 
            for i in range(num_chunks)
        ]
    
    def get_hr_for_chunk(self, chunk_index):
        """
        Retrieves ground truth heart rate value for a specific chunk.
        
        Args:
            chunk_index (int): Index of the chunk
            
        Returns:
            float: Ground truth heart rate value, or None if not available
        """
        if self.gt_chunk_hrs is None or chunk_index >= len(self.gt_chunk_hrs):
            return None
        return self.gt_chunk_hrs[chunk_index]
    
    def calculate_error(self, estimated_hr, chunk_index):
        """
        Calculates error between estimated HR and ground truth.
        
        Args:
            estimated_hr (float): Estimated heart rate value
            chunk_index (int): Chunk index for comparison
            
        Returns:
            tuple: (absolute_error, ground_truth_value) or (None, None) if unavailable
        """
        true_hr = self.get_hr_for_chunk(chunk_index)
        
        # Return None if either value is missing
        if true_hr is None or estimated_hr is None:
            return None, None
        
        # Calculate absolute error
        error = abs(estimated_hr - true_hr)
        return error, true_hr
    
    def get_available_chunks_count(self):
        """
        Returns the total number of available chunks in ground truth.
        
        Returns:
            int: Number of available chunks
        """
        return len(self.gt_chunk_hrs) if self.gt_chunk_hrs else 0
    
    def get_summary_stats(self):
        """
        Provides summary statistics of ground truth data.
        
        Returns:
            dict: Ground truth statistics including min, max, mean, std
        """
        if self.gt_hr is None:
            return {}
        
        return {
            'total_points': len(self.gt_hr),
            'total_chunks': len(self.gt_chunk_hrs) if self.gt_chunk_hrs else 0,
            'hr_min': np.min(self.gt_hr),
            'hr_max': np.max(self.gt_hr),
            'hr_mean': np.mean(self.gt_hr),
            'hr_std': np.std(self.gt_hr)
        }


# Utility function for quick setup
def setup_ground_truth(subject_num, dataset_path, buffer_size):
    """
    Quick setup for ground truth for a specific subject.
    
    Args:
        subject_num (int): Subject number
        dataset_path (str): Base dataset path
        buffer_size (int): Frame chunk size (used as buffer size)
        
    Returns:
        tuple: (GroundTruthHandler instance, video_path)
    """
    # Construct file paths based on subject number
    video_path = f"{dataset_path}/subject{subject_num}/vid.mp4"
    gt_path = f"{dataset_path}/subject{subject_num}/ground_truth.txt"
    
    # Initialize ground truth handler
    gt_handler = GroundTruthHandler(gt_path, buffer_size)
    success = gt_handler.load_ground_truth()
    
    if not success:
        print(f"⚠️  Could not load Ground Truth for subject {subject_num}")
    
    return gt_handler, video_path