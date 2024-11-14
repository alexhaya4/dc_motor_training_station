import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime
import numpy as np

class DataLogger:
    def __init__(self, base_path: str = "data"):
        """
        Initialize the Data Logger
        Args:
            base_path: Base directory for storing data files
        """
        # Setup paths
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.logs_path = self.base_path / "logs"
        self.training_data_path = self.base_path / "training_data"
        self.configs_path = self.base_path / "configs"
        
        for path in [self.logs_path, self.training_data_path, self.configs_path]:
            path.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize data buffers
        self.current_session_data = []
        self.buffer_size = 1000  # Number of samples to hold in memory
        
        self.logger.info("Data Logger initialized")
        
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('DataLogger')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler(self.logs_path / 'data_logger.log')
        console_handler = logging.StreamHandler()
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_data(self, data: Dict[str, float]):
        """
        Log a single data point
        Args:
            data: Dictionary containing measurements
        """
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = time.time()
        
        # Add to current session
        self.current_session_data.append(data)
        
        # If buffer is full, save to file
        if len(self.current_session_data) >= self.buffer_size:
            self.save_session_data()
    
    def save_session_data(self):
        """Save current session data to file"""
        if not self.current_session_data:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.training_data_path / f"session_data_{timestamp}.csv"
        
        try:
            # Get all keys from the first data point
            fieldnames = list(self.current_session_data[0].keys())
            
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.current_session_data)
            
            self.logger.info(f"Session data saved to {filename}")
            self.current_session_data = []  # Clear buffer
            
        except Exception as e:
            self.logger.error(f"Error saving session data: {str(e)}")
    
    def load_training_data(self, filename: str) -> List[Dict[str, float]]:
        """
        Load training data from file
        Args:
            filename: Name of the file to load
        Returns:
            List of data points
        """
        filepath = self.training_data_path / filename
        data = []
        
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert string values to float
                    converted_row = {k: float(v) for k, v in row.items()}
                    data.append(converted_row)
            
            self.logger.info(f"Loaded {len(data)} data points from {filename}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading training data: {str(e)}")
            return []
    
    def generate_training_dataset(self, data: List[Dict[str, float]], 
                                input_keys: List[str], 
                                output_keys: List[str]) -> tuple:
        """
        Generate training dataset from raw data
        Args:
            data: List of data points
            input_keys: List of keys to use as inputs
            output_keys: List of keys to use as outputs
        Returns:
            Tuple of (inputs, outputs) as numpy arrays
        """
        try:
            inputs = []
            outputs = []
            
            for point in data:
                input_values = [point[key] for key in input_keys]
                output_values = [point[key] for key in output_keys]
                
                inputs.append(input_values)
                outputs.append(output_values)
            
            return np.array(inputs), np.array(outputs)
            
        except Exception as e:
            self.logger.error(f"Error generating training dataset: {str(e)}")
            return np.array([]), np.array([])
    
    def save_config(self, config: Dict[str, Any], name: str):
        """
        Save configuration to file
        Args:
            config: Configuration dictionary
            name: Name for the configuration file
        """
        filename = self.configs_path / f"{name}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=4)
            self.logger.info(f"Configuration saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
    
    def load_config(self, name: str) -> Dict[str, Any]:
        """
        Load configuration from file
        Args:
            name: Name of the configuration file
        Returns:
            Configuration dictionary
        """
        filename = self.configs_path / f"{name}.json"
        
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Configuration loaded from {filename}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            return {}

    def calculate_statistics(self, data: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each measurement
        Args:
            data: List of data points
        Returns:
            Dictionary of statistics for each measurement
        """
        if not data:
            return {}
        
        # Get all keys except timestamp
        keys = [k for k in data[0].keys() if k != 'timestamp']
        stats = {}
        
        for key in keys:
            values = [point[key] for point in data]
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return stats

# Test code
if __name__ == "__main__":
    # Create data logger instance
    logger = DataLogger()
    
    print("\nData Logger Test Sequence")
    print("-" * 50)
    
    # Generate some test data
    print("Generating test data...")
    
    test_data = []
    for i in range(100):
        data_point = {
            'timestamp': time.time(),
            'speed': np.random.normal(1000, 50),
            'current': np.random.normal(2.5, 0.2),
            'duty_cycle': 50 + np.random.normal(0, 2)
        }
        test_data.append(data_point)
        logger.log_data(data_point)
        time.sleep(0.01)
    
    # Save session data
    print("Saving session data...")
    logger.save_session_data()
    
    # Calculate and display statistics
    print("\nData Statistics:")
    stats = logger.calculate_statistics(test_data)
    for measurement, values in stats.items():
        print(f"\n{measurement}:")
        for stat_name, stat_value in values.items():
            print(f"  {stat_name}: {stat_value:.2f}")
    
    print("\nTest completed")