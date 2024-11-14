import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time

class ANFISController:
    def __init__(self, config_path: str = "config/anfis_config.json"):
        """
        Initialize ANFIS Controller
        Args:
            config_path: Path to ANFIS configuration file
        """
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize ANFIS parameters
        self._initialize_parameters()
        
        # Training state
        self.is_trained = False
        self.training_epoch = 0
        
        self.logger.info("ANFIS Controller initialized")

    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        self.logger = logging.getLogger('ANFISController')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler('logs/anfis_controller.log')
        console_handler = logging.StreamHandler()
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load ANFIS configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise

    def _initialize_parameters(self):
        """Initialize ANFIS parameters from configuration"""
        try:
            # Get structure parameters
            structure = self.config['anfis_structure']
            self.num_inputs = structure['input_vars']
            self.num_mf = structure['membership_functions']
            self.num_rules = self.num_mf ** self.num_inputs
            
            # Initialize membership function parameters
            self.mf_params = {
                'error': self.config['membership_functions']['error'],
                'delta_error': self.config['membership_functions']['delta_error']
            }
            
            # Initialize rule consequent parameters (initially random)
            self.consequent_params = np.random.randn(self.num_rules, self.num_inputs + 1)
            
            # Get training parameters
            train_params = self.config['training_params']
            self.learning_rate = train_params['learning_rate']
            self.momentum = train_params['momentum']
            self.error_goal = train_params['error_goal']
            
            self.logger.info("ANFIS parameters initialized")
            
        except KeyError as e:
            self.logger.error(f"Missing configuration parameter: {str(e)}")
            raise

    def membership_function(self, x: float, params: List[float], mf_type: str = 'trimf') -> float:
        """
        Calculate membership grade
        Args:
            x: Input value
            params: MF parameters
            mf_type: Type of membership function
        Returns:
            Membership grade
        """
        if mf_type == 'trimf':
            a, b, c = params
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a)
            else:  # b < x < c
                return (c - x) / (c - b)
        else:
            raise ValueError(f"Unsupported membership function type: {mf_type}")

    def calculate_rule_firing_strengths(self, error: float, delta_error: float) -> np.ndarray:
        """
        Calculate rule firing strengths using fuzzy inference
        Args:
            error: Current error
            delta_error: Change in error
        Returns:
            Array of rule firing strengths
        """
        # Calculate membership grades for error
        error_mf_grades = []
        for mf_range in self.mf_params['error']['ranges']:
            grade = self.membership_function(error, mf_range)
            error_mf_grades.append(grade)
        
        # Calculate membership grades for delta_error
        delta_error_mf_grades = []
        for mf_range in self.mf_params['delta_error']['ranges']:
            grade = self.membership_function(delta_error, mf_range)
            delta_error_mf_grades.append(grade)
        
        # Calculate rule firing strengths (using min t-norm)
        firing_strengths = np.zeros(self.num_rules)
        rule_idx = 0
        for i in range(self.num_mf):
            for j in range(self.num_mf):
                firing_strengths[rule_idx] = min(error_mf_grades[i], 
                                              delta_error_mf_grades[j])
                rule_idx += 1
        
        return firing_strengths

    def calculate_control(self, error: float, measurements: Dict[str, float]) -> float:
        """
        Calculate control output using ANFIS
        Args:
            error: Current error
            measurements: Current system measurements
        Returns:
            Control output (duty cycle)
        """
        try:
            # Calculate change in error (delta_error)
            if not hasattr(self, 'last_error'):
                self.last_error = error
            
            delta_error = (error - self.last_error) / 0.01  # Assuming 0.01s sample time
            self.last_error = error
            
            # Calculate rule firing strengths
            firing_strengths = self.calculate_rule_firing_strengths(error, delta_error)
            
            # Normalize firing strengths
            sum_firing = np.sum(firing_strengths) + 1e-10  # Avoid division by zero
            normalized_firing = firing_strengths / sum_firing
            
            # Calculate consequent outputs
            inputs = np.array([1.0, error, delta_error])  # Include bias term
            rule_outputs = np.dot(self.consequent_params, inputs)
            
            # Calculate weighted sum output
            control = np.sum(normalized_firing * rule_outputs)
            
            # Limit control output
            control = np.clip(control, -100, 100)
            
            return float(control)
            
        except Exception as e:
            self.logger.error(f"Error calculating control: {str(e)}")
            return 0.0

    def train(self, training_data: List[Tuple[float, float, float]]):
        """
        Train the ANFIS controller
        Args:
            training_data: List of (error, delta_error, target_output) tuples
        """
        try:
            num_epochs = self.config['anfis_structure']['training_epochs']
            
            for epoch in range(num_epochs):
                epoch_error = 0.0
                
                for error, delta_error, target in training_data:
                    # Forward pass
                    output = self.calculate_control(error, {'speed': 0})  # Dummy measurements
                    
                    # Calculate error
                    error = target - output
                    epoch_error += error ** 2
                    
                    # Update parameters (simplified for now)
                    # In practice, would use proper backpropagation
                    
                self.training_epoch += 1
                epoch_error = epoch_error / len(training_data)
                
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}, Error: {epoch_error:.6f}")
                
                if epoch_error < self.error_goal:
                    self.logger.info("Training completed - Error goal reached")
                    break
            
            self.is_trained = True
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")

# Test code
if __name__ == "__main__":
    # Create ANFIS controller
    controller = ANFISController()
    
    print("\nANFIS Controller Test Sequence")
    print("-" * 50)
    
    # Test basic functionality
    test_cases = [
        (0.0, 0.0),     # No error
        (50.0, 0.0),    # Positive error
        (-50.0, 0.0),   # Negative error
        (25.0, 5.0),    # Error with positive rate
        (-25.0, -5.0)   # Error with negative rate
    ]
    
    print("\nTesting control responses:")
    print(f"{'Error':>10} {'Delta Error':>12} {'Control':>10}")
    print("-" * 35)
    
    for error, delta_error in test_cases:
        control = controller.calculate_control(error, {'speed': 0})
        print(f"{error:10.1f} {delta_error:12.1f} {control:10.1f}")
    
    # Generate some training data
    print("\nGenerating training data...")
    training_data = []
    for _ in range(100):
        error = np.random.uniform(-100, 100)
        delta_error = np.random.uniform(-10, 10)
        target = np.clip(-0.5 * error - 2 * delta_error, -100, 100)  # Simple control law
        training_data.append((error, delta_error, target))
    
    # Train the controller
    print("\nTraining ANFIS controller...")
    controller.train(training_data)
    
    print("\nTest completed")