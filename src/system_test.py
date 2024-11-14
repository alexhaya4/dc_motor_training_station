import time
import logging
import json
from pathlib import Path
from main_controller import MainController
from data_logger import DataLogger
from anfis_controller import ANFISController

class SystemTest:
    def __init__(self):
        # Setup logging
        self._setup_logging()
        
        # Create system components
        self.main_controller = MainController(simulation_mode=True)
        self.data_logger = DataLogger()
        
        self.logger.info("System Test initialized")

    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        self.logger = logging.getLogger('SystemTest')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler('logs/system_test.log')
        console_handler = logging.StreamHandler()
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def run_test_sequence(self):
        """Run complete test sequence"""
        try:
            print("\nStarting System Test Sequence")
            print("=" * 50)
            
            # Test 1: Manual Control
            self._test_manual_control()
            
            # Test 2: ANFIS Training
            self._test_anfis_training()
            
            # Test 3: ANFIS Control
            self._test_anfis_control()
            
            # Test 4: Emergency Stop
            self._test_emergency_stop()
            
            print("\nSystem Test Completed Successfully")
            
        except Exception as e:
            self.logger.error(f"Test sequence failed: {str(e)}")
            print("\nSystem Test Failed - Check Logs")
        finally:
            self.main_controller.cleanup()

    def _test_manual_control(self):
        """Test manual speed control"""
        print("\nTest 1: Manual Speed Control")
        print("-" * 50)
        
        # Start the system
        self.main_controller.start()
        self.main_controller.set_control_mode('manual')
        
        # Test different speeds
        test_speeds = [
            (0, "Initial state"),
            (500, "Low speed"),
            (1000, "Medium speed"),
            (1500, "High speed"),
            (750, "Speed reduction"),
            (0, "Stop")
        ]
        
        for speed, description in test_speeds:
            print(f"\n{description}:")
            self.main_controller.set_target_speed(speed)
            self.main_controller.hardware.set_duty_cycle(speed / 17.5)
            
            # Monitor for 2 seconds
            start_time = time.time()
            while time.time() - start_time < 2:
                measurements = self.main_controller.hardware.get_measurements()
                print(f"Speed: {measurements['speed']:6.1f} RPM | "
                      f"Current: {measurements['current']:4.2f} A | "
                      f"Duty Cycle: {measurements['duty_cycle']:5.1f}%    ", 
                      end='\r')
                time.sleep(0.05)
            print()  # New line after each test

    def _test_anfis_training(self):
        """Test ANFIS controller training"""
        print("\nTest 2: ANFIS Controller Training")
        print("-" * 50)
        
        # Generate training data
        print("Generating training data...")
        training_data = self._generate_training_data()
        
        # Train ANFIS controller
        print("Training ANFIS controller...")
        self.main_controller.anfis.train(training_data)
        
        print("ANFIS training completed")

    def _test_anfis_control(self):
        """Test ANFIS speed control"""
        print("\nTest 3: ANFIS Speed Control")
        print("-" * 50)
        
        self.main_controller.set_control_mode('anfis')
        
        # Test speed tracking
        test_speeds = [500, 1000, 1500, 750, 0]
        
        for speed in test_speeds:
            print(f"\nTracking {speed} RPM:")
            self.main_controller.set_target_speed(speed)
            
            # Monitor for 3 seconds
            start_time = time.time()
            while time.time() - start_time < 3:
                measurements = self.main_controller.hardware.get_measurements()
                print(f"Speed: {measurements['speed']:6.1f} RPM | "
                      f"Error: {speed - measurements['speed']:6.1f} RPM | "
                      f"Duty Cycle: {measurements['duty_cycle']:5.1f}%    ",
                      end='\r')
                time.sleep(0.05)
            print()

    def _test_emergency_stop(self):
        """Test emergency stop functionality"""
        print("\nTest 4: Emergency Stop")
        print("-" * 50)
        
        # Set motor to medium speed
        print("Setting motor to medium speed...")
        self.main_controller.set_target_speed(1000)
        time.sleep(2)
        
        # Activate emergency stop
        print("Activating emergency stop...")
        self.main_controller.emergency_stop()
        
        # Verify motor stopped
        measurements = self.main_controller.hardware.get_measurements()
        print(f"Final speed: {measurements['speed']:.1f} RPM")
        print(f"Final duty cycle: {measurements['duty_cycle']:.1f}%")

    def _generate_training_data(self):
        """Generate training data for ANFIS"""
        training_data = []
        
        # Run system through various operating points
        test_speeds = [0, 500, 1000, 1500, 750, 0]
        
        for speed in test_speeds:
            self.main_controller.set_target_speed(speed)
            start_time = time.time()
            
            while time.time() - start_time < 2:
                measurements = self.main_controller.hardware.get_measurements()
                error = speed - measurements['speed']
                
                if hasattr(self, 'last_error'):
                    delta_error = (error - self.last_error) / 0.05
                else:
                    delta_error = 0
                    
                self.last_error = error
                
                # Use current control as target output
                target = measurements['duty_cycle']
                
                training_data.append((error, delta_error, target))
                time.sleep(0.05)
        
        return training_data

if __name__ == "__main__":
    # Run system test
    test = SystemTest()
    test.run_test_sequence()