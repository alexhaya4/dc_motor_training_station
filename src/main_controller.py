import time
import logging
from pathlib import Path
import threading
from typing import Dict, Any
from hardware_controller import HardwareController
from data_logger import DataLogger
from anfis_controller import ANFISController  # We'll implement this next

class MainController:
    def __init__(self, simulation_mode: bool = True):
        """
        Initialize the main control system
        Args:
            simulation_mode: If True, runs in simulation without actual hardware
        """
        # Setup logging
        self._setup_logging()
        
        # Initialize subsystems
        self.hardware = HardwareController(simulation_mode=simulation_mode)
        self.data_logger = DataLogger()
        self.anfis = ANFISController()
        
        # System state
        self.running = False
        self.control_mode = 'manual'  # 'manual' or 'anfis'
        self.logging_enabled = True
        
        # Control parameters
        self.sampling_rate = 100  # Hz
        self.control_interval = 1.0 / self.sampling_rate
        
        self.logger.info("Main Controller initialized")

    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        self.logger = logging.getLogger('MainController')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler('logs/main_controller.log')
        console_handler = logging.StreamHandler()
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def start(self):
        """Start the control system"""
        try:
            self.running = True
            self.hardware.start()
            
            # Start control loop in separate thread
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()
            
            self.logger.info("Control system started")
            
        except Exception as e:
            self.logger.error(f"Error starting system: {str(e)}")
            self.stop()

    def stop(self):
        """Stop the control system"""
        self.running = False
        self.hardware.stop()
        self.logger.info("Control system stopped")

    def _control_loop(self):
        """Main control loop"""
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            
            try:
                # Get current measurements
                measurements = self.hardware.get_measurements()
                
                # Log data if enabled
                if self.logging_enabled:
                    self.data_logger.log_data(measurements)
                
                # Execute control based on mode
                if self.control_mode == 'anfis':
                    self._anfis_control(measurements)
                else:
                    self._manual_control(measurements)
                
                # Maintain consistent sampling rate
                sleep_time = self.control_interval - (time.time() - current_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                last_time = current_time
                
            except Exception as e:
                self.logger.error(f"Error in control loop: {str(e)}")
                self.stop()
                break

    def _manual_control(self, measurements: Dict[str, float]):
        """Execute manual control"""
        # Manual control just maintains the current duty cycle
        pass

    def _anfis_control(self, measurements: Dict[str, float]):
        """Execute ANFIS control"""
        try:
            # Calculate error and change in error
            error = self.target_speed - measurements['speed']
            
            # Get control action from ANFIS
            control_action = self.anfis.calculate_control(error, measurements)
            
            # Apply control action
            self.hardware.set_duty_cycle(control_action)
            
        except Exception as e:
            self.logger.error(f"Error in ANFIS control: {str(e)}")

    def set_target_speed(self, speed: float):
        """Set the target speed for the system"""
        try:
            self.target_speed = speed
            self.logger.info(f"Target speed set to {speed} RPM")
            
        except Exception as e:
            self.logger.error(f"Error setting target speed: {str(e)}")

    def set_control_mode(self, mode: str):
        """Set the control mode ('manual' or 'anfis')"""
        if mode in ['manual', 'anfis']:
            self.control_mode = mode
            self.logger.info(f"Control mode set to {mode}")
        else:
            self.logger.error(f"Invalid control mode: {mode}")

    def emergency_stop(self):
        """Execute emergency stop"""
        self.stop()
        self.hardware.emergency_stop()
        self.logger.warning("Emergency stop activated")

    def cleanup(self):
        """Cleanup system resources"""
        self.stop()
        self.hardware.cleanup()
        self.data_logger.save_session_data()
        self.logger.info("System cleanup completed")

# Test code
if __name__ == "__main__":
    # Create main controller in simulation mode
    controller = MainController(simulation_mode=True)
    
    print("\nMain Controller Test Sequence")
    print("-" * 50)
    
    try:
        # Start the system
        controller.start()
        print("System started")
        
        # Test manual control
        print("\nTesting manual control...")
        controller.set_control_mode('manual')
        
        test_speeds = [0, 500, 1000, 1500, 750, 0]
        for speed in test_speeds:
            print(f"\nSetting target speed to {speed} RPM")
            controller.set_target_speed(speed)
            controller.hardware.set_duty_cycle(speed / 17.5)
            
            # Monitor for 2 seconds
            start_time = time.time()
            while time.time() - start_time < 2:
                measurements = controller.hardware.get_measurements()
                print(f"Speed: {measurements['speed']:6.1f} RPM | "
                      f"Current: {measurements['current']:4.2f} A | "
                      f"Duty Cycle: {measurements['duty_cycle']:5.1f}%    ", 
                      end='\r')
                time.sleep(0.05)
            print()  # New line after each test
        
        # Test ANFIS control (once implemented)
        print("\nANFIS control testing will be implemented in the next phase")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        controller.cleanup()
        print("\nTest completed")