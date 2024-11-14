import time
import threading
import json
import logging
from typing import Dict, Any
from pathlib import Path
import numpy as np

# GPIO Import handling
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("RPi.GPIO not available. Running in simulation mode only.")

class HardwareController:
    def __init__(self, config_path: str = "config/hardware_config.json", simulation_mode: bool = True):
        """
        Initialize Hardware Controller
        Args:
            config_path: Path to hardware configuration file
            simulation_mode: If True, runs in simulation without actual hardware
        """
        # Force simulation mode if GPIO is not available
        if not GPIO_AVAILABLE:
            simulation_mode = True
            
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Operating mode
        self.simulation_mode = simulation_mode
        
        # State variables
        self.running = False
        self.emergency_stop_active = False
        self.current_speed = 0.0
        self.current_current = 0.0
        self.target_speed = 0.0
        self.duty_cycle = 0.0
        
        # Initialize hardware or simulation
        if not simulation_mode and GPIO_AVAILABLE:
            self._initialize_hardware()
        else:
            self.logger.info("Running in simulation mode")
            self._initialize_simulation()
        
        # Start monitoring thread
        self._start_monitoring()

    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        self.logger = logging.getLogger('HardwareController')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler('logs/hardware_controller.log')
        console_handler = logging.StreamHandler()
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load hardware configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise

    def _initialize_hardware(self):
        """Initialize GPIO and hardware components"""
        try:
            # Get GPIO pins from config
            self.pins = self.config['gpio_pins']
            
            # Setup GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup Motor Control Pins
            GPIO.setup(self.pins['motor_pwm'], GPIO.OUT)
            GPIO.setup(self.pins['motor_in1'], GPIO.OUT)
            GPIO.setup(self.pins['motor_in2'], GPIO.OUT)
            
            # Setup PWM
            self.pwm = GPIO.PWM(self.pins['motor_pwm'], self.config['motor']['pwm_frequency'])
            self.pwm.start(0)
            
            # Setup Speed Sensor
            GPIO.setup(self.pins['ir_sensor1'], GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(self.pins['ir_sensor1'], GPIO.FALLING, 
                                callback=self._speed_sensor_callback)
            
            # Setup Emergency Stop
            GPIO.setup(self.pins['emergency_stop'], GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(self.pins['emergency_stop'], GPIO.FALLING, 
                                callback=self._emergency_stop_callback)
            
            self.logger.info("Hardware initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing hardware: {str(e)}")
            raise

    def _initialize_simulation(self):
        """Initialize simulation parameters"""
        self.sim_params = {
            'motor_inertia': 0.01,          # Motor moment of inertia
            'motor_damping': 0.1,           # Damping coefficient
            'motor_torque_constant': 0.01,   # Motor torque constant
            'load_torque': 0.1,             # Base load torque
            'time_constant': 0.1,           # Motor electrical time constant
            'noise_level': 0.1,             # Speed measurement noise
            'current_constant': 0.001,       # Current per unit speed
            'base_current': 0.2,            # No-load current
            'current_noise': 0.02,          # Current measurement noise
            'max_speed': 1750               # Maximum speed in RPM
        }
        self.last_sim_time = time.time()
        self.current_speed = 0.0
        self.current_current = self.sim_params['base_current']

    def _start_monitoring(self):
        """Start the monitoring thread"""
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            if self.running and not self.emergency_stop_active:
                if self.simulation_mode:
                    self._update_simulation()
                else:
                    self._update_measurements()
            time.sleep(1.0 / self.config['sampling']['rate_hz'])

    def _update_simulation(self):
        """Update simulation state"""
        current_time = time.time()
        dt = current_time - self.last_sim_time
        self.last_sim_time = current_time
        
        # Motor dynamics
        speed_error = self.target_speed - self.current_speed
        motor_input = self.duty_cycle / 100.0  # Convert duty cycle to 0-1 range
        
        # Calculate acceleration with better scaling
        acceleration = (
            motor_input * self.sim_params['motor_torque_constant'] * self.sim_params['max_speed'] - 
            self.sim_params['motor_damping'] * self.current_speed - 
            self.sim_params['load_torque']
        ) / self.sim_params['motor_inertia']
        
        # Update speed with limits
        self.current_speed += acceleration * dt
        self.current_speed = np.clip(self.current_speed, 0, self.sim_params['max_speed'])
        
        # Add minimal noise to speed
        self.current_speed += np.random.normal(0, self.sim_params['noise_level'])
        self.current_speed = max(0, self.current_speed)  # Ensure non-negative speed
        
        # Calculate current with realistic scaling
        self.current_current = (
            self.sim_params['base_current'] +                    # No-load current
            abs(motor_input) * 2.0 +                            # Current due to duty cycle
            (self.current_speed/self.sim_params['max_speed']) * 3.0  # Current due to speed/load
        )
        
        # Add minimal noise to current
        self.current_current += np.random.normal(0, self.sim_params['current_noise'])
        self.current_current = max(0, self.current_current)  # Ensure non-negative current

    def _update_measurements(self):
        """Update real hardware measurements"""
        # This would be implemented for real hardware
        pass

    def _speed_sensor_callback(self, channel):
        """Handle speed sensor interrupts"""
        if not self.simulation_mode and GPIO_AVAILABLE:
            # Implement actual speed calculation from sensor
            pass

    def _emergency_stop_callback(self, channel):
        """Handle emergency stop button press"""
        if GPIO_AVAILABLE and not GPIO.input(self.pins['emergency_stop']):
            self.emergency_stop()

    def set_target_speed(self, speed: float):
        """Set the target speed for the motor"""
        try:
            # Limit speed to configured range
            speed = np.clip(speed, 
                          self.config['motor']['min_speed_rpm'],
                          self.config['motor']['max_speed_rpm'])
            
            self.target_speed = speed
            self.logger.debug(f"Target speed set to {speed} RPM")
            
        except Exception as e:
            self.logger.error(f"Error setting target speed: {str(e)}")
            raise

    def set_duty_cycle(self, duty_cycle: float):
        """Set the PWM duty cycle"""
        try:
            # Limit duty cycle to [-100, 100]
            duty_cycle = np.clip(duty_cycle, -100, 100)
            
            if self.simulation_mode:
                self.duty_cycle = duty_cycle
            elif GPIO_AVAILABLE:
                # Set motor direction
                if duty_cycle >= 0:
                    GPIO.output(self.pins['motor_in1'], GPIO.HIGH)
                    GPIO.output(self.pins['motor_in2'], GPIO.LOW)
                else:
                    GPIO.output(self.pins['motor_in1'], GPIO.LOW)
                    GPIO.output(self.pins['motor_in2'], GPIO.HIGH)
                
                # Set PWM
                self.pwm.ChangeDutyCycle(abs(duty_cycle))
                
            self.duty_cycle = duty_cycle
            self.logger.debug(f"Duty cycle set to {duty_cycle}%")
            
        except Exception as e:
            self.logger.error(f"Error setting duty cycle: {str(e)}")
            raise

    def get_measurements(self) -> Dict[str, float]:
        """Get current measurements"""
        return {
            'speed': self.current_speed,
            'current': self.current_current,
            'duty_cycle': self.duty_cycle,
            'target_speed': self.target_speed,
            'timestamp': time.time()
        }

    def start(self):
        """Start motor operation"""
        self.running = True
        self.emergency_stop_active = False
        self.logger.info("Motor operation started")

    def stop(self):
        """Stop motor operation"""
        self.running = False
        self.set_duty_cycle(0)
        self.logger.info("Motor operation stopped")

    def emergency_stop(self):
        """Activate emergency stop"""
        self.emergency_stop_active = True
        self.running = False
        self.set_duty_cycle(0)
        self.logger.warning("Emergency stop activated")

    def reset_emergency_stop(self):
        """Reset emergency stop state"""
        if self.simulation_mode or (GPIO_AVAILABLE and GPIO.input(self.pins['emergency_stop'])):
            self.emergency_stop_active = False
            self.logger.info("Emergency stop reset")
            return True
        return False

    def cleanup(self):
        """Cleanup GPIO and other resources"""
        self.stop()
        if not self.simulation_mode and GPIO_AVAILABLE:
            GPIO.cleanup()
        self.logger.info("Hardware cleanup completed")

# Test code
if __name__ == "__main__":
    # Create controller instance in simulation mode
    controller = HardwareController(simulation_mode=True)
    
    print("\nHardware Controller Test Sequence")
    print("-" * 50)
    print("Testing motor response characteristics")
    print("-" * 50)
    
    # Start controller
    controller.start()
    
    # Test sequence with more realistic steps
    test_sequence = [
        (0, "Initial state"),
        (500, "Low speed"),
        (1000, "Medium speed"),
        (1500, "High speed"),
        (750, "Speed reduction"),
        (0, "Stop")
    ]
    
    try:
        for speed, description in test_sequence:
            print(f"\n{description}:")
            controller.set_target_speed(speed)
            controller.set_duty_cycle(speed / 17.5)  # Adjusted conversion factor
            
            # Monitor response for 2 seconds
            start_time = time.time()
            while time.time() - start_time < 2:
                measurements = controller.get_measurements()
                print(f"Speed: {measurements['speed']:6.1f} RPM | "
                      f"Current: {measurements['current']:4.2f} A | "
                      f"Duty Cycle: {measurements['duty_cycle']:5.1f}%    ", end='\r')
                time.sleep(0.05)
            print()  # New line after each test
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        controller.cleanup()
        print("\nTest completed")