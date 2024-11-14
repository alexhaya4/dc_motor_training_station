import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import numpy as np
from main_controller import MainController
from typing import Dict, Any

class TrainingStationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DC Motor ANFIS Training Station")
        self.root.state('zoomed')  # Full screen
        
        # Initialize system controller
        self.controller = MainController(simulation_mode=True)
        
        # Data for plotting
        self.time_data = []
        self.speed_data = []
        self.current_data = []
        self.setpoint_data = []
        
        # GUI state
        self.running = False
        self.current_setpoint = 0
        
        # Setup GUI components
        self.setup_gui()
        
        # Start update loop
        self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()

    def setup_gui(self):
        # Create main layout frames
        self.create_control_panel()
        self.create_visualization_panel()
        self.create_data_panel()
        self.create_status_bar()

    def create_control_panel(self):
        # Control Panel
        control_frame = ttk.LabelFrame(self.root, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        # Mode Selection
        mode_frame = ttk.LabelFrame(control_frame, text="Operation Mode")
        mode_frame.pack(fill="x", padx=5, pady=5)
        
        self.mode_var = tk.StringVar(value="manual")
        ttk.Radiobutton(mode_frame, text="Manual Control", 
                       variable=self.mode_var, value="manual").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="ANFIS Control", 
                       variable=self.mode_var, value="anfis").pack(side="left", padx=5)
        
        # Speed Control
        speed_frame = ttk.LabelFrame(control_frame, text="Speed Control")
        speed_frame.pack(fill="x", padx=5, pady=5)
        
        self.speed_var = tk.DoubleVar(value=0)
        self.speed_scale = ttk.Scale(speed_frame, from_=0, to=1750,
                                   variable=self.speed_var,
                                   orient="horizontal",
                                   command=self.on_speed_change)
        self.speed_scale.pack(fill="x", padx=5, pady=5)
        
        self.speed_label = ttk.Label(speed_frame, text="0 RPM")
        self.speed_label.pack()
        
        # Operation Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start",
                                     command=self.start_system)
        self.start_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop",
                                    command=self.stop_system)
        self.stop_button.pack(side="left", padx=5)
        
        emergency_button = ttk.Button(button_frame, text="EMERGENCY STOP",
                                    style="Emergency.TButton",
                                    command=self.emergency_stop)
        emergency_button.pack(side="left", padx=20)

    def create_visualization_panel(self):
        # Visualization Panel
        viz_frame = ttk.LabelFrame(self.root, text="Real-time Monitoring")
        viz_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=5, sticky="nsew")
        
        # Create matplotlib figure
        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        
        # Speed plot
        self.ax1.set_title('Speed Response')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Speed (RPM)')
        self.speed_line, = self.ax1.plot([], [], 'b-', label='Actual')
        self.setpoint_line, = self.ax1.plot([], [], 'r--', label='Setpoint')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Current plot
        self.ax2.set_title('Armature Current')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Current (A)')
        self.current_line, = self.ax2.plot([], [], 'g-')
        self.ax2.grid(True)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.figure, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_data_panel(self):
        # Data Panel
        data_frame = ttk.LabelFrame(self.root, text="System Parameters")
        data_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Create parameter displays
        params = ['Speed (RPM)', 'Current (A)', 'Duty Cycle (%)', 
                 'Error (RPM)', 'Control Output']
        self.param_labels = {}
        
        for i, param in enumerate(params):
            ttk.Label(data_frame, text=param + ":").grid(row=i, column=0, padx=5, pady=2)
            self.param_labels[param] = ttk.Label(data_frame, text="0")
            self.param_labels[param].grid(row=i, column=1, padx=5, pady=2)

    def create_status_bar(self):
        # Status Bar
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        
        self.status_label = ttk.Label(status_frame, text="System Ready")
        self.status_label.pack(side="left", padx=5)
        
        self.connection_label = ttk.Label(status_frame, text="Simulation Mode")
        self.connection_label.pack(side="right", padx=5)

    def update_loop(self):
        """Background loop for updating measurements and plots"""
        update_interval = 0.05  # 50ms update interval
        plot_length = 20  # 20 seconds of data
        
        while True:
            if self.running:
                try:
                    # Get measurements
                    measurements = self.controller.hardware.get_measurements()
                    
                    # Update data arrays
                    current_time = time.time()
                    if not self.time_data:
                        start_time = current_time
                        self.time_data = [0]
                    else:
                        self.time_data.append(current_time - start_time)
                    
                    self.speed_data.append(measurements['speed'])
                    self.current_data.append(measurements['current'])
                    self.setpoint_data.append(self.current_setpoint)
                    
                    # Limit data length
                    if self.time_data[-1] > plot_length:
                        crop_idx = next(i for i, t in enumerate(self.time_data) 
                                      if t > self.time_data[-1] - plot_length)
                        self.time_data = self.time_data[crop_idx:]
                        self.speed_data = self.speed_data[crop_idx:]
                        self.current_data = self.current_data[crop_idx:]
                        self.setpoint_data = self.setpoint_data[crop_idx:]
                    
                    # Update plots and labels
                    self.root.after(0, self.update_display)
                    
                except Exception as e:
                    print(f"Update error: {str(e)}")
                
            time.sleep(update_interval)

    def update_display(self):
        """Update plots and parameter displays"""
        try:
            # Update speed plot
            self.speed_line.set_data(self.time_data, self.speed_data)
            self.setpoint_line.set_data(self.time_data, self.setpoint_data)
            self.ax1.relim()
            self.ax1.autoscale_view()
            
            # Update current plot
            self.current_line.set_data(self.time_data, self.current_data)
            self.ax2.relim()
            self.ax2.autoscale_view()
            
            self.canvas.draw()
            
            # Update parameter labels
            if self.speed_data:
                self.param_labels['Speed (RPM)'].config(
                    text=f"{self.speed_data[-1]:.1f}")
                self.param_labels['Current (A)'].config(
                    text=f"{self.current_data[-1]:.2f}")
                self.param_labels['Error (RPM)'].config(
                    text=f"{self.current_setpoint - self.speed_data[-1]:.1f}")
                
            measurements = self.controller.hardware.get_measurements()
            self.param_labels['Duty Cycle (%)'].config(
                text=f"{measurements['duty_cycle']:.1f}")
            
        except Exception as e:
            print(f"Display update error: {str(e)}")

    def on_speed_change(self, event=None):
        """Handle speed setpoint changes"""
        self.current_setpoint = self.speed_var.get()
        self.speed_label.config(text=f"{self.current_setpoint:.0f} RPM")
        if self.running:
            self.controller.set_target_speed(self.current_setpoint)

    def start_system(self):
        """Start system operation"""
        if not self.running:
            self.controller.start()
            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status_label.config(text="System Running")

    def stop_system(self):
        """Stop system operation"""
        if self.running:
            self.controller.stop()
            self.running = False
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.status_label.config(text="System Stopped")

    def emergency_stop(self):
        """Handle emergency stop"""
        self.controller.emergency_stop()
        self.running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="EMERGENCY STOP ACTIVATED")

    def cleanup(self):
        """Cleanup before closing"""
        self.stop_system()
        self.controller.cleanup()

def main():
    # Create and configure root window
    root = tk.Tk()
    
    # Configure style
    style = ttk.Style()
    style.configure("Emergency.TButton", 
                   background="red",
                   foreground="white",
                   padding=10,
                   font=('Helvetica', 12, 'bold'))
    
    # Create application
    app = TrainingStationGUI(root)
    
    # Setup window close handler
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    
    # Start GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()