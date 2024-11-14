import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TestWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("DC Motor ANFIS Training Station - Test")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add test button
        self.test_button = ttk.Button(self.main_frame, text="Test Plot", command=self.create_test_plot)
        self.test_button.grid(row=0, column=0, pady=5)
        
        # Create frame for plot
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.grid(row=1, column=0, pady=5)

    def create_test_plot(self):
        # Create sample data
        t = np.linspace(0, 10, 100)
        y = np.sin(t)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, y)
        ax.set_title('Test Sine Wave')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        
        # Embed plot in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)

if __name__ == "__main__":
    root = tk.Tk()
    app = TestWindow(root)
    root.mainloop()