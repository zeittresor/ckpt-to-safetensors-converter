# Requirements
# pip install pytorch
# pip install pytorch_lightning
# pip install safetensors

# Project by github.com/zeittresor
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
from safetensors.torch import save_file
import os

def select_ckpt_file():
    filepath = filedialog.askopenfilename(
        title="Select .ckpt File",
        filetypes=[("Checkpoint Files", "*.ckpt"), ("All Files", "*.*")]
    )
    ckpt_entry.delete(0, tk.END)
    ckpt_entry.insert(0, filepath)

def select_output_file():
    filepath = filedialog.asksaveasfilename(
        title="Save As",
        defaultextension=".safetensors",
        filetypes=[("SafeTensor Files", "*.safetensors"), ("All Files", "*.*")]
    )
    output_entry.delete(0, tk.END)
    output_entry.insert(0, filepath)

def convert():
    ckpt_path = ckpt_entry.get()
    output_path = output_entry.get()
    remove_weights = no_weights_var.get()
    remove_pickles = remove_pickles_var.get()
    ignore_errors = ignore_errors_var.get()
    strip_optimizer = strip_optimizer_var.get()
    use_fp16 = use_fp16_var.get()
    strip_metadata = strip_metadata_var.get()

    if not ckpt_path or not output_path:
        messagebox.showerror("Error", "Please select input and output files.")
        return

    try:
        # Initialize progress bar
        progress_var.set(0)
        root.update_idletasks()

        # Load the .ckpt model with weights_only=True
        try:
            state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        except ModuleNotFoundError as e:
            if 'pytorch_lightning' in str(e):
                messagebox.showerror("Error", "PyTorch Lightning is required to load this model. Please install it and try again.")
                return
            else:
                raise e

        # Update progress bar
        progress_var.set(20)
        root.update_idletasks()

        # Check if the model is under 'state_dict' key
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Strip optimizer state if the option is selected
        if strip_optimizer and 'optimizer_states' in state_dict:
            del state_dict['optimizer_states']

        # Remove pickles if the option is selected
        if remove_pickles:
            state_dict = {k: v for k, v in state_dict.items() if not isinstance(v, bytes)}

        # Remove weights if the option is selected
        if remove_weights:
            for key in list(state_dict.keys()):
                if 'weight' in key or 'bias' in key:
                    del state_dict[key]

        # Strip metadata if the option is selected
        if strip_metadata and 'meta' in state_dict:
            del state_dict['meta']

        # Convert to FP16 if the option is selected
        if use_fp16:
            for key in state_dict.keys():
                tensor = state_dict[key]
                if isinstance(tensor, torch.Tensor):
                    state_dict[key] = tensor.half()

        # Update progress bar
        progress_var.set(60)
        root.update_idletasks()

        # Save as .safetensors model
        save_file(state_dict, output_path)

        # Update progress bar
        progress_var.set(100)
        root.update_idletasks()

        messagebox.showinfo("Success", "Conversion completed successfully!")
    except Exception as e:
        if ignore_errors:
            # Log the error to a file
            log_path = output_path + ".log"
            with open(log_path, 'a') as log_file:
                log_file.write(str(e) + '\n')
            messagebox.showwarning("Warning", f"An error occurred but was ignored:\n{e}\nSee log file for details.")
        else:
            messagebox.showerror("Error", f"An error occurred:\n{e}")

# Function to create tooltips
def create_tooltip(widget, text):
    tool_tip = tk.Toplevel(widget)
    tool_tip.withdraw()
    tool_tip.overrideredirect(True)
    label = tk.Label(tool_tip, text=text, background="#ffffe0", relief='solid', borderwidth=1, font=("tahoma", "8", "normal"))
    label.pack()

    def enter(event):
        x = event.widget.winfo_rootx() + 20
        y = event.widget.winfo_rooty() + 10
        tool_tip.geometry(f"+{x}+{y}")
        tool_tip.deiconify()

    def leave(event):
        tool_tip.withdraw()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

# Create the main window
root = tk.Tk()
root.title("Ckpt to Safetensors Converter")
root.geometry("800x400")

# Input file selection
ckpt_label = tk.Label(root, text="Select .ckpt file:")
ckpt_label.pack(pady=(20, 5))

ckpt_frame = tk.Frame(root)
ckpt_frame.pack()

ckpt_entry = tk.Entry(ckpt_frame, width=60)
ckpt_entry.pack(side=tk.LEFT, padx=(0, 10))

ckpt_button = tk.Button(ckpt_frame, text="Browse...", command=select_ckpt_file)
ckpt_button.pack(side=tk.LEFT)

# Output file selection
output_label = tk.Label(root, text="Select output .safetensors file:")
output_label.pack(pady=(15, 5))

output_frame = tk.Frame(root)
output_frame.pack()

output_entry = tk.Entry(output_frame, width=60)
output_entry.pack(side=tk.LEFT, padx=(0, 10))

output_button = tk.Button(output_frame, text="Browse...", command=select_output_file)
output_button.pack(side=tk.LEFT)

# Options
options_frame = tk.Frame(root)
options_frame.pack(pady=(15, 5))

# No Weights
no_weights_var = tk.BooleanVar()
no_weights_check = tk.Checkbutton(options_frame, text="No Weights", variable=no_weights_var)
no_weights_check.pack(side=tk.LEFT, padx=(0, 15))
create_tooltip(no_weights_check, "Exclude weight and bias parameters from the model.")

# Remove Pickles
remove_pickles_var = tk.BooleanVar()
remove_pickles_check = tk.Checkbutton(options_frame, text="Remove Pickles", variable=remove_pickles_var)
remove_pickles_check.pack(side=tk.LEFT, padx=(0, 15))
create_tooltip(remove_pickles_check, "Remove potentially unsafe pickle objects from the model.")

# Ignore Errors
ignore_errors_var = tk.BooleanVar()
ignore_errors_check = tk.Checkbutton(options_frame, text="Ignore Errors", variable=ignore_errors_var)
ignore_errors_check.pack(side=tk.LEFT)
create_tooltip(ignore_errors_check, "Continue conversion even if errors occur, logging them to a file.")

# Additional Options
options_frame2 = tk.Frame(root)
options_frame2.pack(pady=(10, 5))

# Strip Optimizer State
strip_optimizer_var = tk.BooleanVar()
strip_optimizer_check = tk.Checkbutton(options_frame2, text="Strip Optimizer State", variable=strip_optimizer_var)
strip_optimizer_check.pack(side=tk.LEFT, padx=(0, 15))
create_tooltip(strip_optimizer_check, "Remove optimizer states to reduce file size.")

# Use FP16 Precision
use_fp16_var = tk.BooleanVar()
use_fp16_check = tk.Checkbutton(options_frame2, text="Use FP16 Precision", variable=use_fp16_var)
use_fp16_check.pack(side=tk.LEFT, padx=(0, 15))
create_tooltip(use_fp16_check, "Convert model weights to half-precision (FP16) to save space.")

# Strip Metadata
strip_metadata_var = tk.BooleanVar()
strip_metadata_check = tk.Checkbutton(options_frame2, text="Strip Metadata", variable=strip_metadata_var)
strip_metadata_check.pack(side=tk.LEFT)
create_tooltip(strip_metadata_check, "Remove metadata from the model to reduce file size.")

# Convert button
convert_button = tk.Button(root, text="Convert", command=convert, width=20)
convert_button.pack(pady=(25, 10))

# Progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.pack(fill=tk.X, padx=20, pady=(0, 10))

root.mainloop()
