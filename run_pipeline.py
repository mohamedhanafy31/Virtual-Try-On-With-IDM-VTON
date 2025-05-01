#!/usr/bin/env python
import os
import argparse
import subprocess
import time
import json
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Run MediaPipe fine-tuning pipeline")
    parser.add_argument("--task", type=str, default="segmentation", choices=["segmentation", "pose"],
                        help="Task to fine-tune (segmentation or pose)")
    parser.add_argument("--dataset", type=str, default="viton-hd", choices=["viton-hd", "lip", "human36m", "re-id-clothing"],
                        help="Dataset to use")
    parser.add_argument("--base_dir", type=str, default="./mediapipe_finetuning",
                        help="Base directory for all data and outputs")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip dataset download step")
    parser.add_argument("--skip_preprocessing", action="store_true",
                        help="Skip dataset preprocessing step")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip model training step")
    parser.add_argument("--visualize", action="store_true",
                        help="Enable visualization during evaluation")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--roboflow_api_key", type=str, default="",
                        help="API key for downloading Roboflow datasets")
    return parser.parse_args()

def run_command(cmd, description):
    """Run a shell command and print output"""
    print(f"=== {description} ===")
    print(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding='utf-8',
        errors='ignore'
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    end_time = time.time()
    
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        return False
    
    print(f"Command completed in {end_time - start_time:.2f} seconds")
    print("")
    return True

def create_directories(args):
    """Create necessary directories"""
    # Base directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Dataset directory
    dataset_dir = os.path.join(args.base_dir, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Processed data directory
    processed_dir = os.path.join(args.base_dir, "processed_data")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Output directory
    output_dir = os.path.join(args.base_dir, "trained_models", f"{args.dataset}_{args.task}")
    os.makedirs(output_dir, exist_ok=True)
    
    return dataset_dir, processed_dir, output_dir

def create_env_file(args):
    """Create a JSON file with environment configuration"""
    config = {
        "task": args.task,
        "dataset": args.dataset,
        "base_dir": args.base_dir,
        "dataset_dir": os.path.join(args.base_dir, "datasets"),
        "processed_dir": os.path.join(args.base_dir, "processed_data"),
        "output_dir": os.path.join(args.base_dir, "trained_models", f"{args.dataset}_{args.task}"),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "visualize": args.visualize,
        "fp16": args.fp16,
        "timestamp": time.strftime("%Y%m%d-%H%M%S")
    }
    
    # Save config
    config_path = os.path.join(args.base_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU available: {gpu_name} with {memory:.2f} GB memory")
            return True
        else:
            print("No GPU available, using CPU")
            return False
    except:
        print("Failed to check GPU, assuming CPU only")
        return False

def main():
    args = parse_args()
    
    # Check for GPU
    has_gpu = check_gpu()
    
    # Create necessary directories
    dataset_dir, processed_dir, output_dir = create_directories(args)
    
    # Create environment configuration file
    config_path = create_env_file(args)
    print(f"Configuration saved to {config_path}")
    
    # Get current Python executable path
    python_executable = sys.executable
    print(f"Using Python executable: {python_executable}")
    
    # Step 1: Download dataset if not skipped
    if not args.skip_download:
        download_cmd = [
            python_executable, "download_dataset.py", 
            "--dataset", args.dataset, 
            "--data_dir", dataset_dir
        ]
        
        # Add API key if provided and using Roboflow dataset
        if args.dataset == "re-id-clothing" and args.roboflow_api_key:
            download_cmd.extend(["--roboflow_api_key", args.roboflow_api_key])
        
        success = run_command(
            download_cmd,
            f"Downloading {args.dataset} dataset"
        )
        if not success:
            print("Download step failed. Please check dependencies or try again with --skip_download flag.")
            return
    
    # Step 2: Preprocess dataset if not skipped
    if not args.skip_preprocessing:
        preprocess_cmd = [
            python_executable, "preprocess_dataset.py",
            "--dataset", args.dataset,
            "--data_dir", dataset_dir,
            "--output_dir", processed_dir,
            "--visualize"
        ]
        
        # Add MediaPipe baseline for comparison
        preprocess_cmd.append("--mediapipe_baseline")
        
        success = run_command(
            preprocess_cmd,
            f"Preprocessing {args.dataset} dataset"
        )
        if not success:
            print("Preprocessing step failed. Please check for errors or try again with --skip_preprocessing flag.")
            return
    
    # Step 3: Train model if not skipped
    if not args.skip_training:
        train_cmd = [
            python_executable, "train.py",
            "--task", args.task,
            "--data_dir", os.path.join(processed_dir, args.dataset),
            "--output_dir", output_dir,
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--num_workers", str(args.num_workers),
            "--seed", "42",
            "--log_interval", "10",
            "--eval_interval", "1",
            "--save_interval", "5"
        ]
        
        # For re-id-clothing dataset, we might need to adjust some parameters
        if args.dataset == "re-id-clothing":
            # Add argument for number of classes (13 clothing categories + background)
            train_cmd.extend(["--num_classes", "14"])
        
        # Add visualization flag if specified
        if args.visualize:
            train_cmd.append("--visualize")
        
        # Add FP16 flag if GPU is available and flag is set
        if has_gpu and args.fp16:
            train_cmd.append("--fp16")
        
        success = run_command(
            train_cmd,
            f"Training {args.task} model on {args.dataset}"
        )
        if not success:
            print("Training step failed. Please check for errors.")
            return
    
    # Print completion message
    print(f"""
=== Pipeline Completed Successfully ===

Task: {args.task}
Dataset: {args.dataset}
Output directory: {output_dir}

To use the trained model for inference, you can:
1. Load the best model checkpoint from {os.path.join(output_dir, f"{args.task}_best.pth")}
2. View training logs with TensorBoard: tensorboard --logdir={os.path.join(output_dir, "logs")}
""")

if __name__ == "__main__":
    main() 