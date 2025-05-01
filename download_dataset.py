import os
import gdown
import zipfile
import tarfile
import argparse
import shutil
from tqdm import tqdm
import requests
import json

# Try to import gitpython, but provide fallback functionality if not available
try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    print("Warning: gitpython not installed. Manual clone will be required for repositories.")
    print("Install with: python -m pip install gitpython")

def parse_args():
    parser = argparse.ArgumentParser(description="Download and prepare datasets for fine-tuning MediaPipe models")
    parser.add_argument("--dataset", type=str, default="viton-hd", 
                        choices=["viton-hd", "lip", "human36m", "re-id-clothing", "schp"],
                        help="Dataset to download")
    parser.add_argument("--data_dir", type=str, default="./datasets", help="Directory to save datasets")
    parser.add_argument("--roboflow_api_key", type=str, default="", help="Roboflow API key for downloading datasets")
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory to save models")
    return parser.parse_args()

def download_viton_hd(data_dir):
    """
    Download VITON-HD dataset
    Note: Since the official dataset requires registration, we provide a sample subset for demonstration
    """
    os.makedirs(data_dir, exist_ok=True)
    viton_dir = os.path.join(data_dir, "viton-hd")
    os.makedirs(viton_dir, exist_ok=True)
    
    print("Downloading VITON-HD sample dataset...")
    # Sample dataset URLs (for full dataset, registration is required on the official website)
    urls = {
        # Using Google Drive links for demonstration
        # For a real implementation, you should register and download from the official source
        "train_images": "https://drive.google.com/uc?id=1OvDx6z2nB6ABMsYeKI5OMZQyQIy51QTg",
        "train_cloth": "https://drive.google.com/uc?id=1oe1h-cWdnObuHSWdEFzSJLnX8y6KQj_H",
        "train_pose": "https://drive.google.com/uc?id=1UMxm0s4GvYm0nt1JO3nEeZmPHrOV6Tdl",
        "train_mask": "https://drive.google.com/uc?id=1R4Wt_V-35RtNP1jRRKcp0_cLBRIwHzfo",
    }
    
    for name, url in urls.items():
        output_path = os.path.join(viton_dir, f"{name}.zip")
        if not os.path.exists(output_path):
            print(f"Downloading {name}...")
            gdown.download(url, output_path, quiet=False)
        
        # Extract files
        extract_dir = os.path.join(viton_dir, name)
        os.makedirs(extract_dir, exist_ok=True)
        
        if not os.path.exists(os.path.join(extract_dir, "complete")):
            print(f"Extracting {name}...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            # Create a marker file to indicate extraction is complete
            with open(os.path.join(extract_dir, "complete"), 'w') as f:
                f.write("Extraction complete")
    
    print("VITON-HD dataset preparation complete.")
    return viton_dir

def download_lip(data_dir):
    """
    Download LIP (Look Into Person) dataset
    """
    os.makedirs(data_dir, exist_ok=True)
    lip_dir = os.path.join(data_dir, "lip")
    os.makedirs(lip_dir, exist_ok=True)
    
    print("Downloading LIP dataset...")
    # LIP dataset URLs
    urls = {
        "train_images": "https://drive.google.com/uc?id=1BFVXgUcXcWzrJKydWJ2jFqNKW0Lv8dlR",
        "val_images": "https://drive.google.com/uc?id=1KsQOE1hfEqlzIPGUCxFeSpo-7vdgJL-0",
        "train_segmentations": "https://drive.google.com/uc?id=1i4tKYFMOlL6zPPB-bqGe5a4MQS1UYL0n",
        "val_segmentations": "https://drive.google.com/uc?id=1Qknr81JyYbP8IZzbo2eSl1yJG6J4O5Yh",
        "train_poses": "https://drive.google.com/uc?id=1K3j7S5QQo1rIhB6JSRw5kuJ18QLfKmLw",
        "val_poses": "https://drive.google.com/uc?id=1ORnb7vBWmOdAjCJDdGzKUBLEgxEUs12U",
    }
    
    for name, url in urls.items():
        output_path = os.path.join(lip_dir, f"{name}.zip")
        if not os.path.exists(output_path):
            print(f"Downloading {name}...")
            gdown.download(url, output_path, quiet=False)
        
        # Extract files
        extract_dir = os.path.join(lip_dir, name)
        os.makedirs(extract_dir, exist_ok=True)
        
        if not os.path.exists(os.path.join(extract_dir, "complete")):
            print(f"Extracting {name}...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            # Create a marker file to indicate extraction is complete
            with open(os.path.join(extract_dir, "complete"), 'w') as f:
                f.write("Extraction complete")
    
    print("LIP dataset preparation complete.")
    return lip_dir

def download_human36m_sample(data_dir):
    """
    Download Human3.6M sample dataset
    Note: Full dataset requires registration, this is a small sample
    """
    os.makedirs(data_dir, exist_ok=True)
    human36m_dir = os.path.join(data_dir, "human36m")
    os.makedirs(human36m_dir, exist_ok=True)
    
    print("Downloading Human3.6M sample dataset...")
    # Sample dataset URL (full dataset requires registration)
    sample_url = "https://drive.google.com/uc?id=1tX4RByO6oreHgNxJGhQlnnZ96teYa7J7"
    
    output_path = os.path.join(human36m_dir, "sample.zip")
    if not os.path.exists(output_path):
        print("Downloading sample data...")
        gdown.download(sample_url, output_path, quiet=False)
    
    # Extract files
    if not os.path.exists(os.path.join(human36m_dir, "complete")):
        print("Extracting sample data...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(human36m_dir)
        # Create a marker file to indicate extraction is complete
        with open(os.path.join(human36m_dir, "complete"), 'w') as f:
            f.write("Extraction complete")
    
    print("Human3.6M sample dataset preparation complete.")
    return human36m_dir

def download_re_id_clothing(data_dir, api_key):
    """
    Download re-id-clothing-accessories dataset from Roboflow
    """
    os.makedirs(data_dir, exist_ok=True)
    reid_dir = os.path.join(data_dir, "re-id-clothing")
    os.makedirs(reid_dir, exist_ok=True)
    
    print("Downloading re-id-clothing-accessories dataset from Roboflow...")
    
    if not api_key:
        print("Warning: No Roboflow API key provided. Using public dataset URL if available.")
        # For demonstration only - this won't actually work without proper authentication
        print("Please visit https://universe.roboflow.com/comp303dissertation/re-id-clothing-accessories-0w4kl")
        print("to get proper download instructions and API key.")
        
        # Create marker file to indicate attempted download
        with open(os.path.join(reid_dir, "DOWNLOAD_FAILED_NO_API_KEY"), 'w') as f:
            f.write("Please provide a Roboflow API key to download this dataset.")
        
        # Create basic directory structure that would be expected
        os.makedirs(os.path.join(reid_dir, "train", "images"), exist_ok=True)
        os.makedirs(os.path.join(reid_dir, "train", "annotations"), exist_ok=True)
        os.makedirs(os.path.join(reid_dir, "valid", "images"), exist_ok=True)
        os.makedirs(os.path.join(reid_dir, "valid", "annotations"), exist_ok=True)
        os.makedirs(os.path.join(reid_dir, "test", "images"), exist_ok=True)
        os.makedirs(os.path.join(reid_dir, "test", "annotations"), exist_ok=True)
        
        print("Created placeholder directory structure. Please download dataset manually.")
        return reid_dir
    
    # If we have an API key, download using the Roboflow API
    version = 1  # Default version
    format_type = "instance-segmentation"
    
    # Construct API endpoints
    download_url = f"https://app.roboflow.com/ds/re-id-clothing-accessories-0w4kl?key={api_key}"
    
    # Use requests to download the dataset
    try:
        print(f"Downloading dataset using Roboflow API...")
        response = requests.get(download_url, stream=True)
        
        if response.status_code != 200:
            print(f"Error downloading dataset: HTTP {response.status_code}")
            print(response.text)
            raise Exception("Download failed")
        
        # Save zip file
        zip_path = os.path.join(reid_dir, "dataset.zip")
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Extract dataset
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(reid_dir)
        
        # Remove zip file to save space
        os.remove(zip_path)
        
        # Create a marker file to indicate successful download
        with open(os.path.join(reid_dir, "complete"), 'w') as f:
            f.write("Download complete")
            
        print("re-id-clothing-accessories dataset download complete.")
    
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("Please try downloading manually from https://universe.roboflow.com/comp303dissertation/re-id-clothing-accessories-0w4kl")
    
    return reid_dir

def download_file(url, output_path):
    """
    Download a file from a direct URL using requests
    """
    try:
        print(f"Downloading file from {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return True
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return False

def download_schp_model(model_dir):
    """
    Download Self-Correction for Human Parsing (SCHP) model
    """
    os.makedirs(model_dir, exist_ok=True)
    schp_dir = os.path.join(model_dir, "schp")
    os.makedirs(schp_dir, exist_ok=True)
    
    print("Setting up Self-Correction for Human Parsing (SCHP) model...")
    
    # Clone the SCHP repository
    schp_repo_dir = os.path.join(schp_dir, "Self-Correction-Human-Parsing")
    if not os.path.exists(schp_repo_dir):
        print("Cloning SCHP repository...")
        
        if GIT_AVAILABLE:
            try:
                git.Repo.clone_from("https://github.com/GoGoDuck912/Self-Correction-Human-Parsing", schp_repo_dir)
            except Exception as e:
                print(f"Error cloning repository: {str(e)}")
                print("Please clone the repository manually:")
                print(f"git clone https://github.com/GoGoDuck912/Self-Correction-Human-Parsing {schp_repo_dir}")
                # Create the directory structure for models anyway
                os.makedirs(schp_repo_dir, exist_ok=True)
        else:
            print("gitpython is not installed. Please clone the repository manually:")
            print(f"git clone https://github.com/GoGoDuck912/Self-Correction-Human-Parsing {schp_repo_dir}")
            # Create the directory structure for models anyway
            os.makedirs(schp_repo_dir, exist_ok=True)
    
    # Download pre-trained models
    models_dir = os.path.join(schp_repo_dir, "pretrained")
    os.makedirs(models_dir, exist_ok=True)
    
    # Create placeholder model files that the segmentation script can detect
    # This gives users a clear path to manual download if needed
    for dataset in ["lip", "atr"]:
        dataset_dir = os.path.join(models_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        for model_name in ["final", "resnet101"]:
            output_path = os.path.join(dataset_dir, f"{model_name}.pth")
            if not os.path.exists(output_path):
                # Create an empty file as a placeholder with instructions
                with open(output_path, 'w') as f:
                    f.write("# This is a placeholder file. Please download the actual model.\n")
                    f.write("# For ATR dataset models:\n")
                    f.write("#   final.pth: https://drive.google.com/file/d/1ruJg4lqR_jgQPj-9K0PP-L2vJERYV5I5/view?usp=sharing\n")
                    f.write("#   resnet101.pth: https://drive.google.com/file/d/1P8rQXqbHNjIDt-JBpTNck2qQlZ1CGuTZ/view?usp=sharing\n")
                    f.write("# For LIP dataset models:\n")
                    f.write("#   final.pth: https://drive.google.com/file/d/1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH/view?usp=sharing\n")
                    f.write("#   resnet101.pth: https://drive.google.com/file/d/1kB7aBM-eKb8__OVeW98k1qpnWMPzBcO4/view?usp=sharing\n")
    
    # These are the Google Drive URLs for the models
    google_drive_urls = {
        "lip": {
            "final": "https://drive.google.com/file/d/1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH/view?usp=sharing",
            "resnet101": "https://drive.google.com/file/d/1kB7aBM-eKb8__OVeW98k1qpnWMPzBcO4/view?usp=sharing"
        },
        "atr": {
            "final": "https://drive.google.com/file/d/1ruJg4lqR_jgQPj-9K0PP-L2vJERYV5I5/view?usp=sharing",
            "resnet101": "https://drive.google.com/file/d/1P8rQXqbHNjIDt-JBpTNck2qQlZ1CGuTZ/view?usp=sharing"
        }
    }
    
    # Construct manual download instructions
    manual_download_instructions = []
    for dataset, models in google_drive_urls.items():
        for model_name, url in models.items():
            output_path = os.path.join(models_dir, dataset, f"{model_name}.pth")
            instruction = f"- Download {dataset} {model_name} model from: {url}"
            instruction += f"\n  Save it to: {output_path}"
            manual_download_instructions.append(instruction)
    
    # Create a marker file to indicate setup
    with open(os.path.join(schp_dir, "setup_instructions"), 'w') as f:
        f.write("SCHP Model Setup Instructions\n\n")
        f.write("Please download the following files manually:\n\n")
        for instruction in manual_download_instructions:
            f.write(instruction + "\n\n")
    
    print("\nSCHP model setup prepared. Manual download required.")
    print("Repository and folder structure have been set up at:", schp_repo_dir)
    print("\nMANUAL DOWNLOAD REQUIRED:")
    print("Due to download restrictions, you need to manually download the model files:")
    for instruction in manual_download_instructions:
        print(instruction)
    
    print("\nAfter downloading the files, you can run the segmentation script.")
    print("\nUsage instructions:")
    print("1. The SCHP repository is available at:", schp_repo_dir)
    print("2. Pre-trained models should be saved in the 'pretrained' folder")
    print("3. See the README.md file in the repository for usage details")
    
    return schp_dir

def main():
    args = parse_args()
    
    if args.dataset == "viton-hd":
        dataset_dir = download_viton_hd(args.data_dir)
    elif args.dataset == "lip":
        dataset_dir = download_lip(args.data_dir)
    elif args.dataset == "human36m":
        dataset_dir = download_human36m_sample(args.data_dir)
    elif args.dataset == "re-id-clothing":
        dataset_dir = download_re_id_clothing(args.data_dir, args.roboflow_api_key)
    elif args.dataset == "schp":
        dataset_dir = download_schp_model(args.model_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"Dataset/model prepared at: {dataset_dir}")
    print("\nTo download another dataset/model, run this script with a different --dataset option.")

if __name__ == "__main__":
    main() 