# CLIP Vision Model Fine-tuning for Virtual Try-On

This repository contains code to fine-tune the CLIP vision model for the Virtual Try-On system. The fine-tuned model will improve the vision encoder's understanding of clothing items, leading to better virtual try-on results.

## Overview

The script `train_clip_encoder.py` fine-tunes the CLIP vision model on a clothing dataset, which includes pairs of clothing images and their text descriptions. This allows the model to better understand the semantic relationship between clothing items and their descriptions, improving the overall performance of the virtual try-on system.

## Dataset Structure

The script expects a dataset with the following structure:

```
merged_dataset/
├── metadata.csv       # Contains text descriptions and metadata for each item
├── train_split.csv    # List of UIDs for training
├── val_split.csv      # List of UIDs for validation
└── images/            # Directory containing the clothing images
    ├── 0000000.jpg
    ├── 0000001.jpg
    ├── 0000002.jpg
    └── ...
```

The CSV files should have the following structure:

- `metadata.csv`: Contains at least `uid`, `text`, and `category` columns
- `train_split.csv` and `val_split.csv`: Contains at least the `uid` column

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd <repository_directory>
```

2. Install the required dependencies:
```bash
pip install torch torchvision transformers pandas numpy pillow tqdm wandb matplotlib albumentations scikit-learn
```

## Usage

### Fine-tuning on Google Colab

To fine-tune the CLIP vision model on Google Colab:

1. Upload the `train_clip_encoder.py` script to your Colab environment
2. Mount your Google Drive containing the dataset
3. Run the script with the appropriate arguments:

```bash
!python train_clip_encoder.py \
  --dataset_dir "/content/drive/MyDrive/merged_dataset" \
  --train_csv "train_split.csv" \
  --val_csv "val_split.csv" \
  --metadata_csv "metadata.csv" \
  --output_dir "/content/drive/MyDrive/best_model" \
  --batch_size 16 \
  --num_epochs 5 \
  --learning_rate 2e-5 \
  --freeze_text_encoder
```

### Fine-tuning Locally

To fine-tune the model locally:

```bash
python train_clip_encoder.py \
  --dataset_dir "/path/to/merged_dataset" \
  --output_dir "./best_model" \
  --batch_size 8 \
  --num_epochs 10 \
  --learning_rate 3e-5 \
  --freeze_text_encoder
```

### Key Arguments

- `--dataset_dir`: Path to the dataset directory
- `--train_csv`: Name of the training CSV file (default: "train_split.csv")
- `--val_csv`: Name of the validation CSV file (default: "val_split.csv")
- `--metadata_csv`: Name of the metadata CSV file (default: "metadata.csv")
- `--model_name`: Pretrained CLIP model to use (default: "openai/clip-vit-base-patch32")
- `--output_dir`: Directory to save the fine-tuned model (default: "./best_model")
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of training epochs (default: 10)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--freeze_text_encoder`: Freeze the text encoder during training (only fine-tune the vision model)
- `--use_wandb`: Use Weights & Biases for logging

Run `python train_clip_encoder.py --help` for the full list of options.

## Integration with Virtual Try-On System

To use the fine-tuned model in the Virtual Try-On system:

1. Fine-tune the model using the script and save it to the `best_model` directory
2. Make sure your Virtual Try-On code looks for the model in this directory:

```python
finetuned_clip_path = "best_model"  # Path to your fine-tuned model
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    finetuned_clip_path,
    torch_dtype=weight_dtype,
    local_files_only=True
).to(device)
```

3. Use the same processor for preprocessing:

```python
clip_processor = CLIPImageProcessor.from_pretrained(finetuned_clip_path, local_files_only=True)
```

## Tips for Better Results

1. Use a larger dataset with diverse clothing items and descriptions
2. Try different learning rates and batch sizes
3. Experiment with freezing or unfreezing the text encoder
4. Use mixed precision training for faster training on GPUs
5. Add more data augmentation if the dataset is small

## Example for Google Colab (Full Script)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required dependencies
!pip install transformers pandas numpy pillow tqdm wandb matplotlib albumentations scikit-learn

# Run the fine-tuning script
!python train_clip_encoder.py \
  --dataset_dir "/content/drive/MyDrive/merged_dataset" \
  --output_dir "/content/drive/MyDrive/best_model" \
  --batch_size 16 \
  --num_epochs 5 \
  --learning_rate 2e-5 \
  --freeze_text_encoder \
  --use_wandb
``` 