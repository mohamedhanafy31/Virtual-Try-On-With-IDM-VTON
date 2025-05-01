# Using SCHP for Human Parsing in Virtual Try-On

This guide explains how to use the Self-Correction for Human Parsing (SCHP) model for accurate clothing segmentation in virtual try-on applications.

## About SCHP

SCHP (Self-Correction for Human Parsing) is an advanced human parsing model that achieves state-of-the-art segmentation results:
- 82.29% mIoU on the ATR dataset (17,000+ images, 18 labels)
- 59.36% mIoU on the LIP dataset (50,000 images, 19 labels)

The model is particularly effective for fashion-related applications due to its performance on the ATR dataset, which is fashion-focused.

## Getting Started

### 1. Install Dependencies

First, install all required dependencies:

```bash
python -m pip install -r requirements.txt
```

This includes:
- PyTorch and torchvision
- OpenCV
- PIL
- gitpython (for repository cloning)
- ninja (for C++ extension compilation)

### 2. Set up the SCHP Model

First, run the setup script to prepare the directory structure:

```bash
python download_dataset.py --dataset schp
```

This will:
- Clone the SCHP repository (requires gitpython)
- Create the directory structure for models
- Create placeholder files with download instructions

### 3. Download the Model Files

The models need to be manually downloaded from Google Drive due to API restrictions. 

#### Required Downloads

You'll need to download the following models:

##### Most Important (for default configuration)
- [ATR Final Model](https://drive.google.com/file/d/1ruJg4lqR_jgQPj-9K0PP-L2vJERYV5I5/view?usp=sharing)
- Save to: `./models/schp/Self-Correction-Human-Parsing/pretrained/atr/final.pth`

##### Optional Additional Models
- [ATR ResNet101 Model](https://drive.google.com/file/d/1P8rQXqbHNjIDt-JBpTNck2qQlZ1CGuTZ/view?usp=sharing)
- Save to: `./models/schp/Self-Correction-Human-Parsing/pretrained/atr/resnet101.pth`
- [LIP Final Model](https://drive.google.com/file/d/1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH/view?usp=sharing)
- Save to: `./models/schp/Self-Correction-Human-Parsing/pretrained/lip/final.pth`
- [LIP ResNet101 Model](https://drive.google.com/file/d/1kB7aBM-eKb8__OVeW98k1qpnWMPzBcO4/view?usp=sharing)
- Save to: `./models/schp/Self-Correction-Human-Parsing/pretrained/lip/resnet101.pth`

**Note:** The default model used by the segmentation script is the ATR Final Model, so at minimum download that one if you're using the default settings.

### 4. Verify the Model Files

Due to C++ compilation issues that may occur on some systems, we've provided a simplified script that verifies the model files are downloaded correctly:

```bash
python simple_schp_segmentation.py --input_image path/to/your/image.jpg
```

This script will:
- Check if the model file exists
- Verify it's not just a placeholder file
- Provide instructions if the model needs to be downloaded
- Report success if the model file is valid

### 5. Advanced Usage (With Proper Development Environment)

To use the full segmentation functionality, you'll need a proper C++ development environment with:
- Visual Studio with C++ development tools (on Windows)
- CUDA toolkit if using GPU
- A properly configured Python environment

If your environment is properly set up, you can use the full segmentation script:

```bash
python schp_segmentation.py --input_image path/to/your/image.jpg --output_dir ./output
```

Options:
- `--input_image`: Path to the input image (required)
- `--output_dir`: Directory to save output files (default: ./output)
- `--model_path`: Path to SCHP model weights (default: uses ATR model)
- `--dataset`: Dataset the model was trained on, either 'lip' or 'atr' (default: atr)

### 6. Alternative Methods for Using SCHP

If you encounter compilation issues, consider these alternatives:
1. Use SCHP in a Docker container
2. Run the model in Google Colab
3. Use a pre-compiled version in a cloud environment

## Clothing Categories

### ATR Dataset (default)
The ATR dataset includes 18 categories:
1. Background
2. Hat
3. Hair
4. Sunglasses
5. Upper-clothes
6. Skirt
7. Pants
8. Dress
9. Belt
10. Left-shoe
11. Right-shoe
12. Face
13. Left-leg
14. Right-leg
15. Left-arm
16. Right-arm
17. Bag
18. Scarf

### LIP Dataset
The LIP dataset includes 20 categories, with more detailed clothing segmentation.

## Integration with Virtual Try-On Pipeline

To integrate SCHP segmentation into your virtual try-on pipeline:

1. **Replace the current segmentation model**: Use the output from the SCHP model instead of your current segmentation approach
2. **Extract relevant clothing masks**: Use the individual clothing masks (e.g., upper_clothes_mask.png, pants_mask.png) in your try-on process
3. **For finer control**: Modify the `segment_image` function in `schp_segmentation.py` to extract additional clothing segments

## Troubleshooting

### Model Files Not Found
If you see an error about model files not being found:
1. Ensure you've run `python download_dataset.py --dataset schp` to set up the directories
2. Manually download the model files from the Google Drive links provided above
3. Make sure the files are saved to the correct paths shown in the instructions

### Found Placeholder File
If you see an error about finding a placeholder file:
1. This means you need to download the actual model file 
2. Follow the instructions in the placeholder file or download from the Google Drive links above
3. Save the file to the correct path indicated in the error message

### C++ Compilation Issues
If you encounter errors related to C++ compilation:
1. Make sure you have Visual Studio with C++ development tools installed (on Windows)
2. Check that Ninja is properly installed (`python -m pip install ninja`)
3. Consider using the simplified script (`simple_schp_segmentation.py`) which avoids compilation
4. As an alternative, use the SCHP model in a Docker container or cloud environment

### SCHP Import Errors
If you encounter import errors with the SCHP modules:
1. Make sure the repository was cloned successfully
2. If needed, manually clone it: `git clone https://github.com/GoGoDuck912/Self-Correction-Human-Parsing ./models/schp/Self-Correction-Human-Parsing`

## Advanced Usage

### Using with Custom Preprocessing

You can modify the preprocessing in the `segment_image` function to match your specific needs:

```python
# Custom preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
])
```

### Batch Processing

To process multiple images, you can create a loop around the `segment_image` function call in your own script. 