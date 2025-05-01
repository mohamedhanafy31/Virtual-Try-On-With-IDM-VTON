import os
import cv2
import numpy as np
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random


class MediaPipeDataset(Dataset):
    """Dataset for training MediaPipe-inspired models"""
    def __init__(self, root_dir, task='segmentation', split='train', transform=None, augment=False):
        """
        Args:
            root_dir: Root directory containing the processed dataset
            task: Task type - 'segmentation', 'pose', or 'face'
            split: 'train' or 'val'
            transform: Optional transforms to apply to images
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.task = task
        self.split = split
        self.transform = transform
        self.augment = augment
        
        # Load metadata
        with open(os.path.join(root_dir, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)
        
        # Split dataset
        if split == 'val':
            # Use 10% of data for validation
            self.metadata = self.metadata[::10]
        elif split == 'train':
            # Remove validation samples
            train_indices = [i for i in range(len(self.metadata)) if i % 10 != 0]
            self.metadata = [self.metadata[i] for i in train_indices]
        
        # Directories
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")
        self.poses_dir = os.path.join(root_dir, "poses")
        
        # Default transforms if none provided
        if self.transform is None:
            if self.task == 'segmentation' or self.task == 'face':
                self.transform = transforms.Compose([
                    transforms.Resize((256, 192)),  # Smaller resolution for training
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            elif self.task == 'pose':
                self.transform = transforms.Compose([
                    transforms.Resize((384, 288)),  # Higher resolution for pose estimation
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get metadata for sample
        sample_meta = self.metadata[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, sample_meta['image'])
        image = Image.open(img_path).convert('RGB')
        
        # Initialize result dictionary
        result = {}
        
        # Get original image size for later normalization
        orig_width, orig_height = image.size
        
        # Apply augmentation if enabled
        if self.augment and self.split == 'train':
            image, augmented = self._apply_augmentation(image, idx)
        else:
            augmented = False
        
        # Apply transformation
        image_tensor = self.transform(image)
        result['image'] = image_tensor
        
        # Process according to task
        if self.task == 'segmentation':
            # Load mask
            mask_path = os.path.join(self.masks_dir, sample_meta['mask'])
            mask = Image.open(mask_path).convert('L')
            
            # Apply same augmentation to mask if needed
            if augmented:
                mask = self._apply_mask_augmentation(mask, idx)
            
            # Resize mask to match image size
            mask = mask.resize((image_tensor.shape[2], image_tensor.shape[1]), Image.NEAREST)
            
            # Convert to tensor
            mask_tensor = transforms.ToTensor()(mask)
            
            # Binarize mask
            mask_tensor = (mask_tensor > 0.5).float()
            
            result['mask'] = mask_tensor
        
        elif self.task == 'pose':
            # Load pose data
            pose_path = os.path.join(self.poses_dir, sample_meta['pose'])
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
            
            # Extract keypoints
            keypoints = pose_data['keypoints']
            
            # Convert to tensor format [num_keypoints, 3] (x, y, visibility)
            keypoint_tensor = torch.zeros(len(keypoints), 3)
            
            # Current tensor image size
            tensor_width, tensor_height = image_tensor.shape[2], image_tensor.shape[1]
            
            for i, kp in enumerate(keypoints):
                # Normalize coordinates to [0, 1]
                x = kp['x'] / orig_width
                y = kp['y'] / orig_height
                
                # Apply augmentation transforms if needed
                if augmented:
                    x, y = self._apply_keypoint_augmentation(x, y, idx)
                
                # Store normalized coordinates and visibility
                keypoint_tensor[i, 0] = x
                keypoint_tensor[i, 1] = y
                keypoint_tensor[i, 2] = float(kp['confidence']) if 'confidence' in kp else 1.0
            
            result['keypoints'] = keypoint_tensor
            
            # Create pose target heatmaps for visualization and training
            heatmaps = self._generate_heatmaps(keypoint_tensor, tensor_width, tensor_height)
            result['heatmaps'] = heatmaps
        
        # Include metadata for reference
        result['meta'] = {
            'file_name': sample_meta['image'],
            'original_size': (orig_width, orig_height)
        }
        
        return result
    
    def _apply_augmentation(self, image, idx):
        """Apply data augmentation to image"""
        # Set random seed based on idx for reproducibility
        random.seed(idx)
        
        # Initialize augmentation transforms
        augmented = False
        
        # Random horizontal flip (50% probability)
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            augmented = True
        
        # Random rotation (±10 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            image = transforms.functional.rotate(image, angle, fill=(0, 0, 0))
            augmented = True
        
        # Random brightness and contrast adjustment
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            
            image = transforms.functional.adjust_brightness(image, brightness)
            image = transforms.functional.adjust_contrast(image, contrast)
            image = transforms.functional.adjust_saturation(image, saturation)
            augmented = True
        
        return image, augmented
    
    def _apply_mask_augmentation(self, mask, idx):
        """Apply same augmentation to mask as applied to image"""
        # Set random seed based on idx for reproducibility
        random.seed(idx)
        
        # Apply same augmentations as in _apply_augmentation
        if random.random() > 0.5:
            mask = transforms.functional.hflip(mask)
        
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            mask = transforms.functional.rotate(mask, angle, fill=0)
        
        # Skip brightness/contrast for masks
        
        return mask
    
    def _apply_keypoint_augmentation(self, x, y, idx):
        """Apply augmentation to keypoint coordinates"""
        # Set random seed based on idx for reproducibility
        random.seed(idx)
        
        # Apply same augmentations as in _apply_augmentation
        if random.random() > 0.5:
            # Horizontal flip
            x = 1.0 - x
        
        if random.random() > 0.5:
            # Rotation is more complex for keypoints, simplified approximation
            angle = random.uniform(-10, 10) * (3.14159 / 180.0)  # Convert to radians
            
            # Move to origin
            x -= 0.5
            y -= 0.5
            
            # Rotate
            new_x = x * np.cos(angle) - y * np.sin(angle)
            new_y = x * np.sin(angle) + y * np.cos(angle)
            
            # Move back
            x = new_x + 0.5
            y = new_y + 0.5
        
        return x, y
    
    def _generate_heatmaps(self, keypoints, width, height, sigma=3):
        """Generate target heatmaps for keypoints"""
        # keypoints: [num_keypoints, 3] (x, y, visibility)
        num_keypoints = keypoints.shape[0]
        
        # Create empty heatmaps
        heatmaps = torch.zeros(num_keypoints, height, width)
        
        # Create coordinate grids
        x_grid = torch.arange(width).float().view(1, width).repeat(height, 1)
        y_grid = torch.arange(height).float().view(height, 1).repeat(1, width)
        
        # Convert normalized keypoint coordinates to pixel coordinates
        for i in range(num_keypoints):
            # Skip if keypoint is not visible
            if keypoints[i, 2] < 0.1:
                continue
            
            # Convert to pixel coordinates
            kp_x = keypoints[i, 0] * width
            kp_y = keypoints[i, 1] * height
            
            # Generate Gaussian heatmap
            heatmap = torch.exp(-((x_grid - kp_x) ** 2 + (y_grid - kp_y) ** 2) / (2 * sigma ** 2))
            
            # Normalize heatmap
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            heatmaps[i] = heatmap
        
        return heatmaps

class SegmentationDataset(Dataset):
    """Dataset for segmentation tasks"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        self.samples = []
        
        # Determine dataset type based on directory structure
        if os.path.exists(os.path.join(data_dir, 'viton-hd')):
            self._load_viton_hd_samples()
        elif os.path.exists(os.path.join(data_dir, 'lip')):
            self._load_lip_samples()
        elif os.path.exists(os.path.join(data_dir, 'human36m')):
            self._load_human36m_samples()
        elif os.path.exists(os.path.join(data_dir, 're-id-clothing')):
            self._load_reid_clothing_samples()
        else:
            raise ValueError(f"Unsupported dataset format in {data_dir}")
    
    def _load_viton_hd_samples(self):
        """Load VITON-HD dataset samples"""
        split_dir = os.path.join(self.data_dir, 'viton-hd', self.split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} not found")
        
        # Get image files
        for filename in os.listdir(split_dir):
            if filename.endswith('_image.png'):
                img_path = os.path.join(split_dir, filename)
                # Get corresponding mask filename
                mask_filename = filename.replace('_image.png', '_mask.png')
                mask_path = os.path.join(split_dir, mask_filename)
                
                if os.path.exists(mask_path):
                    self.samples.append({
                        'image': img_path,
                        'mask': mask_path
                    })
    
    def _load_lip_samples(self):
        """Load LIP dataset samples"""
        # Similar structure to VITON-HD loading
        split_dir = os.path.join(self.data_dir, 'lip', self.split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} not found")
        
        # Get image files
        for filename in os.listdir(split_dir):
            if filename.endswith('_image.png'):
                img_path = os.path.join(split_dir, filename)
                # Get corresponding mask filename
                mask_filename = filename.replace('_image.png', '_mask.png')
                mask_path = os.path.join(split_dir, mask_filename)
                
                if os.path.exists(mask_path):
                    self.samples.append({
                        'image': img_path,
                        'mask': mask_path
                    })
    
    def _load_human36m_samples(self):
        """Load Human3.6M dataset samples"""
        # Similar structure to VITON-HD loading
        split_dir = os.path.join(self.data_dir, 'human36m', self.split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} not found")
        
        # Get image files
        for filename in os.listdir(split_dir):
            if filename.endswith('_image.png'):
                img_path = os.path.join(split_dir, filename)
                # Get corresponding mask filename
                mask_filename = filename.replace('_image.png', '_mask.png')
                mask_path = os.path.join(split_dir, mask_filename)
                
                if os.path.exists(mask_path):
                    self.samples.append({
                        'image': img_path,
                        'mask': mask_path
                    })
    
    def _load_reid_clothing_samples(self):
        """Load re-id-clothing dataset samples"""
        # Maps to val if split is validation
        split_name = 'val' if self.split == 'validation' else self.split
        split_dir = os.path.join(self.data_dir, 're-id-clothing', split_name)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} not found")
        
        # Load class mapping
        class_map_path = os.path.join(self.data_dir, 're-id-clothing', 'class_map.json')
        if os.path.exists(class_map_path):
            with open(class_map_path, 'r') as f:
                self.class_map = json.load(f)
                self.num_classes = len(self.class_map) + 1  # +1 for background
        else:
            # Default class map
            self.class_map = {
                "bag": 1,
                "belt": 2,
                "boots": 3,
                "dress": 4,
                "footwear": 5,
                "headwear": 6,
                "outer": 7,
                "pants": 8,
                "scarf-tie": 9,
                "shorts": 10,
                "skirt": 11,
                "sunglasses": 12,
                "top": 13
            }
            self.num_classes = 14  # 13 classes + background
        
        # Get all image files
        for filename in os.listdir(split_dir):
            if filename.endswith('_image.png'):
                img_path = os.path.join(split_dir, filename)
                # Get corresponding mask filename
                mask_filename = filename.replace('_image.png', '_mask.png')
                mask_path = os.path.join(split_dir, mask_filename)
                
                if os.path.exists(mask_path):
                    self.samples.append({
                        'image': img_path,
                        'mask': mask_path
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        
        # Convert to PIL images for transforms
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask)
        
        # Apply transformations if any
        if self.transform:
            image_pil = self.transform(image_pil)
        else:
            # Default transforms if none provided
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_pil = transform(image_pil)
        
        # Transform mask (just resize and convert to tensor)
        mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        mask_tensor = mask_transform(mask_pil)
        
        # Convert mask to integer tensor (from float)
        mask_tensor = mask_tensor.squeeze().long()
        
        return {
            'image': image_pil,
            'mask': mask_tensor,
            'path': sample['image']
        }

class PoseDataset(Dataset):
    """Dataset for pose estimation tasks"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        self.samples = []
        
        # Determine dataset type based on directory structure
        if os.path.exists(os.path.join(data_dir, 'viton-hd')):
            self._load_viton_hd_samples()
        elif os.path.exists(os.path.join(data_dir, 'lip')):
            self._load_lip_samples()
        elif os.path.exists(os.path.join(data_dir, 'human36m')):
            self._load_human36m_samples()
        elif os.path.exists(os.path.join(data_dir, 're-id-clothing')):
            self._load_reid_clothing_samples()
        else:
            raise ValueError(f"Unsupported dataset format in {data_dir}")
    
    def _load_viton_hd_samples(self):
        """Load VITON-HD dataset samples for pose estimation"""
        split_dir = os.path.join(self.data_dir, 'viton-hd', self.split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} not found")
        
        # Get image files
        for filename in os.listdir(split_dir):
            if filename.endswith('_image.png'):
                img_path = os.path.join(split_dir, filename)
                # Get corresponding pose keypoints filename
                pose_filename = filename.replace('_image.png', '_keypoints.json')
                pose_path = os.path.join(split_dir, pose_filename)
                
                if os.path.exists(pose_path):
                    self.samples.append({
                        'image': img_path,
                        'keypoints': pose_path
                    })
    
    def _load_lip_samples(self):
        """Load LIP dataset samples for pose estimation"""
        # Implementation similar to VITON-HD
        split_dir = os.path.join(self.data_dir, 'lip', self.split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} not found")
        
        # Get image files
        for filename in os.listdir(split_dir):
            if filename.endswith('_image.png'):
                img_path = os.path.join(split_dir, filename)
                # Get corresponding keypoints filename
                keypoints_filename = filename.replace('_image.png', '_keypoints.json')
                keypoints_path = os.path.join(split_dir, keypoints_filename)
                
                if os.path.exists(keypoints_path):
                    self.samples.append({
                        'image': img_path,
                        'keypoints': keypoints_path
                    })
    
    def _load_human36m_samples(self):
        """Load Human3.6M dataset samples for pose estimation"""
        # Implementation similar to VITON-HD
        split_dir = os.path.join(self.data_dir, 'human36m', self.split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} not found")
        
        # Get image files
        for filename in os.listdir(split_dir):
            if filename.endswith('_image.png'):
                img_path = os.path.join(split_dir, filename)
                # Get corresponding keypoints filename
                keypoints_filename = filename.replace('_image.png', '_keypoints.json')
                keypoints_path = os.path.join(split_dir, keypoints_filename)
                
                if os.path.exists(keypoints_path):
                    self.samples.append({
                        'image': img_path,
                        'keypoints': keypoints_path
                    })
    
    def _load_reid_clothing_samples(self):
        """Load re-id-clothing dataset samples for pose estimation"""
        # Maps to val if split is validation
        split_name = 'val' if self.split == 'validation' else self.split
        split_dir = os.path.join(self.data_dir, 're-id-clothing', split_name)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} not found")
        
        # Check for pose keypoints files
        for filename in os.listdir(split_dir):
            if filename.endswith('_image.png'):
                img_path = os.path.join(split_dir, filename)
                
                # Get corresponding keypoints filename
                # For re-id, we might need to generate keypoints during preprocessing
                # if they're not already available
                keypoints_filename = filename.replace('_image.png', '_keypoints.json')
                keypoints_path = os.path.join(split_dir, keypoints_filename)
                
                # If keypoints file exists, add to samples
                if os.path.exists(keypoints_path):
                    self.samples.append({
                        'image': img_path,
                        'keypoints': keypoints_path
                    })
                # If no keypoints file, we could use MediaPipe to generate them
                elif os.path.exists(img_path):
                    # This would be handled in preprocessing, but mark as needing keypoints
                    self.samples.append({
                        'image': img_path,
                        'keypoints': None  # Will be generated during training
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL image for transforms
        image_pil = Image.fromarray(image)
        
        # Apply transformations if any
        if self.transform:
            image_pil = self.transform(image_pil)
        else:
            # Default transforms if none provided
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_pil = transform(image_pil)
        
        # For keypoints, we need to read from JSON and convert to tensor
        keypoints = None
        if sample['keypoints'] and os.path.exists(sample['keypoints']):
            with open(sample['keypoints'], 'r') as f:
                keypoints_data = json.load(f)
            
            # Process keypoints based on format
            # This will be specific to each dataset format
            keypoints = torch.zeros((33, 3))  # MediaPipe has 33 keypoints, each with x,y,visibility
            
            if 'keypoints' in keypoints_data:
                kp_data = keypoints_data['keypoints']
                for i, kp in enumerate(kp_data):
                    if i < 33:  # Limit to MediaPipe format
                        keypoints[i, 0] = kp[0]  # x
                        keypoints[i, 1] = kp[1]  # y
                        keypoints[i, 2] = kp[2] if len(kp) > 2 else 1.0  # visibility/confidence
        
        return {
            'image': image_pil,
            'keypoints': keypoints,
            'path': sample['image']
        } 