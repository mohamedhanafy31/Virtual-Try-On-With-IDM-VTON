import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MediaPipeSegmentationModel(nn.Module):
    """PyTorch model for human segmentation, inspired by MediaPipe architecture"""
    def __init__(self, pretrained=True):
        super(MediaPipeSegmentationModel, self).__init__()
        
        # Use MobileNetV3 as backbone (similar to MediaPipe's architecture)
        self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
        
        # Remove the classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Feature dimension from MobileNetV3
        feature_dim = 960
        
        # Decoder architecture inspired by MediaPipe
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Final layer to output segmentation mask
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Reshape feature tensor
        batch_size = features.size(0)
        features = features.view(batch_size, -1, 1, 1)
        
        # Decode features to segmentation mask
        mask = self.decoder(features)
        
        return mask


class MediaPipePoseModel(nn.Module):
    """PyTorch model for human pose estimation, inspired by MediaPipe architecture"""
    def __init__(self, num_keypoints=33, pretrained=True):
        super(MediaPipePoseModel, self).__init__()
        
        # Number of keypoints to predict (MediaPipe pose uses 33)
        self.num_keypoints = num_keypoints
        
        # Use ResNet50 as backbone (similar to MediaPipe's BlazePose architecture)
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Feature dimension from ResNet50
        feature_dim = 2048
        
        # Spatial resolution of final feature map
        self.feature_size = 7  # For 224x224 input
        
        # Pose estimation head
        self.pose_head = nn.Sequential(
            nn.Conv2d(feature_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Keypoint prediction branches
        # For each keypoint: x, y, visibility
        self.keypoint_pred = nn.Conv2d(128, num_keypoints * 3, kernel_size=1)
        
        # Heatmap branch for visualization
        self.heatmap_pred = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, kernel_size=1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features from backbone
        features = self.backbone(x)
        
        # Process features through pose head
        pose_features = self.pose_head(features)
        
        # Predict keypoints
        keypoint_outputs = self.keypoint_pred(pose_features)
        
        # Reshape to get keypoint predictions
        keypoints = keypoint_outputs.view(batch_size, self.num_keypoints, 3, self.feature_size, self.feature_size)
        
        # Apply spatial softmax to get coordinates
        softmax_x = F.softmax(keypoints[:, :, 0, :, :].view(batch_size, self.num_keypoints, -1), dim=2)
        softmax_y = F.softmax(keypoints[:, :, 1, :, :].view(batch_size, self.num_keypoints, -1), dim=2)
        
        # Create coordinate grid
        pos_x, pos_y = torch.meshgrid(torch.arange(self.feature_size), torch.arange(self.feature_size))
        pos_x = pos_x.reshape(-1).float().to(x.device)
        pos_y = pos_y.reshape(-1).float().to(x.device)
        
        # Calculate expected coordinates
        expected_x = torch.sum(softmax_x * pos_x, dim=2) / self.feature_size
        expected_y = torch.sum(softmax_y * pos_y, dim=2) / self.feature_size
        
        # Get visibility from third channel
        visibility = torch.sigmoid(keypoints[:, :, 2, 0, 0])
        
        # Stack coordinates and visibility
        keypoint_coords = torch.stack((expected_x, expected_y, visibility), dim=2)
        
        # Generate heatmaps for visualization
        heatmaps = self.heatmap_pred(pose_features)
        
        return {
            'keypoints': keypoint_coords,  # shape: [batch_size, num_keypoints, 3]
            'heatmaps': heatmaps  # shape: [batch_size, num_keypoints, height, width]
        }


class MediaPipeFaceModel(nn.Module):
    """PyTorch model for face detection and landmarks, inspired by MediaPipe architecture"""
    def __init__(self, pretrained=True):
        super(MediaPipeFaceModel, self).__init__()
        
        # Use MobileNetV2 as backbone (similar to MediaPipe's face detection)
        self.backbone = models.mobilenet_v2(pretrained=pretrained).features
        
        # Feature dimension from MobileNetV2
        feature_dim = 1280
        
        # Face detection head
        self.detection_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Output: confidence, bounding box (4 values)
            nn.Conv2d(128, 5, kernel_size=1)
        )
        
        # Face mask generation branch
        self.mask_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Output face mask
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Face detection
        detection = self.detection_head(features)
        
        # Face mask generation
        mask = self.mask_head(features)
        
        return {
            'detection': detection,  # Face bounding box and confidence
            'mask': mask  # Face segmentation mask
        }


# Custom loss functions for the models

class SegmentationLoss(nn.Module):
    """Loss function for segmentation model"""
    def __init__(self, weight=None):
        super(SegmentationLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weight)
    
    def forward(self, predictions, targets):
        # Binary cross-entropy loss
        bce = self.bce_loss(predictions, targets)
        
        # Dice loss
        dice = self.dice_loss(torch.sigmoid(predictions), targets)
        
        # Combine losses
        return bce + dice


class DiceLoss(nn.Module):
    """Dice coefficient loss for segmentation"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1.0 - dice


class KeypointLoss(nn.Module):
    """Loss function for keypoint prediction"""
    def __init__(self):
        super(KeypointLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, predictions, targets, visibility=None):
        """
        Args:
            predictions: Predicted keypoints, shape [batch_size, num_keypoints, 3]
            targets: Target keypoints, shape [batch_size, num_keypoints, 3]
            visibility: Optional visibility weights, shape [batch_size, num_keypoints]
        """
        # If visibility not provided, extract from targets
        if visibility is None:
            visibility = targets[:, :, 2]
        
        # MSE loss for keypoint coordinates (x, y)
        coord_loss = self.mse_loss(predictions[:, :, :2], targets[:, :, :2])
        
        # Apply visibility weights to focus on visible keypoints
        visibility = visibility.unsqueeze(-1).expand_as(coord_loss)
        weighted_coord_loss = (coord_loss * visibility).sum() / (visibility.sum() + 1e-8)
        
        # BCE loss for visibility prediction
        visibility_loss = F.binary_cross_entropy_with_logits(
            predictions[:, :, 2], targets[:, :, 2]
        )
        
        # Combine losses
        total_loss = weighted_coord_loss + 0.1 * visibility_loss
        
        return total_loss 