import os
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# Import custom modules
from models import (
    MediaPipeSegmentationModel, 
    MediaPipePoseModel, 
    MediaPipeFaceModel,
    SegmentationLoss,
    KeypointLoss
)
from dataset import MediaPipeDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train MediaPipe-inspired models")
    parser.add_argument("--task", type=str, required=True, choices=['segmentation', 'pose', 'face'],
                        help="Task to train")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing processed dataset")
    parser.add_argument("--output_dir", type=str, default="./trained_models", 
                        help="Directory to save trained models")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, 
                        help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of workers for data loading")
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--log_interval", type=int, default=10, 
                        help="Number of batches between logging")
    parser.add_argument("--eval_interval", type=int, default=1, 
                        help="Number of epochs between evaluations")
    parser.add_argument("--save_interval", type=int, default=5, 
                        help="Number of epochs between saving model")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize model predictions during evaluation")
    parser.add_argument("--distributed", action="store_true", 
                        help="Enable distributed training")
    parser.add_argument("--local_rank", type=int, default=0, 
                        help="Local rank for distributed training")
    parser.add_argument("--fp16", action="store_true", 
                        help="Use mixed precision training")
    return parser.parse_args()

def setup_distributed(args):
    """Initialize distributed training if enabled"""
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        print(f"Initialized process {args.rank}/{args.world_size} (local_rank: {args.local_rank})")
    else:
        args.world_size = 1
        args.rank = 0

def train_segmentation(args):
    """Train segmentation model"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    
    # Initialize model
    model = MediaPipeSegmentationModel(pretrained=True)
    
    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize DDP if enabled
    if args.distributed:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    
    # Initialize loss function and optimizer
    criterion = SegmentationLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize AMP if enabled
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # Create datasets and data loaders
    train_dataset = MediaPipeDataset(
        root_dir=args.data_dir,
        task='segmentation',
        split='train',
        augment=True
    )
    
    val_dataset = MediaPipeDataset(
        root_dir=args.data_dir,
        task='segmentation',
        split='val',
        augment=False
    )
    
    # Set up data loaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Track best validation loss
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Loaded checkpoint from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            scaler=scaler,
            task='segmentation'
        )
        
        # Log training loss
        if args.rank == 0:
            writer.add_scalar('Loss/train', train_loss, epoch)
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}")
        
        # Evaluate model
        if (epoch + 1) % args.eval_interval == 0:
            val_loss, metrics = evaluate_segmentation(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                args=args,
                visualize=(args.visualize and args.rank == 0),
                epoch=epoch
            )
            
            # Log validation metrics
            if args.rank == 0:
                writer.add_scalar('Loss/val', val_loss, epoch)
                for k, v in metrics.items():
                    writer.add_scalar(f'Metrics/{k}', v, epoch)
                
                print(f"Epoch {epoch+1}/{args.epochs}, Val Loss: {val_loss:.4f}, IOU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}")
                
                # Update LR scheduler
                scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        best_val_loss=best_val_loss,
                        args=args,
                        filename="segmentation_best.pth"
                    )
        
        # Save checkpoint periodically
        if args.rank == 0 and (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                args=args,
                filename=f"segmentation_epoch_{epoch+1}.pth"
            )
    
    # Save final model
    if args.rank == 0:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=args.epochs - 1,
            best_val_loss=best_val_loss,
            args=args,
            filename="segmentation_final.pth"
        )
        
        # Close tensorboard writer
        writer.close()
    
    print("Training complete!")

def train_pose(args):
    """Train pose estimation model"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    
    # Initialize model
    model = MediaPipePoseModel(pretrained=True)
    
    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize DDP if enabled
    if args.distributed:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    
    # Initialize loss function and optimizer
    criterion = KeypointLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize AMP if enabled
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # Create datasets and data loaders
    train_dataset = MediaPipeDataset(
        root_dir=args.data_dir,
        task='pose',
        split='train',
        augment=True
    )
    
    val_dataset = MediaPipeDataset(
        root_dir=args.data_dir,
        task='pose',
        split='val',
        augment=False
    )
    
    # Set up data loaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Track best validation loss
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Loaded checkpoint from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            scaler=scaler,
            task='pose'
        )
        
        # Log training loss
        if args.rank == 0:
            writer.add_scalar('Loss/train', train_loss, epoch)
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}")
        
        # Evaluate model
        if (epoch + 1) % args.eval_interval == 0:
            val_loss, metrics = evaluate_pose(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                args=args,
                visualize=(args.visualize and args.rank == 0),
                epoch=epoch
            )
            
            # Log validation metrics
            if args.rank == 0:
                writer.add_scalar('Loss/val', val_loss, epoch)
                for k, v in metrics.items():
                    writer.add_scalar(f'Metrics/{k}', v, epoch)
                
                print(f"Epoch {epoch+1}/{args.epochs}, Val Loss: {val_loss:.4f}, PCK: {metrics['pck']:.4f}")
                
                # Update LR scheduler
                scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        best_val_loss=best_val_loss,
                        args=args,
                        filename="pose_best.pth"
                    )
        
        # Save checkpoint periodically
        if args.rank == 0 and (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                args=args,
                filename=f"pose_epoch_{epoch+1}.pth"
            )
    
    # Save final model
    if args.rank == 0:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=args.epochs - 1,
            best_val_loss=best_val_loss,
            args=args,
            filename="pose_final.pth"
        )
        
        # Close tensorboard writer
        writer.close()
    
    print("Training complete!")

def train_epoch(model, data_loader, criterion, optimizer, device, epoch, args, scaler=None, task='segmentation'):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)
    
    pbar = tqdm(data_loader, disable=args.rank != 0)
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch['image'].to(device)
        
        if task == 'segmentation':
            targets = batch['mask'].to(device)
        elif task == 'pose':
            targets = batch['keypoints'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with AMP if enabled
        if scaler is not None:
            with torch.cuda.amp.autocast():
                if task == 'segmentation':
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                elif task == 'pose':
                    outputs = model(images)
                    loss = criterion(outputs['keypoints'], targets)
            
            # Backward pass with AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular forward pass
            if task == 'segmentation':
                outputs = model(images)
                loss = criterion(outputs, targets)
            elif task == 'pose':
                outputs = model(images)
                loss = criterion(outputs['keypoints'], targets)
            
            # Regular backward pass
            loss.backward()
            optimizer.step()
        
        # Update total loss
        total_loss += loss.item()
        
        # Update progress bar
        if args.rank == 0 and batch_idx % args.log_interval == 0:
            pbar.set_description(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")
    
    # Calculate average loss
    avg_loss = total_loss / num_batches
    
    return avg_loss

def evaluate_segmentation(model, data_loader, criterion, device, args, visualize=False, epoch=0):
    """Evaluate segmentation model on validation set"""
    model.eval()
    total_loss = 0.0
    num_batches = len(data_loader)
    
    # Metrics
    total_iou = 0.0
    total_dice = 0.0
    
    # Create directory for visualizations
    if visualize:
        vis_dir = os.path.join(args.output_dir, "visualizations", f"epoch_{epoch+1}")
        os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, disable=args.rank != 0)):
            # Move data to device
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Update total loss
            total_loss += loss.item()
            
            # Calculate metrics
            preds = torch.sigmoid(outputs) > 0.5
            
            # IOU
            intersection = (preds & (masks > 0.5)).float().sum((1, 2, 3))
            union = (preds | (masks > 0.5)).float().sum((1, 2, 3))
            iou = (intersection / (union + 1e-8)).mean().item()
            total_iou += iou
            
            # Dice coefficient
            dice = (2.0 * intersection / (preds.float().sum((1, 2, 3)) + masks.float().sum((1, 2, 3)) + 1e-8)).mean().item()
            total_dice += dice
            
            # Visualize predictions
            if visualize and batch_idx < 5:  # Visualize first 5 batches
                # Denormalize images
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
                denorm_images = images * std + mean
                
                for i in range(min(4, images.size(0))):  # Visualize up to 4 images per batch
                    # Get image, ground truth, and prediction
                    img = denorm_images[i].cpu().permute(1, 2, 0).numpy()
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    
                    mask_gt = masks[i, 0].cpu().numpy()
                    mask_pred = preds[i, 0].cpu().numpy()
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Display image
                    axes[0].imshow(img)
                    axes[0].set_title("Image")
                    axes[0].axis("off")
                    
                    # Display ground truth mask
                    axes[1].imshow(mask_gt, cmap="gray")
                    axes[1].set_title("Ground Truth")
                    axes[1].axis("off")
                    
                    # Display predicted mask
                    axes[2].imshow(mask_pred, cmap="gray")
                    axes[2].set_title(f"Prediction (IoU: {iou:.4f})")
                    axes[2].axis("off")
                    
                    # Save figure
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f"batch_{batch_idx}_sample_{i}.png"))
                    plt.close()
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches
    
    # Return metrics
    metrics = {
        'iou': avg_iou,
        'dice': avg_dice
    }
    
    return avg_loss, metrics

def evaluate_pose(model, data_loader, criterion, device, args, visualize=False, epoch=0):
    """Evaluate pose estimation model on validation set"""
    model.eval()
    total_loss = 0.0
    num_batches = len(data_loader)
    
    # Metrics
    total_pck = 0.0  # Percentage of Correct Keypoints
    
    # Create directory for visualizations
    if visualize:
        vis_dir = os.path.join(args.output_dir, "visualizations", f"epoch_{epoch+1}")
        os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, disable=args.rank != 0)):
            # Move data to device
            images = batch['image'].to(device)
            keypoints_gt = batch['keypoints'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs['keypoints'], keypoints_gt)
            
            # Update total loss
            total_loss += loss.item()
            
            # Calculate PCK metric (percentage of correct keypoints)
            # A keypoint is correct if its distance from the ground truth is less than a threshold
            # (usually a fraction of the head or torso size)
            
            # For simplicity, use a fixed threshold of 0.1 (10% of the image size)
            threshold = 0.1
            
            # Calculate distances between predicted and ground truth keypoints
            pred_keypoints = outputs['keypoints']
            distances = torch.sqrt(
                (pred_keypoints[:, :, 0] - keypoints_gt[:, :, 0])**2 + 
                (pred_keypoints[:, :, 1] - keypoints_gt[:, :, 1])**2
            )
            
            # Get visibility mask from ground truth
            visibility = keypoints_gt[:, :, 2] > 0.5
            
            # Calculate PCK: for each image in the batch, compute the percentage of keypoints
            # with distance below threshold
            correct_keypoints = (distances < threshold) & visibility
            pck = correct_keypoints.float().sum(dim=1) / (visibility.float().sum(dim=1) + 1e-8)
            
            # Average PCK over the batch
            batch_pck = pck.mean().item()
            total_pck += batch_pck
            
            # Visualize predictions
            if visualize and batch_idx < 5:  # Visualize first 5 batches
                # Denormalize images
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
                denorm_images = images * std + mean
                
                for i in range(min(4, images.size(0))):  # Visualize up to 4 images per batch
                    # Get image and keypoints
                    img = denorm_images[i].cpu().permute(1, 2, 0).numpy()
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    
                    # Get keypoints (convert normalized coordinates to pixel coordinates)
                    h, w = img.shape[:2]
                    kp_gt = keypoints_gt[i].cpu().numpy()
                    kp_pred = pred_keypoints[i].cpu().numpy()
                    
                    # Convert to pixel coordinates
                    kp_gt_px = np.zeros_like(kp_gt)
                    kp_pred_px = np.zeros_like(kp_pred)
                    
                    kp_gt_px[:, 0] = kp_gt[:, 0] * w
                    kp_gt_px[:, 1] = kp_gt[:, 1] * h
                    kp_gt_px[:, 2] = kp_gt[:, 2]
                    
                    kp_pred_px[:, 0] = kp_pred[:, 0] * w
                    kp_pred_px[:, 1] = kp_pred[:, 1] * h
                    kp_pred_px[:, 2] = kp_pred[:, 2]
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Display image with ground truth keypoints
                    axes[0].imshow(img)
                    for j in range(kp_gt.shape[0]):
                        if kp_gt[j, 2] > 0.5:  # Only show visible keypoints
                            axes[0].scatter(kp_gt_px[j, 0], kp_gt_px[j, 1], c='g', s=20)
                    axes[0].set_title("Ground Truth")
                    axes[0].axis("off")
                    
                    # Display image with predicted keypoints
                    axes[1].imshow(img)
                    for j in range(kp_pred.shape[0]):
                        if kp_gt[j, 2] > 0.5:  # Only show keypoints that should be visible
                            color = 'g' if correct_keypoints[i, j] else 'r'
                            axes[1].scatter(kp_pred_px[j, 0], kp_pred_px[j, 1], c=color, s=20)
                    axes[1].set_title(f"Prediction (PCK: {pck[i]:.4f})")
                    axes[1].axis("off")
                    
                    # Save figure
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f"batch_{batch_idx}_sample_{i}.png"))
                    plt.close()
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_pck = total_pck / num_batches
    
    # Return metrics
    metrics = {
        'pck': avg_pck
    }
    
    return avg_loss, metrics

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, args, filename):
    """Save model checkpoint"""
    # Create directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': args
    }
    
    # Save checkpoint
    torch.save(checkpoint, os.path.join(args.output_dir, filename))
    print(f"Saved checkpoint to {os.path.join(args.output_dir, filename)}")

def export_model(model, output_path, task, device="cpu"):
    """Export trained model to ONNX format"""
    model.eval()
    
    # Create dummy input
    if task == 'segmentation' or task == 'face':
        dummy_input = torch.randn(1, 3, 256, 192).to(device)
    elif task == 'pose':
        dummy_input = torch.randn(1, 3, 384, 288).to(device)
    
    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Exported model to {output_path}")

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up distributed training if enabled
    setup_distributed(args)
    
    # Select and train the model based on the task
    if args.task == 'segmentation':
        train_segmentation(args)
    elif args.task == 'pose':
        train_pose(args)
    elif args.task == 'face':
        # Face model training would be similar to segmentation and pose combined
        print("Face model training not implemented yet")
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    # Clean up distributed process group
    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 