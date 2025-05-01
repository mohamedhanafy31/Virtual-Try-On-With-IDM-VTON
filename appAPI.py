import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import os
import logging
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer
from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
import torchvision.transforms as transforms
import cv2
import mediapipe as mp
from huggingface_hub import hf_hub_download
import warnings
import sys
import torch.nn.functional as F
import glob
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from cloth_segmentation.networks.u2net import U2NET
import segmentation_models_pytorch as smp

# Suppress deprecated warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Prioritize custom src directory
sys.path.insert(0, 'E://Meta_VR/VirtualTryOn/NEW-IDM-VTON/src')

# Enable debug logging, reduce PIL noise
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('PIL').setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="IDM-VTON Inference Script for Virtual Try-On")
    parser.add_argument("--human_image", type=str, required=True, help="Path to the human image")
    parser.add_argument("--cloth_image", type=str, required=True, help="Path to the cloth image")
    parser.add_argument("--garment_caption", type=str, default="Virtual Try-On Result", help="Caption for the garment")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the output image")
    parser.add_argument("--pretrained_model_path", type=str, default="yisol\IDM-VTON", help="Path to pretrained model weights")
    parser.add_argument("--width", type=int, default=768, help="Output image width")
    parser.add_argument("--height", type=int, default=1024, help="Output image height")
    parser.add_argument("--num_inference_steps", type=int, default=75, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="CFG scale for guidance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save intermediate results")
    return parser.parse_args()

def download_weights(model_path, repo_id="yisol/IDM-VTON"):
    print(f"Checking for model weights in {model_path}...")
    required_subfolders = ["unet", "vae", "image_encoder", "unet_encoder", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler"]
    weights_files = {
        "unet": ["diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"],
        "vae": ["diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"],
        "image_encoder": ["model.bin", "model.safetensors"],
        "unet_encoder": ["diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"],
        "text_encoder": ["model.bin", "model.safetensors"],
        "text_encoder_2": ["model.bin", "model.safetensors"],
        "scheduler": ["scheduler_config.json"],
        "tokenizer": ["tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"],
        "tokenizer_2": ["tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]
    }

    os.makedirs(model_path, exist_ok=True)
    for subfolder in required_subfolders:
        os.makedirs(os.path.join(model_path, subfolder), exist_ok=True)

    missing_files = []
    for subfolder, filenames in weights_files.items():
        found = False
        for filename in filenames:
            file_path = os.path.join(model_path, subfolder, filename)
            if os.path.exists(file_path):
                found = True
                break
        if not found:
            missing_files.append(f"{subfolder}/{filenames[0]}")
            if subfolder not in ["unet", "vae", "image_encoder", "unet_encoder", "text_encoder", "text_encoder_2"]:
                for filename in filenames:
                    try:
                        print(f"Downloading {filename} for {subfolder} from {repo_id}...")
                        hf_hub_download(
                            repo_id=repo_id,
                            subfolder=subfolder,
                            filename=filename,
                            local_dir=model_path,
                            local_dir_use_symlinks=False
                        )
                        print(f"Downloaded {filename} to {file_path}")
                        found = True
                        break
                    except Exception as e:
                        print(f"Failed to download {filename} for {subfolder}: {e}")

    model_index_path = os.path.join(model_path, "model_index.json")
    if not os.path.exists(model_index_path):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename="model_index.json",
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            print(f"Downloaded model_index.json to {model_index_path}")
        except Exception as e:
            print(f"Failed to download model_index.json: {e}")
            missing_files.append("model_index.json")

    if missing_files:
        print(f"ERROR: Missing required files: {missing_files}")
        print("Please manually download them from https://huggingface.co/yisol/IDM-VTON or other sources and place them in:")
        for subfolder, filenames in weights_files.items():
            for filename in filenames:
                print(f"  {model_path}\\{subfolder}\\{filename}")
        print(f"  {model_path}\\model_index.json")
        print("Note: Weights for unet, vae, etc., may require manual download from the repository or extraction from a single weights file (~10GB).")
        exit(1)

    print("Weight check complete. All required files are in place.")

def preprocess_image(image_path, size=(768, 1024), normalize=True):
    try:
        image = Image.open(image_path).convert("RGB").resize(size)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) if normalize else transforms.Lambda(lambda x: x),
        ])
        return transform(image)
    except Exception as e:
        raise ValueError(f"Error preprocessing image {image_path}: {e}")

def adjust_cloth_lighting(cloth_image, human_image):
    cloth_np = cloth_image.squeeze().permute(1, 2, 0).cpu().numpy()
    human_np = human_image.squeeze().permute(1, 2, 0).cpu().numpy()
    human_brightness = np.mean(human_np[int(human_np.shape[0]*0.3):int(human_np.shape[0]*0.6), :, :])
    cloth_brightness = np.mean(cloth_np)
    brightness_factor = human_brightness / (cloth_brightness + 1e-6)
    cloth_adjusted = np.clip(cloth_np * brightness_factor, 0, 1)
    return torch.tensor(cloth_adjusted).permute(2, 0, 1).unsqueeze(0)

# ClothDataset class (adapted from first script)
class ClothDataset(Dataset):
    def __init__(self, img_path, transform=None, inference_size=(256, 256)):
        self.img_path = img_path
        self.transform = transform
        self.inference_size = inference_size
        if not os.path.exists(img_path):
            raise ValueError(f"Image not found at {img_path}")
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        img_name = os.path.basename(self.img_path)
        image = cv2.imread(self.img_path)
        if image is None:
            print(f"Error: Failed to load image {img_name}")
            return torch.zeros(3, self.inference_size[0], self.inference_size[1]), img_name, (self.inference_size[0], self.inference_size[1])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, img_name, (orig_height, orig_width)

# Color palette for visualization (from first script)
def get_palette(num_cls):
    palette = [0] * (num_cls * 3)
    for j in range(num_cls):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette

def generate_clothing_segmentation_mask(image_path, size=(768, 1024), latent_size=(128, 96)):
    """Generate a segmentation mask focused on clothing items using smp.Unet."""
    try:
        # Configuration
        inference_size = (256, 256)
        num_classes = 4
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_checkpoint", "best_model (4).pth")
        do_palette = True

        # Define transform
        test_transform = A.Compose([
            A.Resize(height=inference_size[0], width=inference_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        # Initialize dataset
        dataset = ClothDataset(img_path=image_path, transform=test_transform, inference_size=inference_size)

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = smp.Unet(
            encoder_name="resnet34",
            in_channels=3,
            classes=num_classes,
            encoder_weights=None
        ).to(device)

        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=True)
            print("U-Net weights loaded successfully for segmentation")
        except Exception as e:
            raise ValueError(f"Error loading U-Net weights: {e}")

        model.eval()

        # Get color palette
        palette = get_palette(num_classes)

        # Inference
        img_tensor, img_name, orig_size = dataset[0]
        if img_tensor is None:
            print(f"Failed to load image {img_name}")
            return torch.ones((1, 1, latent_size[0], latent_size[1]), device=device) * 0.5

        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            try:
                output_tensor = model(img_tensor)
                output_tensor = F.log_softmax(output_tensor, dim=1)
                output_tensor = F.interpolate(
                    output_tensor,
                    size=orig_size,
                    mode='nearest'
                )
                pred = output_tensor.argmax(dim=1).cpu().numpy()[0]
            except Exception as e:
                print(f"Error predicting for {img_name}: {e}")
                pred = np.zeros(orig_size, dtype=np.uint8)

        # Post-process to create binary mask
        mask_np = np.zeros_like(pred, dtype=np.uint8)
        mask_np[pred > 0] = 1  # Combine all non-background classes (1, 2, 3)

        # Apply size constraints
        mask_np[:int(orig_size[0]*0.2), :] = 0  # Exclude head
        mask_np[int(orig_size[0]*0.75):, :] = 0  # Exclude lower body

        # Convert to tensor
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)

        # Resize to latent space dimensions
        if latent_size:
            mask_tensor = F.interpolate(mask_tensor, size=latent_size, mode='nearest')

        # Save debug visualization if needed
        if args.debug:
            output_img = Image.fromarray(pred.astype("uint8"), mode="L")
            if do_palette:
                output_img.putpalette(palette)
                output_img = output_img.convert("RGB")
            debug_path = os.path.join(args.output_dir, "clothing_mask_debug.png")
            output_img.save(debug_path)
            print(f"Debug segmentation mask saved to {debug_path}")

        return mask_tensor.to(device)

    except Exception as e:
        print(f"Error in clothing segmentation: {e}")
        return torch.ones((1, 1, latent_size[0], latent_size[1]), device=device) * 0.5

def generate_pose_image(image_path, image_size=(768, 1024)):
    try:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose()
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        pose_map = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                pose_map,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=5, circle_radius=5),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=5)
            )
        pose_map = transforms.ToTensor()(pose_map)
        return pose_map.unsqueeze(0)
    except Exception as e:
        raise RuntimeError(f"Error generating pose image: {e}")

def add_caption(image, caption):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    text_bbox = font.getbbox(caption)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = (10, image.height - text_height - 10)
    draw.text(position, caption, fill=(255, 255, 255), font=font)
    return image

def generate_face_mask(image_path, size=(768, 1024), latent_size=None):
    """Generate a mask for the face region to preserve original face"""
    try:
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(min_detection_confidence=0.3) as face_detection:
            image = cv2.imread(image_path)
            image = cv2.resize(image, size)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)
            
            mask = np.zeros((size[1], size[0]), dtype=np.uint8)
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w = image.shape[:2]
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    width, height = int(bbox.width * w), int(bbox.height * h)
                    
                    y_expand = int(height * 0.5)
                    x_expand = int(width * 0.2)
                    
                    y1 = max(0, y - y_expand)
                    y2 = min(h, y + height + int(height * 0.1))
                    x1 = max(0, x - x_expand)
                    x2 = min(w, x + width + x_expand)
                    
                    mask[y1:y2, x1:x2] = 255
                    
                    neck_height = int(height * 0.3)
                    neck_width = int(width * 0.7)
                    neck_x1 = x + (width - neck_width) // 2
                    neck_x2 = neck_x1 + neck_width
                    neck_y1 = y + height
                    neck_y2 = min(h, y + height + neck_height)
                    
                    mask[neck_y1:neck_y2, neck_x1:neck_x2] = 255
            
            if latent_size is not None:
                mask = cv2.resize(mask, latent_size[::-1], interpolation=cv2.INTER_LINEAR)
            
            mask = cv2.GaussianBlur(mask, (9, 9), 3)
            
            return mask
    except Exception as e:
        print(f"Error generating face mask: {e}")
        if latent_size is not None:
            return np.zeros(latent_size[::-1], dtype=np.uint8)
        return np.zeros((size[1], size[0]), dtype=np.uint8)

def preserve_original_face(generated_image, original_image, face_mask):
    """Blend the original face with the generated image"""
    try:
        gen_np = np.array(generated_image)
        orig_np = np.array(original_image)
        
        if gen_np.shape != orig_np.shape:
            orig_np = cv2.resize(orig_np, (gen_np.shape[1], gen_np.shape[0]))
        
        mask_norm = face_mask.astype(float) / 255.0
        
        if len(mask_norm.shape) == 2:
            mask_norm = np.stack([mask_norm] * 3, axis=2)
        
        blended = gen_np * (1 - mask_norm) + orig_np * mask_norm
        
        return Image.fromarray(blended.astype(np.uint8))
    except Exception as e:
        print(f"Error preserving face: {e}")
        return generated_image

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    weight_dtype = torch.float16

    download_weights(args.pretrained_model_path)

    if not os.path.exists(args.human_image):
        print(f"Human image not found: {args.human_image}")
        exit(1)
    if not os.path.exists(args.cloth_image):
        print(f"Cloth image not found: {args.cloth_image}")
        exit(1)
    try:
        Image.open(args.human_image)
        Image.open(args.cloth_image)
        print("Input images verified.")
    except Exception as e:
        print(f"Error opening images: {e}")
        exit(1)

    print("Loading model components...")
    try:
        scheduler_path = os.path.join(args.pretrained_model_path, "scheduler")
        if not os.path.exists(os.path.join(scheduler_path, "scheduler_config.json")):
            raise FileNotFoundError(f"Scheduler config not found at {scheduler_path}")
        noise_scheduler = DDPMScheduler.from_pretrained(scheduler_path, local_files_only=True)
        print("DDPMScheduler loaded.")

        vae_path = os.path.join(args.pretrained_model_path, "vae")
        if not (os.path.exists(os.path.join(vae_path, "diffusion_pytorch_model.bin")) or 
                os.path.exists(os.path.join(vae_path, "diffusion_pytorch_model.safetensors"))):
            raise FileNotFoundError(f"VAE weights not found at {vae_path}")
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=weight_dtype, local_files_only=True)
        print(f"AutoencoderKL loaded from {vae_path}.")

        unet_path = os.path.join(args.pretrained_model_path, "unet")
        if not (os.path.exists(os.path.join(unet_path, "diffusion_pytorch_model.bin")) or 
                os.path.exists(os.path.join(unet_path, "diffusion_pytorch_model.safetensors"))):
            raise FileNotFoundError(f"UNet weights not found at {unet_path}")
        try:
            from peft import set_peft_backend
            set_peft_backend(True)
            print("PEFT backend enabled.")
        except ImportError:
            print("PEFT backend not available. Proceeding without PEFT.")
        unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=weight_dtype, local_files_only=True)
        print(f"UNet2DConditionModel (unet) loaded from {unet_path}.")

        image_encoder_path = os.path.join(args.pretrained_model_path, "image_encoder")
        if not (os.path.exists(os.path.join(image_encoder_path, "model.bin")) or 
                os.path.exists(os.path.join(image_encoder_path, "model.safetensors"))):
            raise FileNotFoundError(f"Image encoder weights not found at {image_encoder_path}")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, torch_dtype=weight_dtype, local_files_only=True)
        print(f"CLIPVisionModelWithProjection loaded from {image_encoder_path}.")

        unet_encoder_path = os.path.join(args.pretrained_model_path, "unet_encoder")
        if not (os.path.exists(os.path.join(unet_encoder_path, "diffusion_pytorch_model.bin")) or 
                os.path.exists(os.path.join(unet_encoder_path, "diffusion_pytorch_model.safetensors"))):
            raise FileNotFoundError(f"UNet encoder weights not found at {unet_encoder_path}")
        unet_encoder = UNet2DConditionModel_ref.from_pretrained(unet_encoder_path, torch_dtype=weight_dtype, local_files_only=True)
        print(f"UNet2DConditionModel_ref (unet_encoder) loaded from {unet_encoder_path}.")

        text_encoder_path = os.path.join(args.pretrained_model_path, "text_encoder")
        if not (os.path.exists(os.path.join(text_encoder_path, "model.bin")) or 
                os.path.exists(os.path.join(text_encoder_path, "model.safetensors"))):
            raise FileNotFoundError(f"Text encoder weights not found at {text_encoder_path}")
        text_encoder_one = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=weight_dtype, local_files_only=True)
        print(f"CLIPTextModel (text_encoder) loaded from {text_encoder_path}.")

        text_encoder_2_path = os.path.join(args.pretrained_model_path, "text_encoder_2")
        if not (os.path.exists(os.path.join(text_encoder_2_path, "model.bin")) or 
                os.path.exists(os.path.join(text_encoder_2_path, "model.safetensors"))):
            raise FileNotFoundError(f"Text encoder 2 weights not found at {text_encoder_2_path}")
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(text_encoder_2_path, torch_dtype=weight_dtype, local_files_only=True)
        print(f"CLIPTextModelWithProjection (text_encoder_2) loaded from {text_encoder_2_path}.")

        tokenizer_path = os.path.join(args.pretrained_model_path, "tokenizer")
        if not all(os.path.exists(os.path.join(tokenizer_path, f)) for f in ["tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]):
            raise FileNotFoundError(f"Tokenizer files not found at {tokenizer_path}")
        tokenizer_one = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, local_files_only=True)
        print(f"Tokenizer (tokenizer) loaded from {tokenizer_path}.")

        tokenizer_2_path = os.path.join(args.pretrained_model_path, "tokenizer_2")
        if not all(os.path.exists(os.path.join(tokenizer_2_path, f)) for f in ["tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]):
            raise FileNotFoundError(f"Tokenizer 2 files not found at {tokenizer_2_path}")
        tokenizer_two = AutoTokenizer.from_pretrained(tokenizer_2_path, use_fast=False, local_files_only=True)
        print(f"Tokenizer (tokenizer_2) loaded from {tokenizer_2_path}.")
    except Exception as e:
        print(f"Error loading model components: {e}")
        raise

    print("Model components loaded.")

    print("Initializing pipeline...")
    try:
        pipe = TryonPipeline(
            vae=vae,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            unet=unet,
            unet_encoder=unet_encoder,
            scheduler=noise_scheduler,
            image_encoder=image_encoder,
            feature_extractor=CLIPImageProcessor(),
            requires_aesthetics_score=False,
            force_zeros_for_empty_prompt=True,
        ).to(device)
        pipe.enable_sequential_cpu_offload()
        print("Pipeline initialized.")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        raise

    print("Preprocessing images...")
    try:
        human_image = preprocess_image(args.human_image, size=(args.width, args.height)).unsqueeze(0).to(device, weight_dtype)
        original_human_pil = Image.open(args.human_image).convert("RGB").resize((args.width, args.height))
        face_mask = generate_face_mask(args.human_image, size=(args.width, args.height))
        print("Face mask generated for identity preservation.")
        cloth_image = preprocess_image(args.cloth_image, size=(args.width, args.height)).unsqueeze(0).to(device, weight_dtype)
        cloth_image_pure = preprocess_image(args.cloth_image, size=(args.width, args.height), normalize=False).unsqueeze(0).to(device, weight_dtype)
        cloth_image_pure = adjust_cloth_lighting(cloth_image_pure, human_image).to(device, weight_dtype)
        print("Images preprocessed.")
    except Exception as e:
        print(f"Error preprocessing images: {e}")
        raise

    print("Generating improved clothing segmentation mask and pose...")
    try:
        latent_height_vae, latent_width_vae = args.height // 8, args.width // 8
        mask = generate_clothing_segmentation_mask(
            args.human_image, 
            size=(args.width, args.height), 
            latent_size=(latent_height_vae, latent_width_vae)
        ).to(device, weight_dtype)
        if args.debug:
            mask_vis = mask.squeeze().cpu().numpy() * 255
            cv2.imwrite(os.path.join(args.output_dir, "clothing_mask_debug.png"), mask_vis)
            print("Debug clothing mask saved.")
        pose_img = generate_pose_image(args.human_image, image_size=(args.width, args.height)).to(device, weight_dtype)
        print("Mask and pose generated.")
    except Exception as e:
        print(f"Error generating mask and pose: {e}")
        raise

    print("Starting CLIP preprocessing...")
    try:
        clip_processor = CLIPImageProcessor()
        cloth_clip = clip_processor(images=Image.open(args.cloth_image).resize((224, 224)), return_tensors="pt").pixel_values.to(device, weight_dtype)
        print("CLIP preprocessing completed.")
    except Exception as e:
        print(f"Error in CLIP preprocessing: {e}")
        raise

    print("Determining gender for prompt...")
    try:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
            image = cv2.imread(args.human_image)
            image = cv2.resize(image, (args.width, args.height))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            
            gender_term = "person"
            
            if results.multi_face_landmarks:
                lower_path = args.human_image.lower()
                if "man" in lower_path or "male" in lower_path or "boy" in lower_path:
                    gender_term = "man"
                elif "woman" in lower_path or "female" in lower_path or "girl" in lower_path:
                    gender_term = "woman"
                else:
                    print("Note: Unable to automatically determine gender from image. Using generic term in prompt.")
    except Exception as e:
        print(f"Error in gender detection: {e}")
        gender_term = "person"
        
    print(f"Using gender term: {gender_term}")

    # Define prompts
    prompt = f"male wearing a casual black polo t-shirt"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blazer, jacket, face change, gender change, blurred face, different face, distorted face, misaligned collar, unnatural sleeves"
    
    if gender_term == "man":
        negative_prompt += ", female face, woman, girl, feminine features, makeup, lipstick"
    elif gender_term == "woman":
        negative_prompt += ", male face, man, boy, masculine features, beard, mustache"
    
    cloth_prompt = args.garment_caption

    print("Encoding prompts...")
    try:
        with torch.no_grad():
            (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
                prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt
            )
            (prompt_embeds_cloth, _, _, _) = pipe.encode_prompt(
                cloth_prompt, num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=negative_prompt
            )
        print("Prompts encoded.")
    except Exception as e:
        print(f"Error encoding prompts: {e}")
        raise

    generator = torch.Generator(device).manual_seed(args.seed) if args.seed else None

    print("Starting inference...")
    try:
        with torch.cuda.amp.autocast() if device.type == "cuda" else torch.no_grad():
            images = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_img,
                text_embeds_cloth=prompt_embeds_cloth,
                cloth=cloth_image_pure,
                mask_image=mask,
                image=(human_image + 1.0) / 2.0,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                ip_adapter_image=cloth_clip,
            )[0]
        print("Inference completed.")
    except Exception as e:
        print(f"Error during inference: {e}")
        raise

    if images is None or len(images) == 0:
        print("No images generated by the pipeline.")
        return None

    print("Saving result...")
    try:
        result_image = preserve_original_face(images[0], original_human_pil, face_mask)
        result_image = add_caption(result_image, caption=args.garment_caption)
        os.makedirs(args.output_dir, exist_ok=True)
        existing_files = glob.glob(os.path.join(args.output_dir, "tryon_result*.png"))
        file_num = len(existing_files) + 1
        output_path = os.path.join(args.output_dir, f"tryon_result{file_num}.png")
        result_image.save(output_path)
        print(f"Try-on image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
        raise

    return result_image

if __name__ == "__main__":
    args = parse_args()
    result = run_inference(args)