import logging
import os
import time
from enum import Enum
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from diffusers import DPMSolverMultistepScheduler, AutoencoderKL
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
)
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
import apply_net
from utils_mask import get_mask_location

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device setup
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# Constants
PROCESS_SIZE = (720, 960)  # Target size for processing (height, width)
OUTPUT_SIZE = (960, 720)   # Output size after upscaling
BASE_MODEL_PATH = "yisol/IDM-VTON"  # Path to pretrained models
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Suppress symlink warnings

# Enum for resize methods
class ResizeMethod(Enum):
    STANDARD = "standard"
    ESRGAN = "esrgan"
    SEAM_CARVE = "seam_carve"

# Pydantic model for API request
class TryOnRequest(BaseModel):
    resize_method: str = ResizeMethod.STANDARD.value
    upscale_method: str = ResizeMethod.STANDARD.value
    garment_description: str = "a garment"

# Cache for human parsing results
parsing_cache: Dict[str, tuple] = {}

# Initialize FastAPI app
app = FastAPI(title="Virtual Try-On API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ESRGAN model
esrgan_model: Optional[RealESRGANer] = None

def initialize_esrgan() -> RealESRGANer:
    try:
        logger.info("Initializing RealESRGAN model...")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        weights_path = "weights/RealESRGAN_x4plus.pth"  # Adjusted path to weights directory
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"ESRGAN model weights not found at {weights_path}. "
                "Please download from https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth "
                "and place it in the weights/ directory."
            )
        
        try:
            with open(weights_path, 'rb') as f:
                pass  # Test read access
        except PermissionError as e:
            logger.error(f"Permission denied accessing {weights_path}: {e}")
            raise HTTPException(status_code=500, detail="Permission denied for model weights")
        
        esrgan = RealESRGANer(
            scale=4,
            model_path=weights_path,
            model=model,
            device=DEVICE,
            half=True if 'cuda' in DEVICE.type else False
        )
        logger.info("RealESRGAN model initialized")
        return esrgan
    except FileNotFoundError as e:
        logger.error(f"ESRGAN weights not found: {e}")
        raise HTTPException(status_code=500, detail="ESRGAN weights not found")
    except PermissionError as e:
        logger.error(f"Permission denied for ESRGAN weights: {e}")
        raise HTTPException(status_code=500, detail="Permission denied for model weights")
    except Exception as e:
        logger.error(f"Failed to initialize ESRGAN: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize ESRGAN")

# Image resize functions with aspect ratio preservation
def resize_with_aspect_ratio(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Resize image while preserving aspect ratio, padding if necessary."""
    img_w, img_h = image.size
    target_h, target_w = target_size
    aspect = img_w / img_h
    target_aspect = target_w / target_h

    if aspect > target_aspect:
        new_w = target_w
        new_h = int(new_w / aspect)
    else:
        new_h = target_h
        new_w = int(new_h * aspect)

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(resized, (paste_x, paste_y))
    return new_img

def standard_resize(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Resize image using standard method with aspect ratio preservation."""
    return resize_with_aspect_ratio(image, target_size)

def esrgan_resize(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Resize image using ESRGAN with aspect ratio preservation."""
    global esrgan_model
    if esrgan_model is None:
        esrgan_model = initialize_esrgan()
    intermediate = resize_with_aspect_ratio(image, (target_size[1] // 4, target_size[0] // 4))
    img_array = np.array(intermediate).astype(np.uint8)
    upscaled, _ = esrgan_model.enhance(img_array, outscale=4)
    return Image.fromarray(upscaled).resize(target_size, Image.Resampling.LANCZOS)

def seam_carve_resize(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Resize image using seam carving with aspect ratio preservation."""
    img_array = np.array(image).astype(np.uint8)
    resized = cv2.resize(img_array, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized)

def resize_if_needed(
    pil_img: Image.Image, 
    target_size: Tuple[int, int], 
    method: ResizeMethod = ResizeMethod.STANDARD,
    label: str = ""
) -> Image.Image:
    original_width, original_height = pil_img.size
    target_width, target_height = target_size
    
    logger.info(f"Resizing {label} image: {original_width}x{original_height} -> {target_width}x{target_height} using {method}")
    
    if original_width == target_width and original_height == target_height:
        return pil_img.convert("RGB")
        
    if method == ResizeMethod.ESRGAN:
        return esrgan_resize(pil_img, target_size)
    elif method == ResizeMethod.SEAM_CARVE:
        return seam_carve_resize(pil_img, target_size)
    else:
        return standard_resize(pil_img, target_size)

# Final upscale function
def final_upscale(img: Image.Image, output_size: Tuple[int, int], method: ResizeMethod) -> Image.Image:
    if method == ResizeMethod.ESRGAN:
        return esrgan_resize(img, output_size)
    elif method == ResizeMethod.SEAM_CARVE:
        return seam_carve_resize(img, output_size)
    else:
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, torch.float16)
        upscaled_tensor = torch.nn.functional.interpolate(
            img_tensor, 
            size=output_size[::-1], 
            mode='bicubic', 
            align_corners=False
        )
        # Apply mild sharpening to enhance edges
        upscaled_np = upscaled_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0
        upscaled_np = upscaled_np.astype(np.uint8)
        kernel = np.array([[-0.5, -1, -0.5], [-1, 7, -1], [-0.5, -1, -0.5]]) / 2
        sharpened = cv2.filter2D(upscaled_np, -1, kernel)
        return Image.fromarray(sharpened)

# Load pretrained models
def load_models(base_path: str) -> dict:
    """Load all necessary pretrained models."""
    try:
        models = {
            'unet': UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=torch.float16),
            'vae': AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=torch.float16),
            'text_encoder_one': CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=torch.float16),
            'text_encoder_two': CLIPTextModelWithProjection.from_pretrained(base_path, subfolder="text_encoder_2", torch_dtype=torch.float16),
            'image_encoder': CLIPVisionModelWithProjection.from_pretrained(base_path, subfolder="image_encoder", torch_dtype=torch.float16),
            'unet_encoder': UNet2DConditionModel_ref.from_pretrained(base_path, subfolder="unet_encoder", torch_dtype=torch.float16),
        }
        
        for name, model in models.items():
            model.to(DEVICE)
            model.requires_grad_(False)
            logger.info(f"Loaded {name} to {DEVICE}")
        
        scheduler = DPMSolverMultistepScheduler.from_pretrained(base_path, subfolder="scheduler")
        tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", use_fast=False)
        tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", use_fast=False)
        
        return models, scheduler, tokenizer_one, tokenizer_two
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise HTTPException(status_code=500, detail="Failed to load pretrained models")

try:
    models, scheduler, tokenizer_one, tokenizer_two = load_models(BASE_MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise

# Initialize pipeline and preprocessors
try:
    pipeline = TryonPipeline.from_pretrained(
        BASE_MODEL_PATH,
        unet=models['unet'],
        vae=models['vae'],
        feature_extractor=CLIPImageProcessor(),
        text_encoder=models['text_encoder_one'],
        text_encoder_2=models['text_encoder_two'],
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=scheduler,
        image_encoder=models['image_encoder'],
        torch_dtype=torch.float16,
    ).to(DEVICE)
    pipeline.unet_encoder = models['unet_encoder']
    logger.info("TryonPipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize TryonPipeline: {e}")
    raise

try:
    parsing_model = Parsing(0)
    pose_model = OpenPose(0)
    pose_model.preprocessor.body_estimation.model.to(DEVICE)
    logger.info("Preprocessors initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize preprocessors: {e}")
    raise

# Image transformation
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Core processing functions
def get_image_hash(image: Image.Image) -> str:
    """Generate a hash for caching based on image bytes."""
    return hash(str(image.tobytes()))

def process_human_image(human_img: Image.Image) -> tuple:
    """Process human image for parsing and pose estimation."""
    human_hash = get_image_hash(human_img)
    if human_hash in parsing_cache:
        logger.info("Using cached human parsing results")
        keypoints, model_parse = parsing_cache[human_hash]
    else:
        logger.info("Processing human pose and parsing")
        openpose_input = human_img.resize((384, 512), Image.LANCZOS)
        keypoints = pose_model(openpose_input)
        model_parse, _ = parsing_model(human_img)
        parsing_cache[human_hash] = (keypoints, model_parse)
    return keypoints, model_parse

def generate_mask(model_parse, keypoints) -> Image.Image:
    """Generate and refine mask for upper body."""
    mask, _ = get_mask_location('hd', "upper_body", model_parse, keypoints)
    mask = resize_if_needed(mask, PROCESS_SIZE, method=ResizeMethod.STANDARD, label="Mask")
    mask.save("debug_mask.png")  # Debug: Save mask for inspection
    mask_np = np.array(mask)
    kernel = np.ones((7, 7), np.uint8)
    mask_np = cv2.erode(mask_np, kernel, iterations=10)
    mask = Image.fromarray(mask_np)
    return mask

def apply_densepose(human_img: Image.Image) -> Image.Image:
    """Apply DensePose to human image."""
    logger.info("Applying DensePose")
    human_img_arg = _apply_exif_orientation(human_img)
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
    args = apply_net.create_argument_parser().parse_args((
        'show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
        './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v',
        '--opts', 'MODEL.DEVICE', 'cuda'
    ))
    human_img_arg = Image.fromarray(human_img_arg).resize((256, 192), Image.LANCZOS)
    pose_img = args.func(args, np.array(human_img_arg))
    pose_img = Image.fromarray(pose_img[:, :, ::-1]).resize(PROCESS_SIZE, Image.LANCZOS)
    return pose_img

def run_diffusion_pipeline(
    human_img: Image.Image,
    garm_img: Image.Image,
    mask: Image.Image,
    pose_img: Image.Image,
    garment_des: str,
) -> list[Image.Image]:
    """Run diffusion pipeline for virtual try-on."""
    logger.info("Running virtual try-on inference")
    with torch.no_grad(), torch.cuda.amp.autocast():
        prompt = f"model is wearing {garment_des}"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        
        logger.info("Encoding prompt")
        prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = pipeline.encode_prompt(
            prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt
        )
        prompt_c = f"a photo of {garment_des}"
        prompt_embeds_c, _, _, _ = pipeline.encode_prompt(
            prompt_c, num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=negative_prompt
        )

        # Transform pose and garment images
        pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(DEVICE, torch.float16)
        garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(DEVICE, torch.float16)

        # Transform mask to ensure it's a single-channel tensor
        mask = mask.convert("L")  # Convert to grayscale
        mask_tensor = transforms.ToTensor()(mask).to(DEVICE, torch.float16)  # Shape: (1, 720, 960)
        mask_tensor = mask_tensor.unsqueeze(0)  # Shape: (1, 1, 720, 960)

        # Transform human image (ensure it's a tensor)
        human_tensor = transforms.ToTensor()(human_img).unsqueeze(0).to(DEVICE, torch.float16)

        # Log tensor shapes for debugging
        logger.info(f"pose_tensor shape: {pose_tensor.shape}")
        logger.info(f"garm_tensor shape: {garm_tensor.shape}")
        logger.info(f"mask_tensor shape: {mask_tensor.shape}")
        logger.info(f"human_tensor shape: {human_tensor.shape}")

        generator = torch.Generator(DEVICE).manual_seed(42)

        logger.info("Running diffusion pipeline")
        images = pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
            num_inference_steps=2,
            generator=generator,
            strength=1.0,
            pose_img=pose_tensor,
            text_embeds_cloth=prompt_embeds_c,
            cloth=garm_tensor,
            mask_image=mask_tensor,
            image=human_tensor,
            height=PROCESS_SIZE[0],  # Swap height and width
            width=PROCESS_SIZE[1],   # Swap width and height
            ip_adapter_image=garm_img,
            guidance_scale=15,
        )[0]
    return images

def blend_images(original: Image.Image, inpainted: Image.Image, mask: Image.Image) -> Image.Image:
    """Blend inpainted image with original using refined mask."""
    original_np = np.array(original)
    inpainted_np = np.array(inpainted)
    mask_resized = mask.resize(original.size, Image.Resampling.BILINEAR)
    mask_tensor = transforms.ToTensor()(mask_resized)
    mask_np = np.array(mask) / 255.0
    mask_np = cv2.resize(mask_np, (inpainted_np.shape[1], inpainted_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    if len(mask_np.shape) == 2:  # If mask is (height, width)
        mask_np = mask_np[..., np.newaxis]  # Shape becomes (height, width, 1)
    elif len(mask_np.shape) == 3 and mask_np.shape[2] == 3:  # If mask is (height, width, 3)
        mask_np = mask_np[..., 0:1]  # Take one channel, shape becomes (height, width, 1)
    blended_np = (inpainted_np * mask_np + original_np * (1 - mask_np)).astype(np.uint8)
    return Image.fromarray(blended_np)

def start_tryon(
    human_img: Image.Image,
    garm_img: Image.Image,
    resize_method: ResizeMethod,
    upscale_method: ResizeMethod,
    garment_des: str,
) -> Image.Image:
    """Process human and garment images for virtual try-on."""
    try:
        start_time = time.time()

        # Resize images
        garm_img_resized = resize_if_needed(garm_img, PROCESS_SIZE, method=resize_method, label="Garment")
        human_img_orig = human_img.convert("RGB")
        human_img_resized = resize_if_needed(human_img_orig, PROCESS_SIZE, method=resize_method, label="Human")

        # Process human image
        keypoints, model_parse = process_human_image(human_img_resized)

        # Generate mask
        mask = generate_mask(model_parse, keypoints)

        # Apply DensePose
        pose_img = apply_densepose(human_img_resized)

        # Run diffusion pipeline
        inpainted = run_diffusion_pipeline(human_img_resized, garm_img_resized, mask, pose_img, garment_des)

        # Blend images
        blended = blend_images(human_img_resized, inpainted[0], mask)

        # Final upscale
        final_result = final_upscale(blended, OUTPUT_SIZE, method=upscale_method)
        final_result = final_result.transpose(Image.FLIP_LEFT_RIGHT)

        total_time = time.time() - start_time
        logger.info(f"Try-on completed in {total_time:.2f} seconds")
        
        return final_result

    except Exception as e:
        logger.error(f"Error in try-on process: {e}")
        raise HTTPException(status_code=500, detail=f"Try-on process failed: {str(e)}")

# API Endpoints
@app.post("/tryon/")
async def tryon_endpoint(
    human_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    resize_method: str = Query(ResizeMethod.STANDARD.value, description="Resize method: standard, esrgan, seam_carve"),
    upscale_method: str = Query(ResizeMethod.STANDARD.value, description="Upscale method: standard, esrgan, seam_carve"),
    garment_description: str = Query("a garment", description="Description of the garment (e.g., 'a red shirt')"),
):
    """Endpoint for virtual try-on."""
    try:
        # Validate resize and upscale methods
        resize_method_enum = ResizeMethod(resize_method)
        upscale_method_enum = ResizeMethod(upscale_method)

        # Validate and load images
        if not human_image.content_type.startswith("image/") or not garment_image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded files must be images")

        human_img = Image.open(human_image.file).convert("RGB")
        garm_img = Image.open(garment_image.file).convert("RGB")

        if human_img.size[0] < 50 or human_img.size[1] < 50 or garm_img.size[0] < 50 or garm_img.size[1] < 50:
            raise HTTPException(status_code=400, detail="Images must be at least 50x50 pixels")

        # Initialize ESRGAN if needed
        if resize_method_enum == ResizeMethod.ESRGAN or upscale_method_enum == ResizeMethod.ESRGAN:
            global esrgan_model
            if esrgan_model is None:
                esrgan_model = initialize_esrgan()

        # Process images
        logger.info(f"Starting try-on with resize: {resize_method}, upscale: {upscale_method}, garment description: {garment_description}")
        result_img = start_tryon(
            human_img,
            garm_img,
            resize_method=resize_method_enum,
            upscale_method=upscale_method_enum,
            garment_des=garment_description,
        )

        # Save result to bytes
        from io import BytesIO
        img_byte_arr = BytesIO()
        result_img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Virtual Try-On API",
        "status": "running",
        "resize_methods": [method.value for method in ResizeMethod],
        "upscale_methods": [method.value for method in ResizeMethod],
        "documentation": "/docs"
    }

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)