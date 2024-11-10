# experiment 4

import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, AutoencoderKL
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

def load_image(image_path, size=512):
    """Load and preprocess image."""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

def save_image(tensor, path):
    """Convert tensor to image and save."""
    image = tensor.cpu().permute(0, 2, 3, 1).numpy()
    image = ((image[0] * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)

def setup_pipeline(device='cuda'):
    """Setup the img2img pipeline with optimized settings."""
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    # Load pipeline with better defaults
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None  # Disable safety checker for faster processing
    ).to(device)
    
    # Optimize memory usage
    pipeline.enable_attention_slicing()
    if device == 'cuda':
        pipeline.enable_model_cpu_offload()
    
    # Use DDIM scheduler for better reconstruction
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(50)  # Reduced steps for faster processing
    
    return pipeline

def reconstruct_image(pipeline, image, num_inference_steps=50, strength=0.75):
    """Reconstruct image with proper conditioning."""
    
    # Generate automatic prompt based on image content
    # In a production system, you might want to use a vision-language model here
    prompt = "a detailed photograph of a cat, high quality, professional lighting, sharp focus"
    negative_prompt = "blurry, low quality, distorted, abstract, artistic"
    
    with torch.no_grad():
        # Convert to PIL
        init_image = image.cpu().squeeze(0).permute(1, 2, 0)
        init_image = ((init_image + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
        init_image = Image.fromarray(init_image)
        
        # Generate reconstruction
        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=7.5,  # Reduced for better reconstruction
        ).images[0]
        
        return output

def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load image
    image_path = 'cat.jpg'
    original_image = load_image(image_path)
    
    # Setup pipeline
    pipeline = setup_pipeline(device)
    
    # Process with different strengths
    strengths = [0.3, 0.5, 0.7]
    for strength in strengths:
        print(f"\nProcessing with strength {strength}")
        
        # Reconstruct
        reconstructed = reconstruct_image(
            pipeline,
            original_image.to(device),
            strength=strength
        )
        
        # Save
        reconstructed.save(f'reconstructed_strength_{strength}.png')
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()