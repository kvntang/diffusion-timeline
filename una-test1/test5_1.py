#  experiment 2.1

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def save_latent_as_image(latent, pipeline, filename):
    """Helper function to decode and save a latent as an image"""
    with torch.no_grad():
        # Decode the latent using the VAE
        decoded = pipeline.vae.decode(latent / 0.18215).sample
        # Convert to image format
        image = (decoded / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = Image.fromarray((image[0] * 255).astype(np.uint8))
        # Save the image
        image.save(filename)
        return image

# Load and preprocess image
image_path = 'cat.jpg'  
image = Image.open(image_path).convert('RGB')

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
image_tensor = preprocess(image).unsqueeze(0)

# Setup device and pipeline
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32

pipeline = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=dtype
).to(device)

# Setup scheduler
scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler = scheduler

# Set number of steps and timesteps
num_inference_steps = 100
scheduler.set_timesteps(num_inference_steps)
timesteps = scheduler.timesteps

# Encode image to latent space
with torch.no_grad():
    image_tensor = image_tensor.to(device, dtype=pipeline.unet.dtype)
    latent = pipeline.vae.encode(image_tensor).latent_dist.sample() * 0.18215

# Function to add noise and save intermediate results
def add_noise_and_save(latent, scheduler, timesteps, pipeline, save_steps=[0, 25, 50, 75, 99]):
    noisy_latents = []
    noise = torch.randn_like(latent)  # Generate noise once for deterministic results
    
    plt.figure(figsize=(20, 4))
    for i, t in enumerate(timesteps):
        # Add noise for this timestep
        noisy_latent = scheduler.add_noise(latent, noise, t)
        noisy_latents.append(noisy_latent)
        
        # Save at specified steps
        if i in save_steps:
            # Save individual image
            img = save_latent_as_image(noisy_latent, pipeline, f'noise_step_{i}.png')
            
            # Add to plot
            plt.subplot(1, len(save_steps), save_steps.index(i) + 1)
            plt.imshow(np.array(img))
            plt.title(f'Step {i}/{num_inference_steps-1}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('noise_progression.png')
    plt.close()
    
    return noisy_latents

# Add noise and save visualizations
noisy_latents = add_noise_and_save(latent, scheduler, timesteps, pipeline)

# Save the final pure noise state separately
final_noisy_latent = noisy_latents[-1]
save_latent_as_image(final_noisy_latent, pipeline, 'pure_noise.png')

print("Saved images:")
print("- noise_progression.png: Shows the progression of noise addition")
print("- pure_noise.png: The final pure noise state")
print("- Individual steps saved as noise_step_X.png")