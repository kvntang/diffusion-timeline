import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Helper function to decode latent and save as an image
def save_latent_as_image(latent, pipeline, filename):
    """Decode latent and save as an image file"""
    with torch.no_grad():
        # Decode the latent using the VAE
        decoded = pipeline.vae.decode(latent / 0.18215).sample
        # Convert to image format
        image = (decoded / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = Image.fromarray((image[0] * 255).astype(np.uint8))
        # Save and return image
        image.save(filename)
        return image

# Helper function to generate noise with specific variance and seed
def generate_noise(latent_shape, device, dtype, seed, noise_variance_scale):
    """Generate noise with a given variance and seed"""
    generator = torch.Generator(device=device).manual_seed(seed)  # Set up generator for reproducibility
    noise = noise_variance_scale * torch.randn(latent_shape, device=device, dtype=dtype, generator=generator)
    return noise

# Helper function to calculate target timestep index based on renoise steps
def calculate_target_timestep_index(timesteps, renoise_steps):
    """Calculate the target index in timesteps based on renoise steps"""
    return len(timesteps) - renoise_steps

# Helper function to apply noise to the latent at a specific timestep
def apply_noise_at_timestep(latent, noise, scheduler, timestep):
    """Apply noise to latent representation at a specific timestep"""
    return scheduler.add_noise(latent, noise, timestep)

##########################################################################################
##########################################################################################
##########################################################################################

# Adjustable parameters
renoise_steps = 5          # Define how many steps backward to visualize
seed = 42                  # Seed for reproducibility
noise_variance_scale = 1.0 # Control the variance of the noise

# Load and preprocess input image
image_path = 'a2.jpg'
image = Image.open(image_path).convert('RGB')

preprocess = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
image_tensor = preprocess(image).unsqueeze(0)

# Setup device and load pipeline
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32

pipeline = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=dtype
).to(device)

# Configure scheduler and timesteps
scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler = scheduler

num_inference_steps = 100
scheduler.set_timesteps(num_inference_steps)
timesteps = scheduler.timesteps

# Encode image to latent space
with torch.no_grad():
    image_tensor = image_tensor.to(device, dtype=pipeline.unet.dtype)
    latent = pipeline.vae.encode(image_tensor).latent_dist.sample() * 0.18215

# Generate noise
noise = generate_noise(latent.shape, latent.device, latent.dtype, seed, noise_variance_scale)

# Calculate target timestep index
target_index = calculate_target_timestep_index(timesteps, renoise_steps)

# Apply noise at the target timestep to get the renoised latent
noisy_latent = apply_noise_at_timestep(latent, noise, scheduler, timesteps[target_index])

# Save the renoised image
renoised_image = save_latent_as_image(noisy_latent, pipeline, f'renoised_step_back_{renoise_steps}.png')

# Output saved file summary
print("Saved image:")
print(f"- renoised_step_back_{renoise_steps}.png: Renoised image at {renoise_steps} steps back from the fully noisy state")
