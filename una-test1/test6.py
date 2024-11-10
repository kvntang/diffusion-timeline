# Import required libraries
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
from PIL import Image
import numpy as np

# Load your input image
image_path = 'cat.jpg' 
image = Image.open(image_path).convert('RGB')

# Resize and center-crop the image to 512x512 pixels
preprocess = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
])
image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

# Set the device to 'cuda' if available, otherwise 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the Stable Diffusion pipeline
if device == 'cuda':
    # Use float16 for GPU
    pipeline = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        torch_dtype=torch.float16
    ).to(device)
else:
    # Use float32 for CPU
    pipeline = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        torch_dtype=torch.float32
    ).to(device)

# Use DDIMScheduler for deterministic reconstruction
scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler = scheduler  # Set the scheduler in the pipeline

# Encode the image into latent space
with torch.no_grad():
    image_tensor = image_tensor.to(device, dtype=pipeline.unet.dtype)
    latent = pipeline.vae.encode(image_tensor).latent_dist.sample() * 0.18215

# Function to add noise to the latent - forward process
def add_noise(latent, scheduler, timesteps):
    # Use a fixed noise tensor
    noise = torch.randn_like(latent)
    # Get the final noisy latent (fully noised)
    final_noisy_latent = scheduler.add_noise(latent, noise, timesteps[0])
    return final_noisy_latent, noise

# Helper function to convert latent to image
def latent_to_image(latent, pipeline):
    with torch.no_grad():
        image = pipeline.vae.decode(latent / 0.18215).sample
    
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return Image.fromarray((image[0] * 255).astype(np.uint8))

# Select timesteps
num_inference_steps = 100
scheduler.set_timesteps(num_inference_steps)
timesteps = scheduler.timesteps

# Add noise to the latent
noisy_latent, noise = add_noise(latent, scheduler, timesteps)

# Save the pure noise
with torch.no_grad():
    pure_noise_latent = noise * scheduler.init_noise_sigma
    pure_noise_image = latent_to_image(pure_noise_latent, pipeline)
    pure_noise_image.save('pure_noise.jpg')

# Save the noisy image (fully noised input)
noisy_image = latent_to_image(noisy_latent, pipeline)
noisy_image.save('noisy_input.jpg')

# Prepare the text embeddings
text_input = pipeline.tokenizer(
    "cat", padding="max_length", max_length=pipeline.tokenizer.model_max_length, return_tensors="pt"
)
text_embeddings = pipeline.text_encoder(text_input.input_ids.to(device))[0]

# Denoising process for 10 steps
@torch.no_grad()
def partially_denoise_latent(noisy_latent, scheduler, num_steps, text_embeddings):
    # Take only the first num_steps timesteps
    partial_timesteps = timesteps[-num_steps:]
    
    current_latent = noisy_latent
    for t in reversed(partial_timesteps):
        latent_model_input = current_latent
        noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        current_latent = scheduler.step(noise_pred, t, current_latent).prev_sample
    
    return current_latent

# Denoise for 10 steps
partially_denoised_latent = partially_denoise_latent(noisy_latent, scheduler, 10, text_embeddings)

# Save the partially denoised image
partially_denoised_image = latent_to_image(partially_denoised_latent, pipeline)
partially_denoised_image.save('partially_denoised_10steps.jpg')

# For comparison, fully denoise the image
@torch.no_grad()
def fully_denoise_latent(noisy_latent, scheduler, timesteps, text_embeddings):
    current_latent = noisy_latent
    for t in reversed(timesteps):
        latent_model_input = current_latent
        noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        current_latent = scheduler.step(noise_pred, t, current_latent).prev_sample
    return current_latent

# Fully denoise and save
fully_denoised_latent = fully_denoise_latent(noisy_latent, scheduler, timesteps, text_embeddings)
fully_denoised_image = latent_to_image(fully_denoised_latent, pipeline)
fully_denoised_image.save('fully_denoised.jpg')