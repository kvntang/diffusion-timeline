# deterministic denoising using DDIMScheduler
# kinda works 
# experiment 2

# Import required libraries
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load your input image
image_path = 'cat.jpg'  # Replace with your image path
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
    image_tensor = image_tensor.to(device, dtype=pipeline.unet.dtype)  # Ensure the image tensor matches model dtype
    latent = pipeline.vae.encode(image_tensor).latent_dist.sample() * 0.18215

# Function to add noise to the latent - forward process
def add_noise(latent, scheduler, timesteps):
    noisy_latents = []
    for t in timesteps:
        noise = torch.randn_like(latent)
        noisy_latent = scheduler.add_noise(latent, noise, t)
        noisy_latents.append(noisy_latent)
    return noisy_latents

# Select timesteps (using fewer steps for faster and more stable results)
num_inference_steps = 100  # You can experiment with 50-100 steps
scheduler.set_timesteps(num_inference_steps)
timesteps = scheduler.timesteps

# Add noise to the latent
noisy_latents = add_noise(latent, scheduler, timesteps)

# Prepare the text embeddings (even if empty, required by the model)
text_input = pipeline.tokenizer(
    "", padding="max_length", max_length=pipeline.tokenizer.model_max_length, return_tensors="pt"
)
text_embeddings = pipeline.text_encoder(text_input.input_ids.to(device))[0]

# Denoising process - backward process
@torch.no_grad()
def denoise_latent(noisy_latent, scheduler, timesteps, text_embeddings):
    for t in reversed(timesteps):
        # Prepare model input
        latent_model_input = noisy_latent

        # Predict the noise residual
        noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Compute the previous noisy sample x_t -> x_t-1
        noisy_latent = scheduler.step(noise_pred, t, noisy_latent).prev_sample
    return noisy_latent

# Use the noisiest latent (after full forward process)
noisy_latent = noisy_latents[-1]

# Denoise the latent deterministically
reconstructed_latent = denoise_latent(noisy_latent, scheduler, timesteps, text_embeddings)

# Decode the latent back to image space
with torch.no_grad():
    reconstructed_image = pipeline.vae.decode(reconstructed_latent / 0.18215).sample

# Convert the reconstructed image tensor to a PIL image
reconstructed_image = (reconstructed_image / 2 + 0.5).clamp(0, 1)
reconstructed_image = reconstructed_image.cpu().permute(0, 2, 3, 1).numpy()
reconstructed_image_pil = Image.fromarray((reconstructed_image[0] * 255).astype(np.uint8))

# Display the reconstructed image
reconstructed_image_pil.show()


#denoise for 10 steps, modify the pixels a bit and then finish the denoising process
