import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
from PIL import Image
import numpy as np

# Helper function to decode latent and save as an image
def save_latent_as_image(latent, pipeline, filename):
    with torch.no_grad():
        decoded = pipeline.vae.decode(latent / 0.18215).sample
        image = (decoded / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = Image.fromarray((image[0] * 255).astype(np.uint8))
        image.save(filename)
        return image

# Function to perform the denoising steps forward
@torch.no_grad()
def denoise_latent(noisy_latent, scheduler, timesteps, start_timestep_index, denoise_steps, text_embeddings):
    """Denoise the latent for a specified number of forward steps."""
    for i in range(denoise_steps):
        t = timesteps[start_timestep_index + i]  # Move forward in time
        # Predict the noise residual
        noise_pred = pipeline.unet(noisy_latent, t, encoder_hidden_states=text_embeddings).sample
        # Perform a single denoising step
        noisy_latent = scheduler.step(noise_pred, t, noisy_latent).prev_sample
    return noisy_latent

##########################################################################################
##########################################################################################
##########################################################################################

# Adjustable parameters
start_timestep_position = 20  # Starting position in the timeline (where input image is assumed to be)
denoise_steps = 80           # Number of steps to denoise forward

# Load and preprocess input image
image_path = 'renoised_step_back_20.png'
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

# Set starting point based on the `start_timestep_position`
start_timestep_index = min(len(timesteps) - 1, start_timestep_position)

# Encode image to latent space
with torch.no_grad():
    image_tensor = image_tensor.to(device, dtype=pipeline.unet.dtype)
    latent = pipeline.vae.encode(image_tensor).latent_dist.sample() * 0.18215

# Generate text embeddings (optional) if you want to condition on text
prompt = " "  # empty prompt
text_input = pipeline.tokenizer(prompt, return_tensors="pt").to(device)
text_embeddings = pipeline.text_encoder(text_input.input_ids)[0]

# Denoise the latent forward by `denoise_steps`
denoised_latent = denoise_latent(latent, scheduler, timesteps, start_timestep_index, denoise_steps, text_embeddings)

# Save the final denoised image after the forward steps
denoised_image = save_latent_as_image(denoised_latent, pipeline, f'denoised_step_forward_{denoise_steps}.png')

# Output saved file summary
print("Saved image:")
print(f"- denoised_step_forward_{denoise_steps}.png: Denoised image after {denoise_steps} forward steps from timestep {start_timestep_position}")
