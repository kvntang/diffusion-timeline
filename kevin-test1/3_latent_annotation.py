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
start_timestep_position = 50  # Starting position in the timeline (where input image is assumed to be)
denoise_steps = 50           # Number of steps to denoise forward

# Load and preprocess input images
image_paths = ['renoised_step_back_50.png', 'anot1.png']
preprocess = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Encode images into latent space
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32

pipeline = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=dtype
).to(device)

scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler = scheduler

num_inference_steps = 100
scheduler.set_timesteps(num_inference_steps)
timesteps = scheduler.timesteps

start_timestep_index = min(len(timesteps) - 1, start_timestep_position)

latents = []
for image_path in image_paths:
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device, dtype=pipeline.unet.dtype)
    with torch.no_grad():
        latent = pipeline.vae.encode(image_tensor).latent_dist.sample() * 0.18215
    latents.append(latent)


###pick one method of merging
# 1. Merge the two latents by averaging
#merged_latent = (latents[0] + latents[1]) / 2

# 2. weighted blending
alpha = 0.7  # Weight for primary latent
beta = 1 - alpha  # Weight for scribble overlay
merged_latent = alpha * latents[0] + beta * latents[1]




# Generate text embeddings for conditioning
prompt = "A photorealistic cat with graceful bird wings, soaring high in a bright blue sky filled with soft, white clouds, with sunlight illuminating the wings and casting a warm glow."
text_input = pipeline.tokenizer(prompt, return_tensors="pt").to(device)
text_embeddings = pipeline.text_encoder(text_input.input_ids)[0]

# Denoise the merged latent forward by `denoise_steps`
denoised_latent = denoise_latent(merged_latent, scheduler, timesteps, start_timestep_index, denoise_steps, text_embeddings)

# Save the final denoised image after the forward steps
denoised_image = save_latent_as_image(denoised_latent, pipeline, f'merged_denoised_step_forward_{denoise_steps}.png')

# Output saved file summary
print("Saved image:")
print(f"- merged_denoised_step_forward_{denoise_steps}.png: Denoised image after {denoise_steps} forward steps from timestep {start_timestep_position}")
