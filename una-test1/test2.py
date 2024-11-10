#experiment 1 

import torch
from diffusers import DDPMScheduler, UNet2DModel
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load and process the input image
image_path = 'cat.jpg'  
image = Image.open(image_path).convert('RGB')

# Resize and center-crop the image to 64x64 pixels
preprocess = transforms.Compose([
    transforms.Resize(64), # 64 for faster results
    transforms.CenterCrop(64),
    transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
])

image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension


#initialize model and scheduler
# Load pre-trained UNet model
model = UNet2DModel.from_pretrained('google/ddpm-cifar10-32')


# Load the scheduler
scheduler = DDPMScheduler(num_train_timesteps=1000)


# Add noise to the image - forward process
def add_noise(image, scheduler, timesteps):
    noisy_images = []
    for t in timesteps:
        noise = torch.randn_like(image)
        alpha_prod = scheduler.alphas_cumprod[t] ** 0.5
        beta_prod = (1 - scheduler.alphas_cumprod[t]) ** 0.5
        noisy_image = alpha_prod * image + beta_prod * noise
        noisy_images.append(noisy_image)
    return noisy_images

# Select timesteps to observe
timesteps = torch.linspace(0, 999, steps=200).long()

noisy_images = add_noise(image_tensor, scheduler, timesteps)


# Denoising process - backward process
@torch.no_grad()
def denoise_image(noisy_image, model, scheduler, timesteps):
    for t in reversed(timesteps):
        # Prepare model input
        model_input = noisy_image

        # Predict the noise residual
        noise_pred = model(model_input, t)['sample']

        # Compute the previous noisy sample x_t -> x_t-1
        noisy_image = scheduler.step(noise_pred, t, noisy_image)['prev_sample']
    return noisy_image

# Use the noisiest image (after full forward process)
noisy_image = noisy_images[-1]

# Denoise
reconstructed_image = denoise_image(noisy_image, model, scheduler, timesteps)


# Display images
def show_images(images, titles=None):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    if titles is None:
        titles = [''] * len(images)
    for img, ax, title in zip(images, axes, titles):
        img = img.squeeze().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.show()

# Display original, noisy, and reconstructed images
show_images(
    [image_tensor, noisy_image, reconstructed_image],
    titles=['Original Image', 'Noisy Image', 'Reconstructed Image']
)


# #-----------------------------modify midway------------------------------------------------------
# # Select a timestep halfway
# mid_timestep = timesteps[len(timesteps) // 2]

# # Get the noisy image at midway
# mid_noisy_image = noisy_images[len(timesteps) // 2]

# # Modify a few pixels
# modified_image = mid_noisy_image.clone()
# # Example modification: Set a square region to zero
# modified_image[:, :, 20:30, 20:30] = 0

# # Continue denoising from the modified image
# remaining_timesteps = timesteps[:len(timesteps) // 2]

# modified_reconstructed_image = denoise_image(modified_image, model, scheduler, remaining_timesteps)

# # Display the modified and final images
# show_images(
#     [mid_noisy_image, modified_image, modified_reconstructed_image],
#     titles=['Midway Noisy Image', 'Modified Noisy Image', 'Final Reconstructed Image']
# )
