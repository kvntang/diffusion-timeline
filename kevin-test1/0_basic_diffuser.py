import torch
from diffusers import StableDiffusionPipeline


# Load the Stable Diffusion v1-5 model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Define the prompt
prompt = "A cat with bird wings soaring in the clouds. Photorealstic."

# Generate the image
image = pipe(
    prompt,
    num_inference_steps=28,
    guidance_scale=7.5  # Higher guidance scale usually means better adherence to the prompt
).images[0]

# Save the generated image
image.save("cat_flying.jpg")
