from diffusers import UNet2DModel
from diffusers import DDPMScheduler
import PIL.Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm

repo_id = "google/ddpm-cat-256"
model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)
scheduler = DDPMScheduler.from_pretrained(repo_id)

def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    print(f"Image at step {i}")
    plt.imshow(image_pil)
    plt.show()


torch.manual_seed(0)
noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)

model.to("cuda")
noisy_sample = noisy_sample.to("cuda")

sample = noisy_sample

print(scheduler.timesteps)

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    # 1. predict noise residual
    with torch.no_grad():
        residual = model(sample, t).sample

    # 2. compute less noisy image and set x_t -> x_t-1
    sample = scheduler.step(residual, t, sample).prev_sample

    # 3. optionally look at image
    if (i + 1) % 50 == 0:
        display_sample(sample, i + 1)