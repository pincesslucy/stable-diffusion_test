from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL, PNDMScheduler, UniPCMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import PIL.Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm

torch_device = "cuda"

#시각화
def display_sample(sample, i):
    image = (sample / 2 + 0.5).clamp(0, 1)
    image_processed = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image_processed = (image_processed * 255).round().astype("uint8")

    image_pil = PIL.Image.fromarray(image_processed[0])
    print(f"Image at step {i}")
    plt.imshow(image_pil)
    plt.show()

#모델 설정
repo_id = "runwayml/stable-diffusion-v1-5"
unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet")

#스케줄러 설정
scheduler = UniPCMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")
#스케줄러 노이즈 제거 timestemp 설정
#scheduler.set_timesteps(50)

#vae설정
vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae")

#토크나이저설정
tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer")

#텍스트 인코더 설정
text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder")

#gpu설정
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)



#컨디션 설정
prompt = ["a photograph of an astronaut riding a horse"]
height = 512  # Stable Diffusion의 기본 높이
width = 512  # Stable Diffusion의 기본 너비
num_inference_steps = 80  # 노이즈 제거 스텝 수
guidance_scale = 7.5  # classifier-free guidance를 위한 scale
generator = torch.manual_seed(0)  # 초기 잠재 노이즈를 생성하는 seed generator
batch_size = len(prompt)

#토큰화
text_input = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
)

#임베딩
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]


max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

#랜덤 노이즈 생성
noisy_sample = torch.randn(batch_size, unet.config.in_channels, height//8, width//8, generator=generator)

noisy_sample = noisy_sample.to(torch_device)



noisy_sample = noisy_sample * scheduler.init_noise_sigma
scheduler.set_timesteps(num_inference_steps)

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):

    latent_model_input = torch.cat([noisy_sample] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # 1. predict noise residual
    with torch.no_grad():
        residual = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    #guidance
    noise_pred_uncond, noise_pred_text = residual.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # 2. compute less noisy image and set x_t -> x_t-1
    noisy_sample = scheduler.step(noise_pred, t, noisy_sample).prev_sample

    # latent를 스케일링하고 vae로 이미지 디코딩
    latents = 1 / 0.18215 * noisy_sample
    with torch.no_grad():
        image = vae.decode(latents).sample

    # 3. optionally look at image
    if (i + 1) % 20 == 0:
        display_sample(noisy_sample, i + 1)
    if i == num_inference_steps-1:
        display_sample(noisy_sample, i + 1)