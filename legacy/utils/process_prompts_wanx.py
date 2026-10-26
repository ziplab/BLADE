from pathlib import Path
import os
import torch
import random
from tqdm import tqdm
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

# Set paths
prompt_file = str(Path(__file__).resolve().parent / "all_dimension_aug_wanx.txt")
target_dir = str(Path(__file__).resolve().parents[1] / "wanx" / "prompts")
os.makedirs(target_dir, exist_ok=True)

# Create individual embeddings directory
individual_dir = os.path.join(target_dir, "individual_embeddings")
os.makedirs(individual_dir, exist_ok=True)

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = str(Path(__file__).resolve().parents[1] / "wanx" / "wan1.3b")
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = scheduler
pipe.to("cuda")

# 设置批大小
batch_size = 32  # 可以根据你的显存调整此值

# 读取文本文件中的所有prompts
print("Loading prompts from file...")
with open(prompt_file, "r", encoding="utf-8") as f:
    all_prompts = [line.strip() for line in f.readlines() if line.strip()]

# 打乱prompts的顺序
print(f"Shuffling {len(all_prompts)} prompts...")
random.shuffle(all_prompts)

# 分批处理
device = pipe._execution_device
dtype = pipe.text_encoder.dtype
processed_count = 0

for i in tqdm(range(0, len(all_prompts), batch_size), desc="Processing prompts"):
    batch_prompts = all_prompts[i:i+batch_size]
    
    # 使用encode_prompt方法生成embeddings
    with torch.no_grad():
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=batch_prompts,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=device,
            dtype=dtype
        )
    
    # 逐个保存每个embedding到独立文件
    for j in range(prompt_embeds.shape[0]):
        embed_idx = processed_count + j
        single_embed = prompt_embeds[j:j+1].cpu()
        embed_path = os.path.join(individual_dir, f"{embed_idx}.pt")
        torch.save(single_embed, embed_path, _use_new_zipfile_serialization=True)
    
    processed_count += prompt_embeds.shape[0]
    
    # 释放内存
    del prompt_embeds
    torch.cuda.empty_cache()
    
    # 显示进度
    if processed_count % 100 == 0 or processed_count == len(all_prompts):
        print(f"已处理 {processed_count} 个embeddings")

print(f"已单独保存 {processed_count} 个embeddings到目录: {individual_dir}")

# 处理fixed prompt
fixed_prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
uncond_prompt = ""
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
print("Processing fixed prompt and negative prompt...")
with torch.no_grad():
    # 处理fixed prompt
    fixed_prompt_embeds, _ = pipe.encode_prompt(
        prompt=fixed_prompt,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=device,
        dtype=dtype
    )
    
    # 处理uncond prompt (空字符串)
    uncond_prompt_embeds, _ = pipe.encode_prompt(
        prompt=uncond_prompt,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=device,
        dtype=dtype
    )

    negative_prompt_embeds, _ = pipe.encode_prompt(
        prompt=negative_prompt,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=device,
        dtype=dtype
    )
    
    # 移到CPU
    fixed_prompt_embeds = fixed_prompt_embeds.cpu()
    uncond_prompt_embeds = uncond_prompt_embeds.cpu()
    negative_prompt_embeds = negative_prompt_embeds.cpu()

# 清除缓存
torch.cuda.empty_cache()

# 保存其他数据（不包括prompt_embeds_shuffled.pt，改为单独文件存储）
torch.save(all_prompts, os.path.join(target_dir, "prompts_shuffled.pt"))
torch.save(fixed_prompt, os.path.join(target_dir, "fixed_prompt.pt"))
torch.save(fixed_prompt_embeds, os.path.join(target_dir, "fixed_prompt_embeds.pt"))
torch.save(uncond_prompt, os.path.join(target_dir, "uncond_prompt.pt"))
torch.save(uncond_prompt_embeds, os.path.join(target_dir, "uncond_prompt_embeds.pt"))
torch.save(negative_prompt, os.path.join(target_dir, "negative_prompt.pt"))
torch.save(negative_prompt_embeds, os.path.join(target_dir, "negative_prompt_embeds.pt"))

print(f"All data saved to {target_dir}:")
print(f"- Shuffled prompts count: {len(all_prompts)}")
print(f"- Individual embeddings count: {processed_count}")
print(f"- Individual embeddings directory: {individual_dir}")
print(f"- fixed_prompt_embeds shape: {fixed_prompt_embeds.shape}")
print(f"- uncond_prompt_embeds shape: {uncond_prompt_embeds.shape}")
print(f"- negative_prompt_embeds shape: {negative_prompt_embeds.shape}") 