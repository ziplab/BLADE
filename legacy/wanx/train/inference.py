import torch
import os
import argparse
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.models import WanTransformer3DModel


def load_prompts(prompt_file):
    """Load a list of prompts from a file."""
    prompts = []
    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                prompts.append(line)
    return prompts


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='WanX Batch Inference Script')
    parser.add_argument('--lora_path', type=str,
                        required=True,
                        help='Path to the LoRA weights file')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Specify the GPU device ID to use (default: 0)')

    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES to make only the specified GPU visible
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print(f"Set CUDA_VISIBLE_DEVICES to: {args.gpu}")

    # Set up the GPU device (now always use cuda:0 since only one GPU is visible)
    if not torch.cuda.is_available():
        print("CUDA is not available, using CPU.")
        device = "cpu"
    else:
        device = "cuda:0"  # Always use cuda:0 since we set CUDA_VISIBLE_DEVICES

    print(f"Using device: {device}")
    print(f"LoRA weights path: {args.lora_path}")

    # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    model_id = "wan1.3b"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler

    # 设置稀疏注意力
    from modify_wan import set_adaptive_block_sparse_attn_wanx
    set_adaptive_block_sparse_attn_wanx(pipe.transformer)
    print('Successfully set sparsity to the transformer')

    # Check if the LoRA weights file exists
    if os.path.exists(args.lora_path):
        pipe.load_lora_weights(args.lora_path)
        print(f"Successfully loaded LoRA weights from: {args.lora_path}")
    else:
        raise ValueError(f"LoRA weights file not found at: {args.lora_path}")

    pipe.to(device)

    # Load prompts
    prompt_file = "prompts/inference_prompts.txt"
    prompts = load_prompts(prompt_file)
    print(f"Loaded {len(prompts)} prompts.")

    # Create the output directory
    output_dir = "outputs/inference/wanx_batch"
    os.makedirs(output_dir, exist_ok=True)

    # Default negative prompt
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    # Batch inference
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}/{len(prompts)}:")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")

        try:
            # Use a different seed for each prompt
            generator = torch.manual_seed(8888 + i)
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=480,
                width=832,
                num_frames=81,
                num_inference_steps=8,
                guidance_scale=1.0,
                generator=generator,
            ).frames[0]

            # Save the video
            output_path = f"{output_dir}/{i+1:02d}.mp4"
            export_to_video(output, output_path, fps=16)
            print(f"Video saved to: {output_path}")

        except Exception as e:
            print(f"An error occurred while processing prompt {i+1}: {str(e)}")
            continue

    print(f"\nBatch inference completed! Processed {len(prompts)} prompts in total.")


if __name__ == "__main__":
    main()