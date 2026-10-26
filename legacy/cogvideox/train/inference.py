import torch
import os
import argparse
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from diffusers.schedulers import CogVideoXDPMScheduler
from modify_cogvideo import set_block_sparse_attn_cogvideox
from diffusers.models import CogVideoXTransformer3DModel


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
    parser = argparse.ArgumentParser(description='CogVideoX Batch Inference Script')
    parser.add_argument('--lora_path', type=str,
                        default="outputs/cogvideox/your_checkpoint",
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

    # Initialize the pipeline
    pipe = CogVideoXPipeline.from_pretrained("CogVideoX-5b", torch_dtype=torch.bfloat16)
    pipe.to(device)
    transformer = pipe.transformer
    # Set the ASA attention for CogVideoX
    set_block_sparse_attn_cogvideox(transformer)

    pipe.vae.enable_slicing() # Save memory
    pipe.vae.enable_tiling() # Save memory

    # Check if the LoRA weights file exists
    if os.path.exists(args.lora_path):
        pipe.load_lora_weights(args.lora_path)
        print(f"Successfully loaded LoRA weights from: {args.lora_path}")
    else:
        if not os.path.exists(args.lora_path):
            raise ValueError(f"LoRA weights file not found at: {args.lora_path}")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )

    # Load prompts
    prompt_file = "prompts/inference_prompts.txt"
    prompts = load_prompts(prompt_file)
    print(f"Loaded {len(prompts)} prompts.")

    # Create the output directory
    output_dir = "outputs/inference"
    os.makedirs(output_dir, exist_ok=True)

    # Batch inference
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}/{len(prompts)}:")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")

        try:
            # Use a different seed for each prompt
            generator = torch.manual_seed(8888 + i)
            frames = pipe(prompt,
                          guidance_scale=1,
                          num_inference_steps=8,
                          num_frames=49,
                          generator=generator,
                          use_dynamic_cfg=True).frames[0]

            # Save the video
            output_path = f"{output_dir}/{i+1:02d}.mp4"
            export_to_video(frames, output_path, fps=8)
            print(f"Video saved to: {output_path}")

        except Exception as e:
            print(f"An error occurred while processing prompt {i+1}: {str(e)}")
            continue

    print(f"\nBatch inference completed! Processed {len(prompts)} prompts in total.")


if __name__ == "__main__":
    main()