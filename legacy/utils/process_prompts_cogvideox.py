from pathlib import Path
import os
# import glob # Not needed when reading from a single file
import random
import torch
from tqdm import tqdm
import argparse

def process_prompts_to_embeddings(
    input_file, # Changed from input_dir
    output_dir,
    model_path,
    batch_size=32,
    device="cuda",
    seed=42,
    max_prompts=10000,
    save_separate=True
):
    """
    Process all prompts from a single txt file (one prompt per line), shuffle them,
    and convert to embeddings using CogVideoX pipeline.
    
    Args:
        input_file: Path to the input txt file (one prompt per line)
        output_dir: Directory to save processed files
        model_path: Path to CogVideoX model
        batch_size: Batch size for processing
        device: Device to use for processing
        seed: Random seed for shuffling
        max_prompts: Maximum number of prompts to process
        save_separate: Whether to save each embedding separately
    """
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a subdirectory for individual embeddings if needed
    separate_dir = os.path.join(output_dir, "individual_embeddings")
    if save_separate:
        os.makedirs(separate_dir, exist_ok=True)
        print(f"创建单独embeddings目录: {separate_dir}")
    
    # Load CogVideoX pipeline
    from diffusers import CogVideoXPipeline
    print(f"正在加载CogVideoX模型: {model_path}")
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    
    # Extract and load all prompts from the single txt file
    prompts = []
    print(f"从文件读取prompts: {input_file}")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="读取prompts"):
                prompt = line.strip()
                if prompt: # Skip empty lines
                    prompts.append(prompt)
    except FileNotFoundError:
        print(f"错误: 输入文件 {input_file} 未找到.")
        return None
    except Exception as e:
        print(f"读取文件 {input_file} 时出错: {e}")
        return None
    
    print(f"成功加载 {len(prompts)} 个prompts")
    
    # Shuffle prompts
    random.shuffle(prompts)
    
    # Limit to max_prompts
    if max_prompts > 0 and len(prompts) > max_prompts:
        prompts = prompts[:max_prompts]
        print(f"限制为前 {max_prompts} 个prompts")
    
    # Save shuffled prompts
    prompts_path = os.path.join(output_dir, "shuffled_prompts.pt")
    torch.save(prompts, prompts_path)
    print(f"已保存打乱后的prompts到 {prompts_path}")
    
    # Define fixed prompt
    fixed_prompt = (
        " A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The"
        " panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other"
        " pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo,"
        " casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays."
        " The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical"
        " atmosphere of this unique musical performance "
    )
    
    # Process prompts to embeddings
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    all_embeddings_list = [] if not save_separate else None # Use a different name to avoid confusion
    
    processed_count = 0
    
    for i in tqdm(range(num_batches), desc="生成embeddings"):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
        with torch.no_grad():
            try:
                prompt_embeds, _ = pipe.encode_prompt(
                    prompt=batch_prompts,
                    do_classifier_free_guidance=False,
                    device=device
                )
                
                if save_separate:
                    for j in range(prompt_embeds.shape[0]):
                        embed_idx = processed_count + j
                        single_embed = prompt_embeds[j:j+1].cpu()
                        embed_path = os.path.join(separate_dir, f"{embed_idx}.pt")
                        torch.save(single_embed, embed_path, _use_new_zipfile_serialization=True)
                else:
                    all_embeddings_list.append(prompt_embeds.cpu())
                
                processed_count += prompt_embeds.shape[0]
                
                if (processed_count == 1 and save_separate) or (processed_count % 100 == 0 and save_separate) :
                    print(f"已处理 {processed_count} 个embeddings")
                    if processed_count > 0:
                        first_file = os.path.join(separate_dir, "0.pt")
                        if os.path.exists(first_file):
                            size_mb = os.path.getsize(first_file) / (1024 * 1024)
                            print(f"单个embedding文件大小示例(0.pt): {size_mb:.2f} MB")
            except Exception as e:
                print(f"处理批次 {i} 时出错: {e}")
                print(f"跳过此批次")
    
    embeddings_output_path = None
    if not save_separate and all_embeddings_list:
        all_embeddings_tensor = torch.cat(all_embeddings_list, dim=0)
        embeddings_output_path = os.path.join(output_dir, "prompt_embeddings.pt")
        torch.save(all_embeddings_tensor, embeddings_output_path)
        print(f"已保存prompt embeddings到 {embeddings_output_path}")
        del all_embeddings_tensor
    elif save_separate:
        print(f"已单独保存 {processed_count} 个embeddings到目录: {separate_dir}")
        embeddings_output_path = separate_dir # For return value
    else:
        print("没有成功生成任何embeddings")
    
    del all_embeddings_list # Free memory

    # Generate uncond_prompt_embed
    print("生成unconditioned prompt embedding")
    with torch.no_grad():
        uncond_prompt_embed, _ = pipe.encode_prompt(
            prompt=[" "],
            do_classifier_free_guidance=False,
            device=device
        )
    
    uncond_path = os.path.join(output_dir, "uncond_prompt_embed.pt")
    torch.save(uncond_prompt_embed.cpu(), uncond_path)
    print(f"已保存uncond_prompt_embed到 {uncond_path}")
    
    # Generate fixed_prompt_embedding
    print("生成fixed prompt embedding")
    with torch.no_grad():
        fixed_prompt_embedding, _ = pipe.encode_prompt(
            prompt=[fixed_prompt],
            do_classifier_free_guidance=False,
            device=device
        )
    
    fixed_path = os.path.join(output_dir, "fixed_prompt_embedding.pt")
    torch.save(fixed_prompt_embedding.cpu(), fixed_path)
    print(f"已保存fixed_prompt_embedding到 {fixed_path}")
    
    del pipe
    torch.cuda.empty_cache()
    
    return {
        "prompts_path": prompts_path,
        "embeddings_path": embeddings_output_path,
        "uncond_path": uncond_path,
        "fixed_path": fixed_path,
        "processed_count": processed_count
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理提示词并生成embeddings")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="包含prompts的txt文件路径，每行一个prompt")
    # // ... existing code ...
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "cogvideox" / "prompts"),
                        help="输出目录")
    parser.add_argument("--model_path", type=str, default="cogvideox/CogVideoX-5b", 
                        help="CogVideoX模型路径")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="处理批次大小")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="处理设备")
    parser.add_argument("--seed", type=int, default=42, 
                        help="随机种子")
    parser.add_argument("--max_prompts", type=int, default=0, 
                        help="最多处理的prompt数量，设为0表示处理全部")
    # // ... existing code ...
    parser.add_argument("--save_separate", action="store_true", default=True,
                        help="是否将每个embedding单独保存")
    
    args = parser.parse_args()
    
    # Changed from args.input_dir to args.input_file
    results = process_prompts_to_embeddings(
        args.input_file, 
        args.output_dir,
        args.model_path,
        args.batch_size,
        args.device,
        args.seed,
        args.max_prompts,
        args.save_separate
    )
    
    if results: # Check if results is not None (in case of file read error)
        print("\n处理完成! 生成的文件:")
        for key, path in results.items():
            if path and key != "processed_count":
                print(f"- {key}: {path}")
        print(f"- 处理的embeddings总数: {results['processed_count']}")
    else:
        print("处理未成功完成.")
