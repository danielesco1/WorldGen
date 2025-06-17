from huggingface_hub import snapshot_download

# 1) Download the LoRA weights repo
lora_dir = snapshot_download(
    repo_id="LeoXie/WorldGen",
    repo_type="model",           # default, but explicit
    local_dir="models/WorldGen-Lora",
    local_dir_use_symlinks=False # copy all files
)

# 2) Download the transformer repo (e.g. for fp16 precision)
transformer_dir = snapshot_download(
    repo_id=f"mit-han-lab/svdq-fp16-flux.1-fill-dev",
    local_dir="models/svdq-fp16-flux",
    local_dir_use_symlinks=False
)

# 3) Download the FLUX Fill pipeline repo
pipe_dir = snapshot_download(
    repo_id="black-forest-labs/FLUX.1-Fill-dev",
    local_dir="models/FLUX.1-Fill-dev",
    local_dir_use_symlinks=False
)

print("Downloaded to:")
print("  LoRA ➞", lora_dir)
print("  Transformer ➞", transformer_dir)
print("  Pipeline ➞", pipe_dir)