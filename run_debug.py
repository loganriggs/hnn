import torch
import gc

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print("GPU cache cleared")
    print(f"GPU Memory free: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

# Now run the training
import subprocess
subprocess.run(["python", "transcoding.py"])
