import torch
import gc

def clear_vram():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    print("VRAM cleared.")