import torch


def get_device() -> torch.device:
    """Return best available device (cuda if available, otherwise cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_hardware_info() -> torch.device:
    """Print runtime hardware detection — call at start of every run.
    
    Returns:
        torch.device: The detected device.
    """
    device = get_device()
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"VRAM: {vram:.1f} GB")
    else:
        print("Running on CPU")
    return device
