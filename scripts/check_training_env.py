"""Check the training libraries and GPUs. No data or pretrained weights needed."""
import argparse
import importlib.metadata
import platform

import torch


def check_gpu(device):
    # Exercise both CUDA computation and backpropagation through a small head.
    head = torch.nn.Linear(768, 1).to(device)
    features = torch.randn(16, 768, device=device)
    targets = torch.zeros(16, 1, device=device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(head(features), targets)
    loss.backward()
    torch.cuda.synchronize(device)
    assert torch.isfinite(loss).item(), "Non-finite CUDA loss"
    assert torch.isfinite(head.weight.grad).all().item(), "Non-finite CUDA gradients"
    properties = torch.cuda.get_device_properties(device)
    print(f"{device}: {properties.name}, {properties.total_memory / 1024**3:.1f} GiB; forward/backward OK")


def check_models(device):
    import timm
    from train_probe import patch_tokens, patch_scores
    from transformers import DINOv3ViTConfig, DINOv3ViTModel, SiglipVisionConfig, SiglipVisionModel

    # Random weights only: verify architecture/API compatibility without a download.
    # These are not pretrained outputs, accuracy checks, or speed benchmarks.
    for name in ("deit3", "dinov3", "siglip2"):
        if name == "deit3":
            model = timm.create_model("deit3_base_patch16_224", pretrained=False, num_classes=0)
        elif name == "dinov3":
            model = DINOv3ViTModel(DINOv3ViTConfig(
                hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                num_attention_heads=12, image_size=224, patch_size=16,
                num_register_tokens=4,
            ))
        else:
            model = SiglipVisionModel(SiglipVisionConfig(
                hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                num_attention_heads=12, image_size=224, patch_size=16,
            ))
            model.use_head = False
        model = model.to(device).eval().requires_grad_(False)
        images = torch.randn(2, 3, 224, 224, device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            tokens = patch_tokens(model, name, images).float()
            features = tokens.mean(dim=1)
        patch_scores(tokens, torch.linspace(-0.1, 0.1, 768, device=device), 0.25)
        torch.cuda.synchronize(device)
        assert tuple(features.shape) == (2, 768), (name, features.shape)
        assert torch.isfinite(features).all().item(), f"Non-finite {name} features"
        parameters = sum(p.numel() for p in model.parameters())
        print(f"{name}: {parameters / 1e6:.1f}M parameters; patches {tuple(tokens.shape)}, mean features {tuple(features.shape)}, XAI identity OK (random weights)")
        del model, images, features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true", help="Also test random-weight Base/16 encoders on one GPU.")
    parser.add_argument("--device", type=int, default=0, help="GPU index for optional encoder checks (default: 0).")
    args = parser.parse_args()
    print(f"Python: {platform.python_version()}")
    for package in ("torch", "torchvision", "timm", "transformers", "scikit-learn", "huggingface-hub"):
        print(f"{package}: {importlib.metadata.version(package)}")
    # Import the actual model classes as well as the top-level libraries.
    import torchvision
    import timm
    import sklearn.metrics
    from transformers import DINOv3ViTModel, SiglipVisionModel

    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Install the GPU wheels and check the NVIDIA driver.")
    count = torch.cuda.device_count()
    if not 0 <= args.device < count:
        parser.error(f"--device must be between 0 and {count - 1}")
    for index in range(count):
        check_gpu(torch.device(f"cuda:{index}"))
    if args.models:
        check_models(torch.device(f"cuda:{args.device}"))
    print("Training environment checks passed.")


if __name__ == "__main__":
    main()
