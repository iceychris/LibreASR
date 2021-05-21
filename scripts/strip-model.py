import argparse
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove optimizer state from a model (inplace)"
    )
    parser.add_argument("path", help="Path to .pth model file")
    args = parser.parse_args()

    d = torch.load(args.path, map_location="cpu")
    keys = list(d.keys())
    if "opt" in keys and "model" in keys:
        d = torch.save(d["model"], args.path)
        print(f"Model saved at {args.path}")
    else:
        print("Error: Model needs to contain 'opt' and 'model' keys")
