"""Predict a EuroSAT category and save a Grad-CAM overlay."""
import argparse
import os
from pathlib import Path
import json

_mpl_dir = Path(__file__).resolve().parent / "outputs" / "mplconfig"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from train import CLASSES, LandCoverCNN

NORM = transforms.ToTensor()


def load_model(checkpoint="outputs/best_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = LandCoverCNN(len(CLASSES)).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, device


def predict_image(image, model, device):
    """Return resized RGB input, predicted label, confidence, and normalized Grad-CAM."""
    original = image.convert("RGB").resize((64, 64))
    x = NORM(original).unsqueeze(0).to(device)
    feature_map, gradients = [], []
    layer = model.features[-1].net[3]

    def capture(_module, _inputs, output):
        feature_map.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    handle = layer.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        probs = logits.softmax(1)[0]
        predicted = int(probs.argmax())
        logits[0, predicted].backward()
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = (weights * feature_map[0]).sum(1, keepdim=True).relu()
        cam = torch.nn.functional.interpolate(
            cam, size=(64, 64), mode="bilinear", align_corners=False
        )[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return original, CLASSES[predicted], float(probs[predicted].detach()), cam
    finally:
        handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--output", default="outputs/prediction_gradcam.png")
    args = parser.parse_args()

    model, device = load_model(args.checkpoint)
    with Image.open(args.image) as image:
        original, label, confidence, cam = predict_image(image, model, device)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(original) / 255.0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb)
    ax.imshow(cam, cmap="jet", alpha=0.38)
    ax.axis("off")
    ax.set_title(f"{label} | confidence {confidence:.1%}")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(json.dumps({"class": label, "confidence": confidence, "gradcam": str(out)}, indent=2))


if __name__ == "__main__":
    main()
