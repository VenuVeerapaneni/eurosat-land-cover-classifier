"""Predict a EuroSAT category and save a Grad-CAM overlay."""
import argparse
import os
from pathlib import Path
import json
_mpl_dir=Path(__file__).resolve().parent/"outputs"/"mplconfig"
_mpl_dir.mkdir(parents=True,exist_ok=True)
os.environ["MPLCONFIGDIR"]=str(_mpl_dir)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from train import CLASSES, LandCoverCNN

NORM=transforms.ToTensor()

def main():
    p=argparse.ArgumentParser(); p.add_argument("image"); p.add_argument("--checkpoint",default="outputs/best_model.pt"); p.add_argument("--output",default="outputs/prediction_gradcam.png"); args=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); ckpt=torch.load(args.checkpoint,map_location=device,weights_only=True); model=LandCoverCNN(len(CLASSES)).to(device); model.load_state_dict(ckpt["model_state"]); model.eval()
    with Image.open(args.image) as im: original=im.convert("RGB").resize((64,64)); x=NORM(original).unsqueeze(0).to(device)
    feature_map=[]; gradient=[]; layer=model.features[-1].net[3]
    def capture(_module, _inputs, output):
        feature_map.append(output)
        output.register_hook(lambda grad: gradient.append(grad))
    layer.register_forward_hook(capture)
    logits=model(x); probs=logits.softmax(1)[0]; pred=int(probs.argmax()); model.zero_grad(set_to_none=True); logits[0,pred].backward()
    weights=gradient[0].mean(dim=(2,3),keepdim=True); cam=(weights*feature_map[0]).sum(1,keepdim=True).relu(); cam=torch.nn.functional.interpolate(cam,size=(64,64),mode="bilinear",align_corners=False)[0,0].detach().cpu().numpy(); cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); rgb=np.asarray(original)/255.; fig,ax=plt.subplots(figsize=(5,5)); ax.imshow(rgb); ax.imshow(cam,cmap="jet",alpha=.38); ax.axis("off"); ax.set_title(f"{CLASSES[pred]} | confidence {float(probs[pred].detach()):.1%}"); fig.tight_layout(); fig.savefig(out,dpi=160); plt.close(fig)
    print(json.dumps({"class":CLASSES[pred],"confidence":float(probs[pred].detach()),"gradcam":str(out)},indent=2))

if __name__=="__main__": main()
