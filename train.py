"""Train a compact CNN on EuroSAT RGB and save transparent evaluation artifacts."""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
from pathlib import Path

_mpl_dir = Path(__file__).resolve().parent / "outputs" / "mplconfig"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_mpl_dir)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
IMAGE_EXT = {".jpg", ".jpeg", ".png"}

class PatchDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        # Satellite scenes have no fixed north-up orientation; flips preserve class labels.
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        self.transform = transforms.Compose((aug if augment else []) + [transforms.ToTensor()])
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as im: x = self.transform(im.convert("RGB"))
        return x, label

class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.BatchNorm2d(co),nn.ReLU(inplace=True),nn.Conv2d(co,co,3,padding=1,bias=False),nn.BatchNorm2d(co),nn.ReLU(inplace=True),nn.MaxPool2d(2))
    def forward(self,x): return self.net(x)

class LandCoverCNN(nn.Module):
    """Small CNN written for 64x64 RGB EuroSAT patches; no pretrained weights."""
    def __init__(self, n_classes=10):
        super().__init__(); self.stem=nn.Sequential(nn.Conv2d(3,24,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(24),nn.ReLU(inplace=True)); self.features=nn.Sequential(ConvBlock(24,32),ConvBlock(32,64),ConvBlock(64,128)); self.pool=nn.AdaptiveAvgPool2d(1); self.classifier=nn.Sequential(nn.Flatten(),nn.Dropout(.25),nn.Linear(128, n_classes))
    def forward(self,x): return self.classifier(self.pool(self.features(self.stem(x))))

def collect(data_dir):
    samples=[]
    for label,name in enumerate(CLASSES):
        folder=data_dir/name
        if not folder.is_dir(): raise FileNotFoundError(f"Missing class folder: {folder}. Run download_data.py first.")
        samples.extend((p,label) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXT)
    if not samples: raise RuntimeError(f"No images found under {data_dir}")
    return samples

def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]; losses=[]; criterion=nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x,y in loader:
            x=x.to(device); y=y.to(device); logits=model(x); losses.append(float(criterion(logits,y))); ps.extend(logits.argmax(1).cpu().tolist()); ys.extend(y.cpu().tolist())
    return np.asarray(ys),np.asarray(ps),float(np.mean(losses))

def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    data_dir=Path(args.data).resolve(); out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    all_samples=collect(data_dir); labels=np.asarray([y for _,y in all_samples]); idx=np.arange(len(all_samples))
    train_i, remainder=train_test_split(idx,test_size=.30,random_state=args.seed,stratify=labels)
    val_i,test_i=train_test_split(remainder,test_size=.50,random_state=args.seed,stratify=labels[remainder])
    splits={"train":train_i,"validation":val_i,"test":test_i}
    used_train=train_i
    if args.train_per_class:
        rng=np.random.default_rng(args.seed); chosen=[]
        for cls in range(len(CLASSES)):
            class_indices=train_i[labels[train_i]==cls]
            chosen.extend(rng.choice(class_indices,size=min(args.train_per_class,len(class_indices)),replace=False).tolist())
        used_train=np.asarray(sorted(chosen))
    selected={**splits,"train":used_train}
    fit_indices=set(used_train.tolist())
    with (out/"split_manifest.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["path","class","split","used_for_fit"])
        for split,indices in splits.items():
            for i in indices:
                p,y=all_samples[i]; w.writerow([str(p.relative_to(data_dir)),CLASSES[y],split,split!="train" or i in fit_indices])
    datasets={s:PatchDataset([all_samples[i] for i in indices],augment=(s=="train")) for s,indices in selected.items()}
    loaders={s:DataLoader(ds,batch_size=args.batch_size,shuffle=(s=="train"),num_workers=0,pin_memory=torch.cuda.is_available()) for s,ds in datasets.items()}
    device=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model=LandCoverCNN(len(CLASSES)).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); criterion=nn.CrossEntropyLoss()
    best=-1; history=[]; best_path=out/"best_model.pt"
    print(f"Device: {device}; images: {len(all_samples)}; split pool: {len(train_i)}/{len(val_i)}/{len(test_i)}; train images used: {len(used_train)}; parameters: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1,args.epochs+1):
        model.train(); running=0.; seen=0
        for x,y in loaders["train"]:
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True); logits=model(x); loss=criterion(logits,y); loss.backward(); opt.step(); running+=float(loss.detach())*len(y); seen+=len(y)
        vy,vp,vl=evaluate(model,loaders["validation"],device); va=float(accuracy_score(vy,vp)); vf=float(f1_score(vy,vp,average="macro")); record={"epoch":epoch,"train_loss":running/seen,"val_loss":vl,"val_accuracy":va,"val_macro_f1":vf}; history.append(record); print(json.dumps(record))
        if vf>best:
            best=vf; torch.save({"model_state":model.cpu().state_dict(),"classes":CLASSES,"seed":args.seed,"epoch":epoch},best_path); model.to(device)
    with (out/"history.json").open("w",encoding="utf-8") as f: json.dump(history,f,indent=2)
    state=torch.load(best_path,map_location=device,weights_only=True); model.load_state_dict(state["model_state"])
    ty,tp,tl=evaluate(model,loaders["test"],device)
    report=classification_report(ty,tp,target_names=CLASSES,output_dict=True,zero_division=0)
    metrics={"test_accuracy":float(accuracy_score(ty,tp)),"test_macro_f1":float(f1_score(ty,tp,average="macro")),"test_loss":tl,"test_samples":len(ty),"train_pool_samples":len(train_i),"train_samples_used":len(used_train),"validation_samples":len(val_i),"best_epoch":state["epoch"],"seed":args.seed,"device":str(device),"epochs_requested":args.epochs,"train_per_class_cap":args.train_per_class,"batch_size":args.batch_size,"learning_rate":args.lr,"weight_decay":1e-4,"threads":args.threads,"parameters":sum(p.numel() for p in model.parameters()),"classification_report":report}
    with (out/"metrics.json").open("w",encoding="utf-8") as f: json.dump(metrics,f,indent=2)
    cm=confusion_matrix(ty,tp,labels=np.arange(len(CLASSES)))
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(cm,cmap="Blues"); ax.set(xticks=np.arange(10),yticks=np.arange(10),xticklabels=CLASSES,yticklabels=CLASSES,xlabel="Predicted",ylabel="True",title="EuroSAT test confusion matrix"); plt.setp(ax.get_xticklabels(),rotation=45,ha="right"); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(out/"confusion_matrix.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot([h["epoch"] for h in history],[h["train_loss"] for h in history],label="train loss"); ax.plot([h["epoch"] for h in history],[h["val_loss"] for h in history],label="validation loss"); ax.set(xlabel="Epoch",ylabel="Cross-entropy",title="Training history"); ax.legend(); fig.tight_layout(); fig.savefig(out/"training_curve.png",dpi=160); plt.close(fig)
    print(json.dumps({k:metrics[k] for k in ("test_accuracy","test_macro_f1","test_samples","best_epoch","device")},indent=2))
    return model,device,all_samples,splits

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/EuroSAT_RGB"); p.add_argument("--output",default="outputs"); p.add_argument("--epochs",type=int,default=8); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--seed",type=int,default=42); p.add_argument("--threads",type=int,default=4); p.add_argument("--train-per-class",type=int,default=0,help="Optional balanced cap per class for a quick local run; 0 uses every training image."); p.add_argument("--cpu",action="store_true"); args=p.parse_args(); train(args)
if __name__=="__main__": main()
