import argparse
import math
import os
import random
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
from unified_data_loader import load_unified_dataset
from time import time
import datetime as _dt

# --------- Augmentations ----------
class RandomJPEG:
    def __init__(self, p=0.5, qmin=35, qmax=95):
        self.p, self.qmin, self.qmax = p, qmin, qmax

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        q = random.randint(self.qmin, self.qmax)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class CenterSquareCrop:
    def __call__(self, img: Image.Image):
        side = min(img.size)
        if img.height == img.width:
            return img
        return TF.center_crop(img, side)


def make_divisible(value: int, divisor: int):
    return max(divisor, int(round(value / divisor)) * divisor)


def build_train_tfm(img_size: int):
    normalize = transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
    return transforms.Compose([
        CenterSquareCrop(),
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.5), ratio=(0.8, 1.3)),
        transforms.RandomHorizontalFlip(p=0.5),
        RandomJPEG(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.40, hue=0.2),
        transforms.RandomApply([transforms.GaussianBlur(7, sigma=(0.1, 3))], p=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.2, 0.33), ratio=(0.3, 3.3), value="random"),
        normalize,
    ])


def build_val_tfm(img_size: int):
    return transforms.Compose([
        CenterSquareCrop(),
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])


# --------- Model ----------
class DinoV2Embedder(nn.Module):
    """
    DINOv2 backbone -> mean pool patch tokens -> projection -> L2 normalize
    """
    def __init__(self, backbone_name="dinov2_vitb14", embed_dim=256):
        super().__init__()
        print(f"[Info] Loading DINOv2 backbone: {backbone_name}")
        hub_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        if os.path.exists(hub_dir):
            self.backbone = torch.hub.load(hub_dir, backbone_name, source='local', pretrained=True)
        else:
            self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        print(f"load pretrained={True}, embed_dim={embed_dim}")
        in_dim = self.backbone.embed_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        out = self.backbone.forward_features(x)
        patch = out["x_norm_patchtokens"]  # [B, N, D]
        feat = patch.mean(dim=1)          # [B, D]
        z = self.proj(feat)               # [B, embed_dim]
        z = F.normalize(z, dim=1)
        return z


class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, s=32.0, m=0.30):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m

    def forward(self, z, y):
        W = F.normalize(self.W, dim=1)
        cos = F.linear(z, W)  # [B,C]
        theta = torch.acos(torch.clamp(cos, -1 + 1e-7, 1 - 1e-7))
        cos_m = torch.cos(theta + self.m)
        one_hot = F.one_hot(y, num_classes=W.size(0)).float()
        logits = self.s * (one_hot * cos_m + (1 - one_hot) * cos)
        return logits


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad = flag


def unfreeze_last_blocks_dinov2(embedder: DinoV2Embedder, n_blocks: int = 1):
    # freeze all first
    set_requires_grad(embedder.backbone, False)
    blocks = embedder.backbone.blocks
    for blk in blocks[-n_blocks:]:
        set_requires_grad(blk, True)
    # also unfreeze norms if exist
    for name in ["norm", "fc_norm"]:
        if hasattr(embedder.backbone, name):
            set_requires_grad(getattr(embedder.backbone, name), True)


@torch.no_grad()
def evaluate(embedder, head, loader, device):
    embedder.eval()
    head.eval()
    correct, total = 0, 0
    W = F.normalize(head.W, dim=1)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        z = embedder(x)
        logits = head.s * F.linear(z, W)  # no margin at eval
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Root directory of dataset")
    ap.add_argument("--train_path", type=str, default=None, help="Explicit path to training data (file or directory)")
    ap.add_argument("--val_path", type=str, default=None, help="Explicit path to validation data (file or directory)")
    ap.add_argument("--backbone", type=str, default="dinov2_vitb14")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", "-b", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--lr_head", "-lr", type=float, default=3e-4)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--arc_s", type=float, default=32.0)
    ap.add_argument("--arc_m", type=float, default=0.30)
    ap.add_argument("--stage1_epochs", type=int, default=12, help="freeze backbone epochs")
    ap.add_argument("--unfreeze_blocks", type=int, default=1, help="unfreeze last N blocks in stage2")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--output_dir", type=str)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    embedder = DinoV2Embedder(args.backbone, args.embed_dim).to(device)
    patch_size = embedder.backbone.patch_embed.patch_size[0]
    orig_img_size = args.img_size
    args.img_size = make_divisible(args.img_size, patch_size)
    if args.img_size != orig_img_size:
        print(f"[Info] Adjusted img_size {orig_img_size} -> {args.img_size} to fit patch size {patch_size}")
    print(f"[Info] Using img_size={args.img_size} (patch size {patch_size})")

    # Use unified data loader
    train_set, train_loader, val_set, val_loader = load_unified_dataset(
        data_root=args.data_root,
        train_path=args.train_path,
        val_path=args.val_path,
        train_transform=build_train_tfm(args.img_size),
        val_transform=build_val_tfm(args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        shuffle_train=True
    )
    
    # Get number of classes
    num_classes = args.num_classes
    print(f"[Info] num_classes={num_classes}, train={len(train_set)}, val={len(val_set)}")

    head = ArcFaceHead(args.embed_dim, num_classes, s=args.arc_s, m=args.arc_m).to(device)

    # Stage 1: freeze backbone
    set_requires_grad(embedder.backbone, False)
    set_requires_grad(embedder.proj, True)
    set_requires_grad(head, True)

    def build_optimizer():
        params = []
        bb_params = [p for p in embedder.backbone.parameters() if p.requires_grad]
        if bb_params:
            params.append({"params": bb_params, "lr": args.lr_backbone})
        params.append({"params": embedder.proj.parameters(), "lr": args.lr_head})
        params.append({"params": head.parameters(), "lr": args.lr_head})
        return torch.optim.AdamW(params, weight_decay=args.weight_decay)

    optimizer = build_optimizer()
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, len(train_loader))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = GradScaler(enabled=(device == "cuda"))

    best_train_acc = 0.0
    best_val_acc = 0.0
    global_step = 0

    output_dir = args.output_dir if args.output_dir else "output_dinov2_arcface_small"
    os.makedirs(output_dir, exist_ok=True)
    best_train_model_path = os.path.join(output_dir, "best_train_dinov2_arcface_small.pt")
    best_val_model_path = os.path.join(output_dir, "best_val_dinov2_arcface_small.pt")
    
    # Log file path
    log_path = os.path.join(output_dir, "results_test.txt")

    # Get classes if available (for saving models)
    classes = None
    if hasattr(train_set, 'classes'):
        classes = train_set.classes
    elif hasattr(train_set, 'dataset') and hasattr(train_set.dataset, 'classes'):
        classes = train_set.dataset.classes

    for epoch in range(1, args.epochs + 1):
        # Stage2 switch
        if epoch == args.stage1_epochs + 1:
            unfreeze_last_blocks_dinov2(embedder, n_blocks=args.unfreeze_blocks)
            set_requires_grad(embedder.proj, True)
            set_requires_grad(head, True)
            optimizer = build_optimizer()
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            print(f"[Stage2] Unfroze last {args.unfreeze_blocks} blocks.")

        embedder.train()
        head.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device == "cuda")):
                z = embedder(x)
                logits = head(z, y)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            # Calculate training accuracy
            with torch.no_grad():
                W = F.normalize(head.W, dim=1)
                eval_logits = head.s * F.linear(z, W)
                pred = eval_logits.argmax(dim=1)
                train_correct += (pred == y).sum().item()
                train_total += y.numel()

            train_loss += loss.item()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        # Calculate training accuracy
        train_acc = train_correct / max(1, train_total)
        avg_train_loss = train_loss / len(train_loader)

        # Evaluate on validation set
        val_acc = evaluate(embedder, head, val_loader, device)
        
        # Print combined accuracy log
        print(f"Epoch {epoch:3d} | train_acc = {train_acc:.5f} | val_acc = {val_acc:.5f} | train_loss = {avg_train_loss:.5f}")

        # Check and save best models
        saved_train = 0
        saved_val = 0
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            saved_train = 1
            torch.save(
                {
                    "embedder": embedder.state_dict(),
                    "head": head.state_dict(),
                    "classes": classes,
                    "args": vars(args),
                    "best_train_acc": best_train_acc,
                    "epoch": epoch,
                },
                best_train_model_path,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            saved_val = 1
            torch.save(
                {
                    "embedder": embedder.state_dict(),
                    "head": head.state_dict(),
                    "classes": classes,
                    "args": vars(args),
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                best_val_model_path,
            )

        # Log combined results to file
        with open(log_path, 'a') as f:
            timestamp = _dt.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M')
            f.write(
                f'[{timestamp}] Iteration {epoch:3d} | train_acc = {train_acc:.5f} | val_acc = {val_acc:.5f} | '
                f'train_loss = {avg_train_loss:.5f} | saved_train = {saved_train} | saved_val = {saved_val}\n'
            )

    print(f"[Done] best_train_acc={best_train_acc:.4f}, best_val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
