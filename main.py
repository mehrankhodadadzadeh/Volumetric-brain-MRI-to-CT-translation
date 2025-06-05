import os, random
import torch, numpy as np, wandb
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, DataLoader, Subset

from dataset import get_dataloaders, PairedNiftiDataset
from models   import build_generator, Discriminator3D
from trainer  import GANTrainer

#PROJECT_NAME  =
#ENTITY        =  
#WANDB_API_KEY =
#wandb.login(key=WANDB_API_KEY)


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(cfg, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- build generator ----------------------------------------
    key = cfg["generator"].lower()
    gen_kwargs = dict(in_channels=1, out_channels=1)

    # width parameter for the U‑Net family only
    if key == "unet3d":
        gen_kwargs["base_channels"] = cfg.get("base_channels", 32)

    # Swin‑UNETR needs the patch size
    if key == "swin_unetr":
        gen_kwargs["img_size"] = cfg["patch_size"]

    generator     = build_generator(key, **gen_kwargs).to(device)
    discriminator = Discriminator3D(in_channels=1).to(device)

    trainer = GANTrainer(
        generator      = generator,
        discriminator  = discriminator,
        train_loader   = train_loader,
        val_loader     = val_loader,
        device         = device,
        lr             = cfg["learning_rate"],
        beta1          = cfg["adam_beta1"],
        beta2          = cfg["adam_beta2"],
        checkpoint_dir = cfg["checkpoint_dir"],
    )
   
    return trainer.train(epochs=cfg["epochs"], return_metrics=True)

# ─────────────────────── k‑fold cross‑validation-optional ─────────────────────

def cross_validate(cfg, n_splits: int = 5):
    tr_ds = PairedNiftiDataset(cfg["train_data_dir"], cfg["patch_size"], "train", augment=True)
    vl_ds = PairedNiftiDataset(cfg["val_data_dir"],   cfg["patch_size"], "val",   augment=False)
    full  = ConcatDataset([tr_ds, vl_ds])

    maes, psnrs, ssims = [], [], []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(full), 1):
        print(f"Fold {fold}/{n_splits}")
        tl = DataLoader(Subset(full, tr_idx), batch_size=cfg["batch_size"], shuffle=True)
        vl = DataLoader(Subset(full, vl_idx), batch_size=cfg["batch_size"], shuffle=False)
        m, p, s = train_one(cfg, tl, vl)
        maes.append(m); psnrs.append(p); ssims.append(s)

    return np.mean(maes), np.mean(psnrs), np.mean(ssims)

# ─────────────────────────────── main ────────────────────────────────

def main():
    cfg = dict(
        # experiment set‑up
        seed           = 0,
        mode           = "simple",          # "simple" or "crossval"
        generator      = "attention_unet",         # |"unet3d" |"attention_unet"| "swin_unetr" |

        # paths 
        train_data_dir = 
        val_data_dir   = 
        test_data_dir  = 
        checkpoint_dir = 
        results_dir    = 

        # hyper‑parameters
        patch_size     = 
        base_channels  = 
        learning_rate  = 
        adam_beta1     =
        adam_beta2     = 
        batch_size     = 
        epochs         = 
    )

    wandb.init(project=PROJECT_NAME, entity=ENTITY, config=cfg, name=cfg["mode"])
    set_seed(cfg["seed"])

    if cfg["mode"] == "simple":
        tl, vl, _ = get_dataloaders(
            cfg["train_data_dir"], cfg["val_data_dir"], cfg["test_data_dir"],
            batch_size=cfg["batch_size"], patch_size=cfg["patch_size"]
        )
        train_one(cfg, tl, vl)

    elif cfg["mode"] == "crossval":
        m, p, s = cross_validate(cfg, n_splits=5)
        print(f"CV average → MAE {m:.4f} | PSNR {p:.2f} | SSIM {s:.4f}")

    else:
        raise ValueError("cfg['mode'] must be 'simple' or 'crossval'")


if __name__ == "__main__":
    main()
