# ===============================================================


import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
from monai.metrics import SSIMMetric


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GANTrainer:
    """
    Generic 3-D GAN trainer (patch discriminator, L1 + GAN loss).
    """

    def __init__(
        self,
        generator,
        discriminator,
        train_loader,
        val_loader,
        device,
        lr,
        beta1,
        beta2,
        checkpoint_dir,
    ):
        self.gen = generator.to(device)
        self.disc = discriminator.to(device)
        self.tl = train_loader
        self.vl = val_loader
        self.dev = device
        self.ckpt = checkpoint_dir

        self.opt_g = optim.Adam(self.gen.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_d = optim.Adam(self.disc.parameters(), lr=lr, betas=(beta1, beta2))

        self.ssim = SSIMMetric(data_range=1.0, spatial_dims=3, reduction="mean")
        os.makedirs(self.ckpt, exist_ok=True)

        wandb.log({
            "model/generator_params": count_parameters(self.gen),
            "model/discriminator_params": count_parameters(self.disc),
        })

        self._train_losses_g, self._train_losses_d, self._val_losses, self._epochs = [], [], [], []

    def train(self, epochs, return_metrics=False):
        best_val = float("inf")
        best_mae = best_psnr = best_ssim = None

        for ep in range(1, epochs + 1):
            self.gen.train()
            self.disc.train()
            g_losses, d_losses = [], []

            for src, tgt, _ in tqdm(self.tl, desc=f"Epoch {ep} [Train]"):
                src, tgt = src.to(self.dev), tgt.to(self.dev)

                real_labels = torch.ones_like(self.disc(tgt))
                fake_labels = torch.zeros_like(real_labels)

                # --- discriminator step ---
                fake = self.gen(src).detach()
                real_pred = self.disc(tgt)
                fake_pred = self.disc(fake)

                d_loss_real = self.bce(real_pred, real_labels)
                d_loss_fake = self.bce(fake_pred, fake_labels)
                d_loss = (d_loss_real + d_loss_fake) / 2

                self.opt_d.zero_grad()
                d_loss.backward()
                self.opt_d.step()
                d_losses.append(d_loss.item())

                # --- generator step ---
                fake = self.gen(src)
                pred_fake = self.disc(fake)
                gan_loss = self.bce(pred_fake, real_labels)
                l1_loss = self.l1(fake, tgt)
                g_loss = gan_loss + 100 * l1_loss

                self.opt_g.zero_grad()
                g_loss.backward()
                self.opt_g.step()
                g_losses.append(g_loss.item())

            val_loss, mae, psnr, ssim = self._validate()
            self.sched.step(val_loss)

            if val_loss < best_val:
                best_val, best_mae, best_psnr, best_ssim = val_loss, mae, psnr, ssim
                torch.save(self.gen.state_dict(), os.path.join(self.ckpt, "best_generator.pth"))

            if ep % 500 == 0:
                torch.save(self.gen.state_dict(), os.path.join(self.ckpt, f"generator_epoch_{ep}.pth"))

            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)

            wandb.log({
                "loss/train_gen": avg_g_loss,
                "loss/train_disc": avg_d_loss,
                "loss/val": val_loss,
                "metrics/mae": mae,
                "metrics/psnr": psnr,
                "metrics/ssim": ssim,
                "epoch": ep,
                "lr": self.opt_g.param_groups[0]["lr"],
            })

            self._train_losses_g.append(avg_g_loss)
            self._train_losses_d.append(avg_d_loss)
            self._val_losses.append(val_loss)
            self._epochs.append(ep)

            print(
                f"Epoch {ep}: train_g={avg_g_loss:.4f} | train_d={avg_d_loss:.4f} | "
                f"val={val_loss:.4f} | MAE={mae:.4f} | PSNR={psnr:.2f} | SSIM={ssim:.4f}"
            )

        if return_metrics:
            return best_mae, best_psnr, best_ssim

    def _validate(self):
        self.gen.eval()
        self.ssim.reset()
        losses, maes, psnrs, ssims = [], [], [], []

        with torch.no_grad():
            for src, tgt, mask in tqdm(self.vl, desc="[Validation]"):
                src, tgt = src.to(self.dev), tgt.to(self.dev)
                mask_b = (mask > 0.5).to(self.dev)

                pred = self.gen(src)
                pred_fake = self.disc(pred)
                real_labels = torch.ones_like(pred_fake)
                gan_loss = self.bce(pred_fake, real_labels)
                l1_loss = self.l1(pred, tgt)

                losses.append((gan_loss + 100 * l1_loss).item())

                p = pred.squeeze(1).cpu().numpy()
                g = tgt.squeeze(1).cpu().numpy()
                m = mask_b.squeeze(1).cpu().numpy().astype(bool)

                if m.sum() == 0:
                    continue

                p[~m] = 0
                g[~m] = 0

                flat_p, flat_g = p[m], g[m]
                maes.append(float(np.mean(np.abs(flat_p - flat_g))))

                mse = float(np.mean((flat_p - flat_g) ** 2))
                dr = float(flat_g.max() - flat_g.min())
                if dr >= 1e-8:
                    psnr = 10 * np.log10((dr ** 2) / (mse + 1e-12))
                    psnrs.append(psnr)

                ssim_val = self.ssim(pred * mask_b, tgt * mask_b)
                if ssim_val.numel() > 1:
                    ssim_val = ssim_val.mean()
                ssims.append(ssim_val.item())

        return (
            np.mean(losses),
            np.nanmean(maes),
            np.nanmean(psnrs),
            np.nanmean(ssims),
        )
