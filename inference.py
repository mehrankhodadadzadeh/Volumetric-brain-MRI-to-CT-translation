# MRI → CT inference with HU outputs and [0, 1] metrics
import os, numpy as np, nibabel as nib, torch
from tqdm import tqdm
from models import build_generator
from utils import mae_psnr_ssim

# ───────── USER CONFIG ──────────────────────────────────────────── #
MODEL_PATH  = 
TEST_DIR    = 
OUTPUT_DIR  = 
GENERATOR   = 
PATCH_SIZE  =
BASE_CH     =
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ────────────────────────────────────────────────────────────────── #

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------- build generator -------------------------------- #
g_kw = dict(in_channels=1, out_channels=1)
if GENERATOR == "unet3d":
    g_kw["base_channels"] = BASE_CH
elif GENERATOR == "swin_unetr":
    g_kw["img_size"] = PATCH_SIZE

net = build_generator(GENERATOR, **g_kw).to(DEVICE)
net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
net.eval()

# ---------------- inference loop --------------------------------- #
mae_all, psnr_all, ssim_all = [], [], []
patients = [p for p in sorted(os.listdir(TEST_DIR)) if os.path.isdir(os.path.join(TEST_DIR, p))]

for pid in tqdm(patients, desc="Patients"):
    p_dir = os.path.join(TEST_DIR, pid)

    mri_nii  = nib.load(os.path.join(p_dir, "mr.nii.gz"))
    ct_nii   = nib.load(os.path.join(p_dir, "ct.nii.gz"))
    mask_nii = nib.load(os.path.join(p_dir, "mask.nii.gz"))

    mri   = mri_nii.get_fdata().astype(np.float32)
    ct    = ct_nii.get_fdata().astype(np.float32)
    mask  = mask_nii.get_fdata().astype(bool)
    affine = mri_nii.affine

    # --- Patch-wise prediction ---------------------------------- #
    mri_pad, orig_shape = pad_to_mult(mri_z, PATCH_SIZE[0])
    patches = []
    for pos, patch in split(mri_pad, PATCH_SIZE):
        with torch.no_grad():
            pred = net(torch.from_numpy(patch[None, None]).to(DEVICE))
        patches.append((pos, pred.cpu().squeeze().numpy()))


    # --- Save prediction in HU scale ---------------------------- #
    nib.save(nib.Nifti1Image(ct_pred_hu.astype(np.float32), affine),
             os.path.join(OUTPUT_DIR, f"synth_ct_hu_{pid}.nii.gz"))

    # --- Normalize GT CT to for metric comparison -------- #


    # --- Compute MAE, PSNR, SSIM on normalized [0, 1] scale ----- #
    mae, psnr, ssim = mae_psnr_ssim(ct_pred_norm, ct_gt_norm, mask)
    print(f"{pid:>8}  MAE={mae:.4f} | PSNR={psnr:.2f} dB | SSIM={ssim:.4f}")
    mae_all.append(mae); psnr_all.append(psnr); ssim_all.append(ssim)

# ---------------- Summary ---------------------------------------- #
print("\nFINAL → MAE %.4f | PSNR %.2f dB | SSIM %.4f"
      % (np.mean(mae_all), np.mean(psnr_all), np.mean(ssim_all)))
