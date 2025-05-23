# o1 = vae_mask.img_to_idxBl(mask_B1HW) #list
# o1 = vae_mask.quantize.idxBl_to_var_input(o1)
# o = var(o1, inp_B3HW)

# o = var(torch.rand(1, 679, 64).cuda(), torch.rand(1, 3, 256, 256).cuda())
# o = var.autoregressive_infer_cfg(torch.rand(1,3,256,256).cuda())
# print(o.shape)

import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
import sys
sys.path.append("../")

from models import VQVAE, build_vae_var

from vqvae.vqvae_utils import display_results
from var_utils import build_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

def test_var(var, test_loader, indices=None, vae_mask=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_images = []
    
    with torch.no_grad():
        for i, (img, mask) in tqdm(enumerate(test_loader), desc="Testing"):
            if indices and i not in indices: continue  # Skip images not in the selected indices
            
            img = img.to(device)
            rec_img = var.autoregressive_infer_cfg(img, top_k=100, top_p=0.95, more_smooth=False)
            if vae_mask:
                lst = vae_mask.img_to_idxBl(mask.to(device))
                for i in range(5): print(lst[i])
                
            if indices:
                # print(img.shape)
                # original = transforms.ToPILImage()(img.squeeze(0).cpu())
                # reconstructed = transforms.ToPILImage()(rec_img.squeeze(0).cpu())
                original = transforms.ToPILImage()(img[0].cpu())
                reconstructed = transforms.ToPILImage()(rec_img[0].cpu())
                selected_images.append((original, reconstructed))
                if len(selected_images) == len(indices): break

    # Display the images in an 8x2 grid
    display_results(selected_images, save_path="var.png")
    if indices: return selected_images

# Main script
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_dataset, val_dataset = build_dataset("/media/viplab/DATADRIVE1/skin_lesion/ISIC2018/")

    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # Load one image at a time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_img, vae_mask, var = build_vae_var(
        V=256, Cvae=64, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        depth=16, shared_aln=False,
    )
    vae_img.load_state_dict(torch.load("/home/viplab/SuperRes/newvar/vqvae/checkpoints/img_best.pth", map_location=device))
    vae_mask.load_state_dict(torch.load("/home/viplab/SuperRes/newvar/vqvae/checkpoints/mask_best.pth", map_location=device))
    var.load_state_dict(torch.load("/home/viplab/SuperRes/newvar/var/checkpoints/var_best.pth", map_location=device))

    test_indices = [0, 5, ]#10, 18, 20, 25, 30, 40]  # Example indices (max 8)
    test_var(var, test_loader, indices=test_indices, vae_mask=vae_mask)