#%%
import os
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from utils import fftconv2d
import matplotlib.pyplot as plt

def make_circ(entire_pixels, radius_pixels, sigma=0.1):
    x = np.linspace(-(entire_pixels-1)/2, (entire_pixels-1)/2, entire_pixels, endpoint=True)
    x, y = np.meshgrid(x, x)

    circ = (1.0*np.sqrt(x ** 2 + y ** 2) <= radius_pixels).astype(np.float32)
    return gaussian_filter(circ, sigma=sigma)

def load_image(path):
    data = np.array(Image.open(path), dtype=np.float32)
    return data # (H, W) [0, 255]


if __name__ == '__main__':
    path = 'examples/0011.png'
    save_path = 'util_verify_results'
    img = load_image(path)
    img = img / 255 * 2 - 1
    N = 300
    psf = make_circ(N, 1, sigma=3)
    
    img_torch = torch.from_numpy(img).permute(2, 0, 1)
    psf_torch = torch.from_numpy(psf)
    
    
    torch_result = fftconv2d(img_torch, psf_torch, rfft=False).permute(1, 2, 0)
    
    rfft_torch_result = fftconv2d(img_torch, psf_torch, rfft=True, roll=True).permute(1, 2, 0)
    unroll_rfft_torch_result = fftconv2d(img_torch, psf_torch, rfft=True).permute(1, 2, 0)
    
    
    result = torch_result.detach().numpy()
    
    plt.imshow(img)
    plt.title('Clean Image')
    plt.savefig(os.path.join(save_path, 'clean_img.png'))
    plt.show()
    
    plt.clf()
    plt.imshow(psf)
    plt.title('PSF')
    plt.savefig(os.path.join(save_path, 'psf_img.png'))
    plt.show()
    
    plt.clf()
    plt.imshow(result.real / result.real.max())
    plt.title('Blurred Image')
    plt.savefig(os.path.join(save_path, 'deg_img.png'))
    plt.show()
    
    
    