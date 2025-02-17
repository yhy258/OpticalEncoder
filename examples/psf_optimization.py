import os, sys, math, datetime, glob, faulthandler
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from system4f import Pupil4FLensSystem
from optical_elements import *

import logging

logging.basicConfig(filename='out.log', level=logging.DEBUG, format="%(message)s")

learning_rate = 1e-4
pm_learning_rate = 1e2 * learning_rate # phase mask learning rate
regularize_lost_power = 0
num_iterations = 1000000


# setup the simulation parameters, make a 4f system.
num_pixels = 2560
pixel_size = 0.325
num_systems = 8 # for parallelization
num_planes = 256 
downsample = 5
plane_subsample = 4
psf_pad = 1280
taper_width = 5
regularize_power = False
devices = [torch.device(f'cuda:{i}') for i in range(num_planes)]

# define SLM Parameters
circle_NA = 0.8
pupil_NA = 0.8
wavelength = 0.532
fNA = circle_NA / wavelength
slm_radius_pixels = 678.374 * circle_NA
slm_num_pixels = int(slm_radius_pixels * 2) + 1
"""
The difference between dk in slm and dk in Fourier plane (4f lens system)
1. slm's dk : slm plays a role in modulating phase. -> the maximum frequency it can modulate is NA/lambda.
    The interval is determined by the resolution of slm (the number of pixels)
    slm_dk = fNA / slm_radius_pixels
    ; 즉, SLM은 phase를 modulating 하기 위한 장치임. 그리고 SLM은 Fourier space에 위치하여있고, spatial frequency component를 조절함.
    ; 시스템의 parameter에 대해 maximum spatial frequency가 정해짐.
    ; 즉, SLM이 Fourier space에 위치해있을떄, SLM pixels의 infinitesimal은 maximum_freq / num_pixels로 정해지는거임.
2. dk in Fourier domain : it is actually the Fourier space corresponding to physical space
    The frequency is highly related to the sampling rate in physical space (inverse relationship)
    The infinitesimal in Fourier space would be 1 / (pixel_size * num_pixels)
    ; pixel * num_pixels로 정의된 공간이 있다면, 존재할 수 있는 가장 큰 sinusoidal function의 period는 pixel * num_pixels.
    ; 즉, 가장 낮은 frequency는 1/pixel_size * num_pixels가 되는거임.
"""
slm_dk = fNA / slm_radius_pixels
slm_k = (
    np.linspace(
        -slm_num_pixels//2 +1,
        slm_num_pixels//2 +1,
        num=slm_num_pixels,
        endpoint=True,
        dtype=np.float32
    ) * slm_dk
)
slm_k = torch.from_numpy(slm_k)
slm_kx, slm_ky = torch.meshgrid(slm_k, slm_k)

# calculate downsampled sizes
subsampled_num_planes = int(num_planes / plane_subsample)
downsampled_num_pixels = int(num_pixels / downsample)
downsampled_radius = int(0.5 * 386 / pixel_size / downsample) # maybe, 386 is diameter, and downsampled_radius is the number of pixels for radius

# set grad sizes
# TODO: 이게 뭐지.
num_grad_im_plnaes = 5
num_grad_recon_planes = 5

# create chunked defocus ranges
defocus_range = np.round(np.linspace(-125, 125, num=subsampled_num_planes))
chunk_size = int(len(defocus_range) / num_systems) # the size of allocated depth per an optical system instance.
if len(defocus_range) % num_systems != 0:
    chunk_size += 1 
sections = [i * chunk_size for i in range(1, num_systems)]
defocus_ranges = np.split(defocus_range, sections)

# create 4fsystem param dicts
param_dict = dict(
    wavelength = [wavelength],
    ratios = [1.0],
    NA = pupil_NA,
    ref_idx = 1.33,
    pixel_size = pixel_size,
    num_pixels = num_pixels,
    pad = psf_pad,
    taper_wdith = taper_width,
    downsample = downsample
)

param_dicts = [
    dict(list(param_dict.items()) + [('device', device)]) for device in devices
]

## TODO: augmentation info


def create_4fsystems():
    opts = [Pupil4FLensSystem(**param_dict) for param_dict in param_dicts]
    opts = nn.ModuleList(opts)
    return opts

def create_phase_mask(kx, ky, phase_mask_init=None):
    if phase_mask_init is None:
        # TODO: defocusing + phase ramping. -> Maybe I should study this.
        defocused_ramps = DefocusedRamps(
            kx, ky, pupil_NA / wavelength, 1.33, wavelength, delta=2374.0
        )
        phase_mask_init = defocused_ramps()
    pixels = Pixels(kx, ky, pixels=phase_mask_init)
    return pixels

def create_reconstruction_networks():
    pass


def create_placeholder_reconstruction_networks():
    pass


def initalize_microscope_reconstruction(latest=None, phase_mask_init=None):
    opts = create_4fsystems()
    pixels = create_phase_mask(opts[0].kx, opts[0].ky, phase_mask_init=phase_mask_init)
    deconvs = create_reconstruction_networks()
    placeholder_deconvs = create_placeholder_reconstruction_networks()
    optdeconv = nn.ModuleDict(
        {
            "opts": opts,
            "deconvs": deconvs,
            "placeholder_deconvs": placeholder_deconvs,
            "phase_mask": pixels
        }
    )
    if latest is not None:
        print("[info] loading from checkpoint")
        optdeconv.load_state_dict(latest["optdeconv_state_dict"], strict=True)
    return optdeconv
