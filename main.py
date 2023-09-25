import sys
sys.path.append("./taming-transformers")
sys.path.append("./CLIP")
sys.path.append("./image-background-remove-tool")
# sys.path.append("./zero123")
sys.path.append("./zero123/zero123")

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import math
import os
import numpy as np
import time
import torch
from contextlib import nullcontext
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess
from lovely_numpy import lo
import omegaconf
from PIL import Image
from torch import autocast
from torchvision import transforms

from ldm.models.diffusion.ddpm import LatentDiffusion

from scripts.utils import convert_to_rgb

_GPU_INDEX = 0

def load_zero123_ld(
        state_dict_path='./checkpoints',
        config_path='./config/latent_diffusion.yml',
        **kwargs
):
    config = omegaconf.OmegaConf.load(config_path)
    with init_empty_weights():
        model = LatentDiffusion(**config['model']['params'])
    model = load_checkpoint_and_dispatch(model, state_dict_path, **kwargs)
    return model

@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z, verbose=False):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            if verbose:
                print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

def preprocess_image(models, input_im, preprocess, verbose=False):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''
    if verbose:
        print('old input_im:', input_im.size)
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].

    else:
        input_im = input_im.resize([256, 256], Image.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    if verbose:
        print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
        print('new input_im:', lo(input_im))

    return input_im

def main_run(models, device,
             x=0.0, y=0.0, z=0.0,
             raw_im=None, preprocess=True,
             scale=3.0, n_samples=4, ddim_steps=50, ddim_eta=1.0,
             precision='fp32', h=256, w=256,
             verbose=False):
    '''
    :param raw_im (PIL Image).
    '''
    torch.cuda.empty_cache()

    raw_im.thumbnail([1536, 1536], Image.LANCZOS)
    if raw_im.mode == "RGBA":
        input_im = np.array(convert_to_rgb(raw_im))
    else:
        input_im = preprocess_image(models, raw_im, preprocess, verbose=verbose)

    if verbose:
        print(x,y,z)

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    input_im = transforms.functional.resize(input_im, [h, w], antialias=True)

    sampler = DDIMSampler(models['turncam'])
    used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
    # used_x = x  # NOTE: Set this way for consistency.
    x_samples_ddim = sample_model(input_im, models['turncam'], sampler, precision, h, w,
                                    ddim_steps, n_samples, scale, ddim_eta, used_x, y, z, verbose=verbose)

    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

    return output_ims

def init(device=None,
        ckpt='./checkpoints',
        device_map=None
        ):
    
    if not device:
        device = f'cuda:{_GPU_INDEX}'
    if os.path.isfile(device_map):
        device_map = omegaconf.OmegaConf.load(device_map)

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print('Instantiating LatentDiffusion...')
    models['turncam'] = load_zero123_ld(
        state_dict_path=ckpt,
        device_map=device_map,
        offload_folder='/tmp'
    )
    models['carvekit'] = create_carvekit_interface()
    return models, device

def generate_view_from_pose(models, device, img, elevation=0, azimuth=0, n_samples=4, ddim_steps=30):
    if type(img) == str:
        img = Image.open(img)

    output_ims = main_run(
        models=models,
        device=device,
        x=elevation, y=azimuth, z=0.0,
        raw_im=img,
        preprocess=True,
        scale=3.0, n_samples=n_samples, ddim_steps=ddim_steps, ddim_eta=1.0,
        precision='fp32', h=256, w=256,
        verbose=False
    )
    return img, output_ims

def generate_nearby_views(model, device, img, spherical_coordinates, ddim_steps=30):
    # Generate nearby views
    print("Generating nearby views...", end="")
    nearby_views = []
    tic = time.time()
    for elevation, azimuth in spherical_coordinates:
        _, output_ims = generate_view_from_pose(model, device, img, elevation, azimuth, n_samples=1, ddim_steps=ddim_steps)
        nearby_views.append(output_ims[0])
    toc = time.time()
    print(f"Done. Time taken: {(toc-tic):.2f}s")
    return nearby_views