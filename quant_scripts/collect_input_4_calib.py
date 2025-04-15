'''
To collect input data, remember to uncomment line 987-988 in ldm/models/diffusion/ddpm.py and comment them after finish collecting.
'''
import sys
sys.path.append(".")
# sys.path.append('./taming-transformers')
import os

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

if __name__ == '__main__':
    model = get_model()
    sampler = DDIMSampler(model)
    total_img = 2000
    batch_size = 8

    ddim_steps = 100
    ddim_eta = 1.0
    scale = 1.5

    all_samples = list()

    with torch.no_grad():
        with model.ema_scope():
            for iter in range(total_img // (batch_size * ddim_steps)):
                uc = model.get_learned_conditioning(
                    {model.cond_stage_key: torch.tensor(batch_size*[1000]).to(model.device)}
                    )
                xc = torch.randint(0,1000,(batch_size,)).to(model.device)
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=batch_size,
                                                shape=[3, 64, 64],
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, 
                                                eta=ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)

    ## save diffusion input data
    import ldm.globalvar as globalvar   
    input_list = globalvar.getInputList()
    torch.save(input_list, f'input/DiffusionInput_{total_img}total_{ddim_steps}steps.pth')
    sys.exit(0)
