"""
compare int result and fp result
"""
# torch
import torch.distributed as dist
import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
# torch.cuda.manual_seed(3407)
    
# ldm
import sys
sys.path.append(".")
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

#qalora
from quant_scripts.quant_model import QuantModel_intnlora
from quant_scripts.quant_layer import QuantModule_intnlora, SimpleDequantizer
from quant_scripts.quant_dataset import DiffusionInputDataset

# others
from omegaconf import OmegaConf
import sys
sys.path.append(".")
import argparse
import os
import time
import logging
import numpy as np
from PIL import Image
from einops import rearrange
import random

# global args
n_bits_w = 4
n_bits_a = 4
ddim_steps = 60
ddim_eta = 0.0
scale = 3.0

def seed_all(seed: int) -> None:
    """This function is used to set the random seed for all the packages.
    .. hint::
        To reproduce the results, you need to set the random seed for all the packages. Including
        ``numpy``, ``random``, ``torch``, ``torch.cuda``, ``torch.backends.cudnn``.
    .. warning::
        If you want to use the ``torch.backends.cudnn.benchmark`` or ``torch.backends.cudnn.deterministic``
        and your ``cuda`` version is over 10.2, you need to set the ``CUBLAS_WORKSPACE_CONFIG`` and
        ``PYTHONHASHSEED`` environment variables.
    Args:
        seed (int): The random seed.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
 
    random.seed(seed)
    np.random.seed(seed)
 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

def get_train_samples(train_loader, num_samples):
    image_data, t_data, y_data = [], [], []
    for (image, t, y) in train_loader:
        image_data.append(image)
        t_data.append(t)
        y_data.append(y)
        if len(image_data) >= num_samples:
            break
    return torch.cat(image_data, dim=0)[:num_samples], torch.cat(t_data, dim=0)[:num_samples], torch.cat(y_data, dim=0)[:num_samples]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--out_dir', default='./generated')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    args = parser.parse_args()
    print(args)
    seed_all(3407)
    # init ddp
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    rank = torch.distributed.get_rank()
    ## for debug, not use ddp
    # rank=0
    # local_rank=0
    # Setup PyTorch:
    logging.basicConfig(level=logging.INFO if rank in [-1, 0] else logging.WARN)

    # torch.manual_seed(0 + rank)

    torch.set_grad_enabled(False)

    # Load fp model
    fp_model = get_model()
    fp_model.cuda()
    fp_model.eval()

    # Load qalora model
    qmodel = get_model()
    dmodel = qmodel.model.diffusion_model

    dataset = DiffusionInputDataset('input/DiffusionInput_250steps.pth')
    data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True) ## each sample is (16,4,32,32)

    wq_params = {'n_bits': n_bits_w, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = QuantModel_intnlora(model=dmodel, weight_quant_params=wq_params, act_quant_params=aq_params, num_steps=ddim_steps)
    if rank == 0:
        print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()

    cali_images, cali_t, cali_y = get_train_samples(data_loader, num_samples=1024)
    device = next(qnn.parameters()).device

    # Initialize weight quantization parameters
    qnn.set_quant_state(True, True)

    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule_intnlora) and module.ignore_reconstruction is False:
            module.intn_dequantizer = SimpleDequantizer(uaq=module.weight_quantizer, weight=module.weight)

    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule_intnlora) and module.ignore_reconstruction is False:
            module.weight.data = module.weight.data.byte() ## for running the model

    print('First run to init model...') ## need run to init act quantizer (delta_list)
    with torch.no_grad():
        _ = qnn(cali_images[:4].to(device),cali_t[:4].to(device),cali_y[:4].to(device))

    setattr(qmodel.model, 'diffusion_model', qnn)
    
    ckpt = torch.load(f'quantw{n_bits_w}a{n_bits_a}_{ddim_steps}steps_efficientdm.pth', map_location='cpu')
    qmodel.load_state_dict(ckpt)
    qmodel.cuda()
    qmodel.eval()

    device = torch.device("cuda", local_rank)
    fp_sampler = DDIMSampler(fp_model, ddim_steps)
    qsampler = DDIMSampler(qmodel, ddim_steps)

    # out_path = os.path.join(args.out_dir, f"samples{args.num_samples}steps{ddim_steps}eta{ddim_eta}scale{scale}.npz")
    base_out_dir = args.out_dir
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    output_subdir = os.path.join(base_out_dir, f"run_{timestamp}")
    if rank == 0:
        os.makedirs(output_subdir, exist_ok=True)
        logging.info(f"Output directory: {output_subdir}")

    logging.info("sampling...")
    generated_num = torch.tensor(0, device=device)
    if rank == 0:
        fp_all_images = []
        q_all_images = []
        fp_all_images_list = []
        q_all_images_list = []      
        all_labels = []

        generated_num = torch.tensor(len(fp_all_images) * args.batch_size, device=device)
    dist.barrier()
    dist.broadcast(generated_num, 0)

    while generated_num.item() < args.num_samples:
        t0 = time.time()
        uc = fp_model.get_learned_conditioning(
            {fp_model.cond_stage_key: torch.tensor(args.batch_size*[1000]).to(fp_model.device)}
            )
        xc = torch.randint(0,1000,(args.batch_size,)).to(fp_model.device)
        c = fp_model.get_learned_conditioning({fp_model.cond_stage_key: xc.to(fp_model.device)})
        
        # fp model generate imgs
        fp_samples_ddim, _ = fp_sampler.sample(S=ddim_steps,
                                        conditioning=c,
                                        batch_size=args.batch_size,
                                        shape=[3, 64, 64],
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc, 
                                        eta=ddim_eta)

        fp_x_samples_ddim = fp_model.decode_first_stage(fp_samples_ddim)
        fp_x_samples_ddim = torch.clamp((fp_x_samples_ddim+1.0)/2.0, 
                                    min=0.0, max=1.0)
        fp_samples = fp_x_samples_ddim.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
        fp_samples = fp_samples.permute(0, 2, 3, 1)
        fp_samples = fp_samples.contiguous()
        t1 = time.time()
        print('throughput : {}'.format(fp_x_samples_ddim.shape[0] / (t1 - t0)))
        fp_gathered_samples = [torch.zeros_like(fp_samples) for _ in range(dist.get_world_size())]
        dist.all_gather(fp_gathered_samples, fp_samples)  # gather not supported with NCCL

        # qmodel generate imgs
        q_samples_ddim, _ = qsampler.sample(S=ddim_steps,
                                        conditioning=c,
                                        batch_size=args.batch_size,
                                        shape=[3, 64, 64],
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc, 
                                        eta=ddim_eta)

        q_x_samples_ddim = qmodel.decode_first_stage(q_samples_ddim)
        q_x_samples_ddim = torch.clamp((q_x_samples_ddim+1.0)/2.0, 
                                    min=0.0, max=1.0)
        q_samples_ddim = q_x_samples_ddim.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
        q_samples_ddim = q_samples_ddim.permute(0, 2, 3, 1)
        q_samples_ddim = q_samples_ddim.contiguous()
        q_gathered_samples = [torch.zeros_like(q_samples_ddim) for _ in range(dist.get_world_size())]
        dist.all_gather(q_gathered_samples, q_samples_ddim)  # gather not supported with NCCL
        
        # generate label
        gathered_labels = [
            torch.zeros_like(xc) for _ in range(dist.get_world_size())
        ]      
        dist.all_gather(gathered_labels, xc)

        # store all images one by one to generate npz file
        if rank == 0:
            fp_all_images.extend([sample.cpu().numpy() for sample in fp_gathered_samples])
            q_all_images.extend([qsample.cpu().numpy() for qsample in q_gathered_samples])
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logging.info(f"created {len(fp_all_images) * args.batch_size} samples")
            generated_num = torch.tensor(len(fp_all_images) * args.batch_size, device=device)

        dist.barrier()
        dist.broadcast(generated_num, 0)

        # store all images batch by batch to visualize img
        if rank == 0:
            current_fp_batch_images = []
            current_q_batch_images = []
            for fp_samples_per_rank in fp_gathered_samples:
                current_fp_batch_images.extend(fp_samples_per_rank.cpu().numpy())
            for q_samples_per_rank in q_gathered_samples:
                current_q_batch_images.extend(q_samples_per_rank.cpu().numpy())

            fp_all_images_list.extend(current_fp_batch_images)
            q_all_images_list.extend(current_q_batch_images)
            for idx_in_batch in range(len(current_fp_batch_images)):
                fp_img = current_fp_batch_images[idx_in_batch]
                q_img = current_q_batch_images[idx_in_batch]

                # left: fp_img, right: q_img
                combined_img = np.concatenate([fp_img, q_img], axis=1)
                global_idx = len(fp_all_images_list) - len(current_fp_batch_images) + idx_in_batch
                filename = os.path.join(
                    output_subdir, f"sample_{global_idx:06d}.png"
                )
                Image.fromarray(combined_img).save(filename)

                if idx_in_batch < 5:
                    logging.info(f"Saved {filename}")
    if rank == 0:
        # fp imgs
        fp_arr = np.concatenate(fp_all_images, axis=0)
        fp_arr = fp_arr[: args.num_samples]

        # q imgs
        q_arr = np.concatenate(q_all_images, axis=0)
        q_arr = q_arr[: args.num_samples]

        # labels
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

        fp_out_path = os.path.join(
            output_subdir,
            f"fp_samples{args.num_samples}_steps{ddim_steps}_eta{ddim_eta}_scale{scale}.npz"
        )

        q_out_path = os.path.join(
            output_subdir,
            f"q_samples{args.num_samples}_steps{ddim_steps}_eta{ddim_eta}_scale{scale}.npz"
        )
        np.savez(fp_out_path, fp_arr, label_arr)
        np.savez(q_out_path, q_arr, label_arr)
        logging.info(f"fp npz file is saving to {fp_out_path}")
        logging.info(f"quant npz file is saving to {q_out_path}")

    dist.barrier()
    logging.info("sampling complete")


if __name__ == "__main__":
    main()