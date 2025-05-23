"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
import argparse
import os
import time
import logging

import numpy as np
import torch.distributed as dist

import torch
torch.set_grad_enabled(False)
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image  
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--out_dir', default='./generated')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--resume", action='store_true')
    args = parser.parse_args()
    print(args)
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
    if args.resume:
        seed = int(time.time())
        torch.manual_seed(seed + rank)
    else:
        torch.manual_seed(0 + rank)

    torch.set_grad_enabled(False)
    device = torch.device("cuda", local_rank)

    ddim_steps = 20
    ddim_eta = 0.0
    scale = 3.0

    # Load model:
    model = get_model()

    # sampler = DDIMSampler(model)
    sampler = DDIMSampler(model, ddim_steps)

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
        all_images = []
        all_labels = []
        all_images_list = []
        if args.resume:
            if os.path.exists(output_subdir):
                ckpt = np.load(output_subdir)
                all_images = ckpt['arr_0']
                all_labels = ckpt['arr_1']
                assert all_images.shape[0] % args.batch_size == 0, f'Wrong resume checkpoint shape {all_images.shape}'
                all_images = np.split(all_images,
                                      all_images.shape[0] // args.batch_size,
                                      0)
                all_labels = np.split(all_labels,
                                      all_labels.shape[0] // args.batch_size,
                                      0)

                logging.info('successfully resume from the ckpt')
                logging.info(f'Current number of created samples: {len(all_images) * args.batch_size}')
        generated_num = torch.tensor(len(all_images) * args.batch_size, device=device)
    dist.barrier()
    dist.broadcast(generated_num, 0)

    while generated_num.item() < args.num_samples:
        t0 = time.time()
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(args.batch_size*[1000]).to(model.device)}
            )
        
        xc = torch.randint(0,1000,(args.batch_size,)).to(model.device)

        c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
        
        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                        conditioning=c,
                                        batch_size=args.batch_size,
                                        shape=[3, 64, 64],
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc, 
                                        eta=ddim_eta)

        x_samples_ddim = model.decode_first_stage(samples_ddim)


        # x_samples_ddim = ((x_samples_ddim + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        # x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1)
        # samples = x_samples_ddim.contiguous()

        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                    min=0.0, max=1.0)
        
        samples = x_samples_ddim.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
        samples = samples.permute(0, 2, 3, 1)
        samples = samples.contiguous()
        t1 = time.time()
        print('throughput : {}'.format(x_samples_ddim.shape[0] / (t1 - t0)))
        gathered_samples = [torch.zeros_like(samples) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, samples)  # gather not supported with NCCL

        gathered_labels = [
            torch.zeros_like(xc) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_labels, xc)

        if rank == 0:
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logging.info(f"created {len(all_images) * args.batch_size} samples")
            generated_num = torch.tensor(len(all_images) * args.batch_size, device=device)
            if args.resume:
                if generated_num % 1024 == 0:
                    arr = np.concatenate(all_images, axis=0)
                    arr = arr[: args.num_samples]

                    label_arr = np.concatenate(all_labels, axis=0)
                    label_arr = label_arr[: args.num_samples]
                    logging.info(f"intermediate results saved to {output_subdir}")
                    np.savez(output_subdir, arr, label_arr)
                    del arr
                    del label_arr
        torch.distributed.barrier()
        dist.broadcast(generated_num, 0)

        if rank == 0:
            current_batch_images = []
            for samples_per_rank in gathered_samples:
                current_batch_images.extend(samples_per_rank.cpu().numpy())
            all_images_list.extend(current_batch_images)
            for idx_in_batch, img_array in enumerate(current_batch_images):
                global_idx = len(all_images_list) - len(current_batch_images) + idx_in_batch
                filename = os.path.join(
                    output_subdir, f"sample_{global_idx:06d}.png"
                )
                Image.fromarray(img_array).save(filename)
                if idx_in_batch < 5:
                    logging.info(f"Saved {filename}")

    if rank == 0:
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]

        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
        out_path = os.path.join(
            output_subdir,
            f"samples{args.num_samples}_steps{ddim_steps}_eta{ddim_eta}_scale{scale}.npz"
        )
        logging.info(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logging.info("sampling complete")


if __name__ == "__main__":
    main()