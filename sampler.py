import os, sys, math, random
from PIL import Image 
from torchvision.utils import save_image
import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf

from utils import util_net
from utils import util_image
from utils import util_common

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from datapipe.datasets import create_dataset
from utils.util_image import ImageSpliterTh

from pathlib import Path
from tqdm import tqdm
import torch

class BaseSampler:
    def __init__(
            self,
            configs,
            sf=None,
            use_fp16=False,
            chop_size=128,
            chop_stride=128,
            chop_bs=1,
            desired_min_size=64,
            seed=10000,
            ddim=False
            ):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        '''
        self.configs = configs
        self.chop_size = chop_size
        self.chop_stride = chop_stride
        self.chop_bs = chop_bs
        self.seed = seed
        self.use_fp16 = use_fp16
        self.desired_min_size = desired_min_size
        self.ddim=ddim
        if sf is None:
            sf = configs.diffusion.params.sf
        self.sf = sf

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = torch.cuda.device_count()
        assert num_gpus == 1, 'Please assign one available GPU using CUDA_VISIBLE_DEVICES!'

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str)

    def build_model(self):
        # self.configs.diffusion.params.steps = 1000
        # diffusion model
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path =self.configs.model.ckpt_path
        assert ckpt_path is not None
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        self.load_model(model, ckpt_path)
        if self.use_fp16:
            model.dtype = torch.float16
            model.convert_to_fp16()
        self.model = model.eval()

        # autoencoder model
        if self.configs.autoencoder is not None:
            ckpt_path = self.configs.autoencoder.ckpt_path
            assert ckpt_path is not None
            self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
            autoencoder = util_common.instantiate_from_config(self.configs.autoencoder).cuda()
            self.load_model(autoencoder, ckpt_path)
            autoencoder.eval()
            if self.configs.autoencoder.use_fp16:
                self.autoencoder = autoencoder.half()
            else:
                self.autoencoder = autoencoder
        else:
            self.autoencoder = None

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)

class Sampler(BaseSampler):    
    def sample_func(self, y0, noise_repeat=False, one_step=False, apply_decoder=True):
        '''
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        '''
        if noise_repeat:
            self.setup_seed()

        desired_min_size = self.desired_min_size
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % desired_min_size == 0 and ori_w % desired_min_size == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / desired_min_size)) * desired_min_size - ori_h
            pad_w = (math.ceil(ori_w / desired_min_size)) * desired_min_size - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')
        else:
            flag_pad = False

        # output_folder="testdata/sample_func_y0"
        # os.makedirs(output_folder, exist_ok=True)
        # save_image(y0.squeeze(0), os.path.join(output_folder, "y0.png"))
        

        model_kwargs={'lq':y0,} if self.configs.model.params.cond_lq else None
        
        if not self.ddim:        
            results = self.base_diffusion.p_sample_loop(
                    y=y0,
                    model=self.model,
                    first_stage_model=self.autoencoder,
                    noise=None,
                    noise_repeat=noise_repeat,
                    clip_denoised=(self.autoencoder is None),
                    denoised_fn=None,
                    model_kwargs=model_kwargs,
                    progress=False,
                    one_step=one_step,
                    apply_decoder=apply_decoder
                    )    # This has included the decoding for latent space
        else:
            results = self.base_diffusion.ddim_sample_loop(
                    y=y0,
                    model=self.model,
                    first_stage_model=self.autoencoder,
                    noise=None,
                    clip_denoised=(self.autoencoder is None),
                    denoised_fn=None,
                    model_kwargs=model_kwargs,
                    progress=True,
                    one_step=one_step,
                    apply_decoder=apply_decoder
                    )    # This has included the decoding for latent space
        if flag_pad and apply_decoder:
            results = results[:, :, :ori_h*self.sf, :ori_w*self.sf]
            
        if not apply_decoder:
            return results["pred_xstart"]
        return results.clamp_(-1.0, 1.0)

    
    def inference(self, in_path, out_path, bs=1, noise_repeat=False, one_step=False, return_tensor=False, apply_decoder=True):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
        '''
        def _process_per_image(im_lq_tensor):
            '''
            Input:
                im_lq_tensor: b x c x h x w, torch tensor, [0,1], RGB
            Output:
                im_sr_tensor: torch tensor, [0,1], RGB
            '''
            if im_lq_tensor.shape[2] > self.chop_size or im_lq_tensor.shape[3] > self.chop_size:
                # 分块处理逻辑不变
                im_spliter = ImageSpliterTh(
                    im_lq_tensor,
                    self.chop_size,
                    stride=self.chop_stride,
                    sf=self.sf,
                    extra_bs=self.chop_bs,
                )
                for im_lq_pch, index_infos in im_spliter:
                    im_sr_pch = self.sample_func(
                        (im_lq_pch - 0.5) / 0.5,
                        noise_repeat=noise_repeat,
                        one_step=one_step,
                        apply_decoder=apply_decoder
                    )
                    im_spliter.update(im_sr_pch.detach(), index_infos)
                im_sr_tensor = im_spliter.gather()
            else:
                im_sr_tensor = self.sample_func(
                    (im_lq_tensor - 0.5) / 0.5,
                    noise_repeat=noise_repeat,
                    one_step=one_step,
                    apply_decoder=apply_decoder
                )

            if apply_decoder:
                im_sr_tensor = im_sr_tensor * 0.5 + 0.5
            return im_sr_tensor

        in_path = Path(in_path)
        out_path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
        return_res = {}

        if bs > 1:
            # 批量模式：目录输入，先列出所有文件并过滤已存在的
            assert in_path.is_dir(), "Input path must be folder when batch size is larger than 1."
            all_imgs = list(in_path.glob("*.[jpJP][pnPN]*[gG]"))
            # 只留下那些输出目录中还不存在的
            to_process = []
            for p in all_imgs:
                out_file = out_path / f"{p.stem}.png"
                if out_file.exists():
                    self.write_log(f"{p.stem}.png already exists, skipping.")
                else:
                    to_process.append(p)
            self.write_log(f'Total {len(all_imgs)} images found, {len(to_process)} to process.')

            if not to_process:
                self.write_log("No new images to process. Exiting.")
                return return_res

            # 重建一个只包含待处理路径的 dataset
            data_config = {
                'type': 'folder',
                'params': {
                    'dir_path': str(in_path),
                    'transform_type': 'default',
                    'transform_kwargs': {'mean': 0.0, 'std': 1.0},
                    'need_path': True,
                    'recursive': True,
                    'length': None,
                }
            }
            full_dataset = create_dataset(data_config)
            # 根据 path 筛选 dataset 中的样本索引
            path_set = {str(p.resolve()) for p in to_process}
            indices = [
                idx for idx, item in enumerate(full_dataset)
                if Path(item['path']).resolve().as_posix() in path_set
            ]
            # SubsetDataset: 假设你有从 indices 构造子集的工具
            dataset = torch.utils.data.Subset(full_dataset, indices)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=bs, shuffle=False, drop_last=False
            )

            for micro_data in tqdm(dataloader):
                paths = micro_data['path']
                lq = micro_data['lq'].cuda()
                results = _process_per_image(lq)

                for jj, p in enumerate(paths):
                    im_name = Path(p).stem
                    out_file = out_path / f"{im_name}.png"
                    im_sr = util_image.tensor2img(
                        results[jj], rgb2bgr=True, min_max=(0.0, 1.0)
                    )
                    util_image.imwrite(im_sr, out_file, chn='bgr', dtype_in='uint8')
                    if return_tensor:
                        return_res[out_file.stem] = results[jj]

        else:
            # 单张或遍历目录模式（bs==1），与之前逻辑一致
            if not in_path.is_dir():
                paths = [in_path]
            else:
                paths = list(in_path.glob("*.[jpJP][pnPN]*[gG]"))
                self.write_log(f'Find {len(paths)} images in {in_path}')

            for p in tqdm(paths):
                out_file = out_path / f"{p.stem}.png"
                if out_file.exists():
                    self.write_log(f"{p.stem}.png already exists, skipping.")
                    continue

                im_lq = util_image.imread(p, chn='rgb', dtype='float32')
                im_lq_tensor = util_image.img2tensor(im_lq).cuda()
                im_sr_tensor = _process_per_image(im_lq_tensor)
                im_sr = util_image.tensor2img(im_sr_tensor, rgb2bgr=True, min_max=(0.0, 1.0))
                util_image.imwrite(im_sr, out_file, chn='bgr', dtype_in='uint8')
                if return_tensor:
                    return_res[out_file.stem] = im_sr_tensor

        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")
        return return_res
    
if __name__ == '__main__':
    pass

