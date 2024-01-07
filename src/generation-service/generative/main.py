# Load the Model some EEG data from the eeg14/eegData_npy and predict some images
from torch.nn import Identity
import random
from io import BytesIO
import base64
from PIL import Image
import numpy as np

#@title Utils for the main part
from einops import rearrange
import torch
import torchvision.transforms as transforms
import os
import pytorch_lightning as pl

import json

from generative.eLDM.eLDM import eLDM

def create_trainer(num_epoch, precision=32, accumulate_grad_batches=2,logger=None,check_val_every_n_epoch=10):
    acc = 'gpu' if torch.cuda.is_available() else 'cpu'
    return pl.Trainer(accelerator=acc, max_epochs=num_epoch, logger=logger,
            precision=precision, accumulate_grad_batches=accumulate_grad_batches,
            enable_checkpointing=False, enable_model_summary=False, gradient_clip_val=0.5,
            check_val_every_n_epoch=check_val_every_n_epoch)

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')

def get_eval_metric(samples, avg=True):
    metric_list = ['mse', 'pcc', 'ssim', 'psm']
    res_list = []

    gt_images = [img[0] for img in samples]
    gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
    samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))
    # No class metric for now
    # res_part = []
    # for s in samples_to_run:
    #     pred_images = [img[s] for img in samples]
    #     pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
    #     res = get_similarity_metric(pred_images, gt_images, 'class', None,
    #                     n_way=50, num_trials=50, top_k=1, device='cuda')
    #     res_part.append(np.mean(res))
    # res_list.append(np.mean(res_part))
    # res_list.append(np.max(res_part))
    # metric_list.append('top-1-class')
    # metric_list.append('top-1-class (max)')
    return res_list, metric_list

def generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config):
    grid = generative_model.generate(eeg_latents_dataset_train, config.num_samples,
                config.ddim_steps, config.HW, 3) # generate 3
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path, 'samples_train.png'))
    # wandb.log({'summary/samples_train': wandb.Image(grid_imgs)})

    grid, samples = generative_model.generate(eeg_latents_dataset_test, config.num_samples,
                config.ddim_steps, config.HW, 3)
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path,f'./samples_test.png'))
    for sp_idx, imgs in enumerate(samples):
        for copy_idx, img in enumerate(imgs[1:]):
            img = rearrange(img, 'c h w -> h w c')
            Image.fromarray(img).save(os.path.join(config.output_path,
                            f'./test{sp_idx}-{copy_idx}.png'))

    # wandb.log({f'summary/samples_test': wandb.Image(grid_imgs)})

    metric, metric_list = get_eval_metric(samples, avg=config.eval_avg)
    metric_dict = {f'summary/pair-wise_{k}':v for k, v in zip(metric_list[:-2], metric[:-2])}
    metric_dict[f'summary/{metric_list[-2]}'] = metric[-2]
    metric_dict[f'summary/{metric_list[-1]}'] = metric[-1]
    # wandb.log(metric_dict)


def transform_normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

def transform_channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')

crop_pix = int(0.2*512)
img_transform_train = transforms.Compose([
    transform_normalize,

    transforms.Resize((512, 512)),
    random_crop(512-crop_pix, p=0.5),

    transforms.Resize((512, 512)),
    transform_channel_last
])

img_transform_test = transforms.Compose([
    transform_normalize,

    transforms.Resize((512, 512)),
    transform_channel_last
])

modelCheckpoint = 'generative/checkpoints/checkpoint.pth'
eegModelPath = 'generative/checkpoints/checkpoint-eeg.pth'
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
if torch.backends.mps.is_available():
    print('metal backend available, switching...', flush=True)
    device = torch.device('mps')

print(device, flush=True)

class Imaginative:
    def generate_images_T(self, generative_model, eegData, clientId):
        all_samples = generative_model.generate(eegData, 2, 
                    250, None, 1, shouldSave = False)
        # # Convert each image in the grid to a base64 string
        images = []
        try:
            # Assuming 'grid' is a numpy array representing the entire grid image
            print('Processing images', flush=True)
            
            # save images into an array as base64
            for img in all_samples:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

            # # Save the image to a buffer
            # buffered = BytesIO()
            # grid_img.save(buffered, format="PNG")

            # # Encode the image to base64
            # img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            print(f'Error occurred: {e}', flush=True)

        # Prepare JSON object with base64 encoded images
        json_data = json.dumps({"images": images, "client": "generation-service", "clientId": clientId})
        return json_data

    def initELDMModel(self):
        model_meta = torch.load(modelCheckpoint, map_location='cpu')
        pretrain_mbm_metafile = torch.load(eegModelPath, map_location='cpu')
        self.generative_model = eLDM(pretrain_mbm_metafile, num_voxels=1024,
                        device=device, pretrain_root='', logger=None,
                        ddim_steps=100, global_pool=False, use_time_cond=True, clip_tune = True, cls_tune = False, temperature=0.15)
        self.generative_model.model.load_state_dict(model_meta['model_state_dict'])
        self.generative_model.model.to(device)
        print('model loaded')

    def imagine(self, eegData, clientId):
        if (not self.generative_model):
            print('Model not instantiated, please instantiate the eLDM model', flush=True)
        else:
            return self.generate_images_T(self.generative_model, eegData, clientId)
