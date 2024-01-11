#@title ********  ELDM  ********
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from einops import rearrange, repeat
from omegaconf import OmegaConf

import gc

import torch.nn as nn

from generative.ldm.LdmUtil import instantiate_from_config
from generative.diffusion.PlmSampler import PLMSSampler
from generative.eeg.MAEforEEG import eeg_encoder, mapping

def create_model_from_config(config, num_voxels, global_pool):
    model = eeg_encoder(time_len=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool)
    return model

class cond_stage_model(nn.Module):
    def __init__(self, metafile, num_voxels=440, cond_dim=1280, global_pool=True, clip_tune = True, cls_tune = False):
        super().__init__()
        # prepare pretrained fmri mae
        if metafile is not None:
            model = create_model_from_config(metafile['config'], num_voxels, global_pool)

            model.load_checkpoint(metafile['model'])
        else:
            model = eeg_encoder(time_len=num_voxels, global_pool=global_pool)
        self.mae = model
        if clip_tune:
            self.mapping = mapping()
        if cls_tune:
            self.cls_net = classify_network()

        self.fmri_seq_len = model.num_patches
        self.fmri_latent_dim = model.embed_dim
        if global_pool == False:
            # dropout
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
                nn.ReLU(),
                nn.Dropout(p=0.25),  # Include dropout after activation
                nn.Conv1d(self.fmri_seq_len // 2, self.fmri_seq_len // 4, 1, bias=True),  # Additional layer
                nn.ReLU(),
                nn.Dropout(p=0.25),  # Include dropout after activation
                nn.Conv1d(self.fmri_seq_len // 4, 77, 1, bias=True)  # Adjusted for additional layer
            )
            # without dropout
            # self.channel_mapper = nn.Sequential(
            #     nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
            #     nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
            # )
            self.channel_batch_norm = nn.BatchNorm1d(num_features=77)  # Adjusted for the final size

        self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.dropout = nn.Dropout(p=0.25)
        self.global_pool = global_pool

        # self.image_embedder = FrozenImageEmbedder()

    # def forward(self, x):
    #     # n, c, w = x.shape
    #     latent_crossattn = self.mae(x)
    #     if self.global_pool == False:
    #         latent_crossattn = self.channel_mapper(latent_crossattn)
    #     latent_crossattn = self.dim_mapper(latent_crossattn)
    #     out = latent_crossattn
    #     return out

    def forward(self, x):
        # n, c, w = x.shape
        latent_crossattn = self.mae(x)
        latent_return = latent_crossattn
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn)
            latent_crossattn = self.channel_batch_norm(latent_crossattn)
        latent_crossattn = self.dropout(latent_crossattn)
        latent_crossattn = self.dim_mapper(latent_crossattn)
        out = latent_crossattn
        return out, latent_return

    # def recon(self, x):
    #     recon = self.decoder(x)
    #     return recon

    def get_cls(self, x):
        return self.cls_net(x)

    def get_clip_loss(self, x, image_embeds, weight_decay=5e-3):
        target_emb = self.mapping(x)
        loss = 1 - torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()
        
        return loss

#         # L2 Regularization (squared L2 norm)
#         l2_reg = sum(torch.sum(param ** 2) for param in self.mapping.parameters())

#         # No need to take the square root for L2 regularization
#         # Apply weight decay to the regularization term
#         loss += weight_decay * l2_reg

        # return loss



class eLDM:

    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/',
                 logger=None, ddim_steps=125, global_pool=True, use_time_cond=False, clip_tune = True, cls_tune = False, temperature=0.1, dataloader=None):
        # self.ckp_path = os.path.join(pretrain_root, 'model.ckpt')
        #self.ckp_path = 'mj/mdjrny-v4.ckpt'
        self.config_path = os.path.join('config15.yaml')
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim

        print(config.model.target)
        model = instantiate_from_config(config.model)
        #pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']

        #m, u = model.load_state_dict(pl_sd, strict=False)
        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(metafile, num_voxels, self.cond_dim, global_pool=global_pool, clip_tune = clip_tune,cls_tune = cls_tune)

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult
        model.dataloader = dataloader


        self.device = device
        self.model = model

        self.model.clip_tune = clip_tune
        self.model.cls_tune = cls_tune

        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim
        self.metafile = metafile
        self.temperature=temperature

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one

        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####', flush=True)
        print(f'batch_size is: {bs1}', flush=True)
        dataloader = DataLoader(dataset, batch_size=bs1, num_workers=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bs1, num_workers=8, shuffle=True)
        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()
        # self.model.freeze_whole_model()
        # self.model.unfreeze_cond_stage()

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        self.model.unfreeze_whole_model()

#         torch.save(
#             {
#                 'model_state_dict': self.model.state_dict(),
#                 'config': config,
#                 'state': torch.random.get_rng_state()

#             },
#             os.path.join(output_path, 'checkpoint.pth')
#         )


    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path = None, shouldSave = True):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels,
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model, temperature=self.temperature)
        # sampler = DDIMSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)

        with model.ema_scope():
            model.eval()
            print(f'generating for: {fmri_embedding}', flush=True)
            latent = fmri_embedding
            # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'

            c, re_latent = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
            # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
            samples_ddim = sampler.sample(S=ddim_steps,
                                            conditioning=c,
                                            batch_size=num_samples,
                                            shape=shape,
                                            verbose=False,
                                            temperature=self.temperature)
            
            del re_latent
            del sampler
            del model.cond_stage_model
            gc.collect()
            print('cleared up memory', flush=True)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            del samples_ddim
            print('decoded', flush=True)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
            print('clamped', flush=True)
            # gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
            
            samples_t = (255. * torch.cat([x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
            for copy_idx, img_t in enumerate(samples_t):
                img_t = rearrange(img_t, 'c h w -> h w c')
                all_samples.append(Image.fromarray(img_t))

            # all_samples.append(torch.cat([x_samples_ddim.detach().cpu()], dim=0))
            # if output_path is not None and shouldSave == True:
            #     samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
            #     for copy_idx, img_t in enumerate(samples_t):
            #         img_t = rearrange(img_t, 'c h w -> h w c')
            #         Image.fromarray(img_t).save(os.path.join(output_path,
            #             f'./test{count}-{copy_idx}.png'))

        # display as grid
        # print('stack display as grid', flush=True)
        # grid = torch.stack(all_samples, 0)
        # print('rearrange display as grid', flush=True)
        # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        # print('make display as grid', flush=True)
        # grid = make_grid(grid, nrow=num_samples+1)

        # to image
        # print('cpu1', flush=True)
        # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        # print('cpu2', flush=True)
        model = model.to('cpu')

        print('finished processing images', flush=True)
        return all_samples
