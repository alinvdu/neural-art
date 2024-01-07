#@title Config
import numpy as np
import os

class Config_MAE_fMRI: # back compatibility
    pass

class Config_MBM_EEG(Config_MAE_fMRI):
    # configs for fmri_pretrain.py
    def __init__(self):
    # --------------------------------------------
    # MAE for fMRI
        # Training Parameters
        self.lr = 2.5e-4
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 500
        self.warmup_epochs = 40
        self.batch_size = 100
        self.clip_grad = 0.8

        # Model Parameters
        self.mask_ratio = 0.1
        self.patch_size = 4 #  1
        self.embed_dim = 1024 #256 # has to be a multiple of num_heads
        self.decoder_embed_dim = 512 #128
        self.depth = 24
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0

        # Project setting
        self.root_path = '/dreamdiffusion/'
        self.output_path = '/dreamdiffusion/exps/'
        self.seed = 2022
        self.roi = 'VC'
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = True
        self.accum_iter = 1

        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0

class Config_Generative_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = 'dreamdiffusion/'
        self.output_path = 'dreamdiffusion/output/'

        #self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        #self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_single.pth')
        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')
        self.roi = 'VC'
        self.patch_size = 4 # 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 1.0

        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains')

        self.dataset = 'EEG'
        self.pretrain_mbm_path = None

        self.img_size = 512

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 2
        self.lr = 1.25e-5
        self.num_epoch = 25

        self.precision = 8
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.global_pool = False
        self.use_time_cond = True
        self.clip_tune = True #False
        self.cls_tune = False
        self.subject = 4
        self.eval_avg = True

        # diffusion sampling parameters
        self.num_samples = 1
        self.ddim_steps = 100
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None
        self.temperature = 0.1
