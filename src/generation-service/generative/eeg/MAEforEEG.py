#@title MAE for EEG

# utils
import math
import os

import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
import torch.nn.functional as F
from torch import nn

def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_l = np.arange(length, dtype=np.float32)

    grid_l = grid_l.reshape([1, length])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_l)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # cls token
        # height (== width) for the checkpoint position embedding
        orig_size = int(pos_embed_checkpoint.shape[-2] - num_extra_tokens)
        # height (== width) for the new position embedding
        new_size = int(num_patches)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %d to %d" % (orig_size, new_size), flush=True)
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, embedding_size).permute(0, 2, 1)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size))
            pos_tokens = pos_tokens.permute(0, 2, 1)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed



def original_schedule(epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config.warmup_epochs:
        lr = config.lr * epoch / config.warmup_epochs
    else:
        lr = config.min_lr + (config.lr - config.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config.warmup_epochs) / (config.num_epoch - config.warmup_epochs)))
    return lr

def adjust_learning_rate(optimizer, epoch, config):
    # Define new schedule parameters
    cycle_length = 50
    max_lr = config.lr
    min_lr = config.lr * 0.1

    # Blending period
    blending_epochs = 20
    start_blending_epoch = 1000
    end_blending_epoch = start_blending_epoch + blending_epochs

    if epoch < start_blending_epoch:
        lr = original_schedule(epoch, config)
    elif start_blending_epoch <= epoch < end_blending_epoch:
        blend_weight = (epoch - start_blending_epoch) / blending_epochs
        lr_old = original_schedule(epoch, config)
        cycle_progress = (epoch % cycle_length) / cycle_length
        lr_new = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * cycle_progress))
        lr = (1 - blend_weight) * lr_old + blend_weight * lr_new
    else:
        # New schedule
        cycle_progress = (epoch % cycle_length) / cycle_length
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * cycle_progress))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def save_model(config, epoch, model, optimizer, loss_scaler, checkpoint_paths):
    os.makedirs(checkpoint_paths, exist_ok=True)
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'config': config,
    }
    torch.save(to_save, os.path.join(checkpoint_paths, 'checkpoint.pth'))


def load_model(config, model, checkpoint_path ):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print(f'Model loaded with {checkpoint_path}', flush=True)

def patchify(imgs, patch_size):
    """
    imgs: (N, 1, num_voxels)
    x: (N, L, patch_size)
    """
    p = patch_size
    assert imgs.ndim == 3 and imgs.shape[2] % p == 0

    h = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], h, p))
    return x

def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size)
    imgs: (N, 1, num_voxels)
    """
    p = patch_size
    h = x.shape[1]

    imgs = x.reshape(shape=(x.shape[0], 1, h * p))
    return imgs

import sys
#sys.path.append('../dreamdiffusion/code/')
# print(sys.path)
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
import torch.nn.functional as F

# class CustomBlock(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, dropout_rate=0.2, attn_dropout_rate=0.2, norm_layer=nn.LayerNorm):
#         super().__init__()
#         # Multi-head self-attention
#         self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout_rate)

#         # Layer Norm
#         self.norm1 = norm_layer(dim)
#         self.norm2 = norm_layer(dim)

#         # Feed-forward network
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(int(dim * mlp_ratio), dim),
#             nn.Dropout(dropout_rate)
#         )
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         x = x + self.dropout(self.attn(self.norm1(x), x, x)[0])
#         x = x + self.dropout(self.mlp(self.norm2(x)))
#         return x

class PatchEmbed1D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """
    def __init__(self, time_len=224, patch_size=1, in_chans=14, embed_dim=256, dropout_rate=0.2):
        super().__init__()
        num_patches = time_len // patch_size
        self.patch_shape = patch_size
        self.time_len = time_len
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, **kwargs):
        B, C, V = x.shape # batch, channel, voxels
        # assert V == self.num_voxels, \
        #     f"Input fmri length ({V}) doesn't match model ({self.num_voxels})."
        x = self.proj(x).transpose(1, 2).contiguous() # put embed_dim at the last dimension
        x = self.dropout(x)
        return x

class MAEforEEG(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, time_len=1024, patch_size=4, embed_dim=1024, in_chans=14,
                 depth=24, num_heads=16, decoder_embed_dim=512,
                 decoder_depth=4, decoder_num_heads=16,
                 mlp_ratio=2., norm_layer=nn.LayerNorm, focus_range=None, focus_rate=None, img_recon_weight=1.0,
                 use_nature_img_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed1D(time_len, patch_size, in_chans, embed_dim)

        num_patches = int(time_len / patch_size)

        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, proj_drop=0.15, attn_drop=0.115)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, proj_drop=0.15, attn_drop=0.15)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans * patch_size, bias=True) # encoder to decoder
        # --------------------------------------------------------------------------

        # nature image decoder specifics
        if use_nature_img_loss:
            self.nature_img_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.nature_img_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.nature_img_decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

            self.nature_img_decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, proj_drop=0.15, attn_drop=0.15)
                for i in range(2)])

            self.nature_img_decoder_norm = norm_layer(decoder_embed_dim)
            self.nature_img_decoder_pred = nn.Sequential(
                nn.Conv1d(num_patches, 512, kernel_size=1, stride=1, bias=True),
                nn.Linear(decoder_embed_dim, 28*28, bias=True)
            )
            # --------------------------------------------------------------------------

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.focus_range = focus_range
        self.focus_rate = focus_rate
        self.img_recon_weight = img_recon_weight
        self.use_nature_img_loss = use_nature_img_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        if self.use_nature_img_loss:
            nature_img_decoder_pos_embed = get_1d_sincos_pos_embed(self.nature_img_decoder_pos_embed.shape[-1], self.num_patches, cls_token=True)
            self.nature_img_decoder_pos_embed.data.copy_(torch.from_numpy(nature_img_decoder_pos_embed).float().unsqueeze(0))
            torch.nn.init.normal_(self.nature_img_mask_token, std=.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def patchify(self, imgs):
        """
        imgs: (N, 1, num_voxels)
        imgs: [N, chan, T]
        x: (N, L, patch_size)
        x: [N, chan * 4, T/4]
        """
        p = self.patch_embed.patch_size
        assert imgs.ndim == 3 and imgs.shape[1] % p == 0

        # h = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1] // p, -1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_embed.patch_size
        h = x.shape[1]

        imgs = x.reshape(shape=(x.shape[0], -1, x.shape[2] // p))
        return imgs.transpose(1,2)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        if self.focus_range is not None:
            len_mask = L - len_keep
            weights = [1-self.focus_rate] * L
            weights[self.focus_range[0] // self.patch_size : self.focus_range[1] // self.patch_size
                        ] = [self.focus_rate] * (self.focus_range[1] // self.patch_size - self.focus_range[0] // self.patch_size)
            weights = torch.tensor(weights).repeat(N, 1).to(x.device)
            ids_mask = torch.multinomial(weights, len_mask, replacement=False)

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        if self.focus_range is not None:
            for i in range(N):
                noise[i, ids_mask[i,:]] = 1.1  # set mask portion to 1.1

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # print('encoder embed')
        # print(x.shape)
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore = None):
        # embed tokens
        x = self.decoder_embed(x)
        # print('decoder embed')
        # print(x.shape)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # x = x_
        # add pos embed
        x = x + self.decoder_pos_embed
        # x = x + self.decoder_pos_embed[:, 1:, :]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # print(x.shape)
        # predictor projection
        x = self.decoder_pred(x)
        # print(x.shape)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_nature_img_decoder(self, x, ids_restore):
        # embed tokens
        x = self.nature_img_decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.nature_img_mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.nature_img_decoder_pos_embed

        # apply Transformer blocks
        for blk in self.nature_img_decoder_blocks:
            x = blk(x)
        x = self.nature_img_decoder_norm(x)
        # remove cls token
        x = x[:, 1:, :]
        # predictor projection
        # x = x.mean(dim=1, keepdim=True)
        x = self.nature_img_decoder_pred(x)
        x = x.view(x.shape[0], 512, 28, 28)

        return x # n, 512, 28, 28

    def forward_nature_img_loss(self, inputs, reconstructions):
        loss = ((torch.tanh(inputs) - torch.tanh(reconstructions))**2).mean()
        if torch.isnan(reconstructions).sum():
            print('nan in reconstructions', flush=True)
        if torch.isnan(inputs).sum():
            print('nan in inputs', flush=True)

        return loss

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, num_voxels]
        imgs: [N, chan, T]
        pred: [N, L, p]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        imgs = imgs.transpose(1,2)
        target = self.patchify(imgs)
        # target = imgs.transpose(1,2)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # loss = loss.mean()
        loss = (loss * mask).sum() / mask.sum()  if mask.sum() != 0 else (loss * mask).sum() # mean loss on removed patches
        return loss

    def forward(self, imgs, img_features=None, valid_idx=None, mask_ratio=0.75):
        # latent = self.forward_encoder(imgs, mask_ratio)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
            # print(x)
        # print(latent.shape)
        # # print(mask)
        # print(mask.shape)
        # # print(ids_restore)
        # print(ids_restore.shape)

        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p]
        # pred = self.forward_decoder(latent)  # [N, L, p]
        # pred = pred
        # print(pred.shape)
        # mask=None
        loss = self.forward_loss(imgs, pred, mask)
        # print(self.unpatchify(pred.transpose(1,2)).shape)

        if self.use_nature_img_loss and img_features is not None:
            # valid_idx = torch.nonzero(nature_image.sum(dim=(1,2,3)) != 0).squeeze(1)
            if len(valid_idx) != 0:
                nature_image_recon = self.forward_nature_img_decoder(latent[valid_idx], ids_restore[valid_idx])
                loss_nature_image_recon = self.forward_nature_img_loss(img_features, nature_image_recon)
                if torch.isnan(loss_nature_image_recon).sum():
                    print(loss_nature_image_recon, flush=True)
                    print("loss_nature_image_recon is nan", flush=True)

                loss = loss + self.img_recon_weight*loss_nature_image_recon

        return loss, pred, mask

class eeg_encoder(nn.Module):
    def __init__(self, time_len=1024, patch_size=4, embed_dim=2048, in_chans=14,
                 depth=24, num_heads=16, mlp_ratio=1., norm_layer=nn.LayerNorm, global_pool=False):
        super().__init__()
        self.patch_embed = PatchEmbed1D(time_len, patch_size, in_chans, embed_dim)

        num_patches = int(time_len / patch_size)

        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed_dim = embed_dim

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.global_pool = global_pool
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        # print(x.shape)
        # print(self.pos_embed[:, 1:, :].shape)
        x = x + self.pos_embed[:, 1:, :]
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # print(x.shape)
        if self.global_pool:
            x = x.mean(dim=1, keepdim=True)
        # print(x.shape)
        x = self.norm(x)
        # print(x.shape)
        return x

    def forward(self, imgs):
        if imgs.ndim == 2:
            imgs = torch.unsqueeze(imgs, dim=0)  # N, n_seq, embed_dim
        latent = self.forward_encoder(imgs) # N, n_seq, embed_dim
        return latent # N, n_seq, embed_dim

    def load_checkpoint(self, state_dict):
        if self.global_pool:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k and 'norm' not in k)}
        else:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
        interpolate_pos_embed(self, state_dict)

        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u, flush=True)
        print('unexpected keys:', m, flush=True)
        return

class classify_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.Conv1d(14, 1, 1, stride=1)#nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(1024, 40)

    def forward(self, x):
        x = self.maxpool(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x


class mapping(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.Conv1d(256, 1, 1, stride=1)#nn.AdaptiveAvgPool1d((1))
        self.dropout = nn.Dropout(p=0.25)  # Adjust p as needed
        self.fc1 = nn.Linear(1024, 1024)  # Expanding the network
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 768)   # The final output size remains the same

    def forward(self, x):
        x = self.maxpool(x)
        x = x.squeeze(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x
