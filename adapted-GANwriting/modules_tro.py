"""
modules_tro.py — DohaScript adaptation.

Only change vs. original: write_image() uses PIL/Pillow for Devanagari text
rendering instead of cv2.putText (which cannot render Unicode scripts).
Set DEVANAGARI_FONT to the path of any Devanagari-compatible TTF/OTF font,
e.g. NotoSansDevanagari-Regular.ttf (install via `pip install fonttools` or
download from https://fonts.google.com/noto).
"""

import numpy as np
import os
import torch
from torch import nn

from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
from vgg_tro_channel3_modi import vgg19_bn
from recognizer.models.encoder_vgg import Encoder as rec_encoder
from recognizer.models.decoder import Decoder as rec_decoder
from recognizer.models.seq2seq import Seq2Seq as rec_seq2seq
from recognizer.models.attention import locationAttention as rec_attention
from load_data import (OUTPUT_MAX_LEN, IMG_HEIGHT, IMG_WIDTH,
                       vocab_size, index2letter, num_tokens, tokens)

gpu = torch.device('cuda')

# ── Font for Devanagari text overlay in debug images ─────────────────────────
# Download from: https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari
DEVANAGARI_FONT = 'NotoSansDevanagari-Regular.ttf'
_FONT_SIZE      = 14


def _get_pil_font(size=_FONT_SIZE):
    try:
        from PIL import ImageFont
        return ImageFont.truetype(DEVANAGARI_FONT, size)
    except Exception:
        try:
            from PIL import ImageFont
            return ImageFont.load_default()
        except Exception:
            return None


def _put_text_devanagari(img_array_uint8, text):
    """Render Devanagari `text` onto a uint8 numpy image using PIL."""
    try:
        from PIL import Image, ImageDraw
        pil_img = Image.fromarray(img_array_uint8, mode='L')
        draw    = ImageDraw.Draw(pil_img)
        font    = _get_pil_font()
        if font is not None:
            draw.text((4, 4), text, font=font, fill=255)
        else:
            draw.text((4, 4), text, fill=255)
        return np.array(pil_img)
    except Exception:
        return img_array_uint8  # fallback: return unchanged


# ─────────────────────────────────────────────────────────────────────────────

def normalize(tar):
    mn, mx = tar.min(), tar.max()
    if mx == mn:
        return np.zeros_like(tar, dtype=np.uint8)
    tar = (tar - mn) / (mx - mn) * 255
    return tar.astype(np.uint8)


def _decode_label(label_list):
    """Convert padded integer sequence → Devanagari string."""
    if not isinstance(label_list, list):
        label_list = [label_list]
    # remove special tokens
    filtered = [x for x in label_list if x >= num_tokens]
    return ''.join(index2letter.get(c - num_tokens, '') for c in filtered)


def write_image(xg, pred_label, gt_img, gt_label, tr_imgs,
                xg_swap, pred_label_swap, gt_label_swap,
                title, num_tr=2):
    folder = 'imgs'
    os.makedirs(folder, exist_ok=True)

    batch_size = gt_label.shape[0]
    tr_imgs_np       = tr_imgs.cpu().numpy()
    xg_np            = xg.cpu().numpy()
    xg_swap_np       = xg_swap.cpu().numpy()
    gt_img_np        = gt_img.cpu().numpy()
    gt_label_np      = gt_label.cpu().numpy()
    gt_label_swap_np = gt_label_swap.cpu().numpy()

    pred_label      = torch.topk(pred_label,      1, dim=-1)[1].squeeze(-1).cpu().numpy()
    pred_label_swap = torch.topk(pred_label_swap, 1, dim=-1)[1].squeeze(-1).cpu().numpy()

    tr_imgs_np = tr_imgs_np[:, :num_tr, :, :]
    outs = []

    for i in range(batch_size):
        src      = normalize(tr_imgs_np[i].reshape(num_tr * IMG_HEIGHT, -1))
        gt       = normalize(gt_img_np[i].squeeze())
        tar      = normalize(xg_np[i].squeeze())
        tar_swap = normalize(xg_swap_np[i].squeeze())

        gt_text      = _decode_label(gt_label_np[i].tolist())
        gt_text_swap = _decode_label(gt_label_swap_np[i].tolist())
        pred_text      = _decode_label(pred_label[i].tolist())
        pred_text_swap = _decode_label(pred_label_swap[i].tolist())

        gt_text_img      = _put_text_devanagari(np.zeros_like(tar), gt_text)
        gt_text_img_swap = _put_text_devanagari(np.zeros_like(tar), gt_text_swap)
        pred_text_img      = _put_text_devanagari(np.zeros_like(tar), pred_text)
        pred_text_img_swap = _put_text_devanagari(np.zeros_like(tar), pred_text_swap)

        out = np.vstack([src, gt, gt_text_img, tar, pred_text_img,
                         gt_text_img_swap, tar_swap, pred_text_img_swap])
        outs.append(out)

    import cv2
    cv2.imwrite(os.path.join(folder, title + '.png'), np.hstack(outs))


# ═══════════════════════════════════════════════════════════════════════════════
# AdaIN helpers
# ═══════════════════════════════════════════════════════════════════════════════

def assign_adain_params(adain_params, model):
    for m in model.modules():
        if m.__class__.__name__ == 'AdaptiveInstanceNorm2d':
            mean = adain_params[:, :m.num_features]
            std  = adain_params[:, m.num_features:2*m.num_features]
            m.bias   = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def get_num_adain_params(model):
    n = 0
    for m in model.modules():
        if m.__class__.__name__ == 'AdaptiveInstanceNorm2d':
            n += 2 * m.num_features
    return n


# ═══════════════════════════════════════════════════════════════════════════════
# Model components  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

class DisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers  = 6
        self.final_size = 1024
        nf = 16
        cnn_f = [Conv2dBlock(1, nf, 7, 1, 3,
                             pad_type='reflect', norm='none', activation='none')]
        for _ in range(self.n_layers - 1):
            nf_out = int(np.min([nf * 2, 1024]))
            cnn_f += [ActFirstResBlock(nf, nf,     None, 'lrelu', 'none'),
                      ActFirstResBlock(nf, nf_out,  None, 'lrelu', 'none'),
                      nn.ReflectionPad2d(1),
                      nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = int(np.min([nf * 2, 1024]))
        nf_out = int(np.min([nf * 2, 1024]))
        cnn_f += [ActFirstResBlock(nf, nf,    None, 'lrelu', 'none'),
                  ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
        cnn_c = [Conv2dBlock(nf_out, self.final_size,
                             IMG_HEIGHT // (2 ** (self.n_layers - 1)),
                             IMG_WIDTH  // (2 ** (self.n_layers - 1)) + 1,
                             norm='none', activation='lrelu', activation_first=True)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)
        self.bce   = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.cnn_c(self.cnn_f(x)).squeeze(-1).squeeze(-1)

    def calc_dis_fake_loss(self, x):
        label = torch.zeros(x.shape[0], self.final_size).to(gpu)
        return self.bce(self.forward(x), label)

    def calc_dis_real_loss(self, x):
        label = torch.ones(x.shape[0], self.final_size).to(gpu)
        return self.bce(self.forward(x), label)

    def calc_gen_loss(self, x):
        label = torch.ones(x.shape[0], self.final_size).to(gpu)
        return self.bce(self.forward(x), label)


class WriterClaModel(nn.Module):
    def __init__(self, num_writers):
        super().__init__()
        self.n_layers = 6
        nf = 16
        cnn_f = [Conv2dBlock(1, nf, 7, 1, 3,
                             pad_type='reflect', norm='none', activation='none')]
        for _ in range(self.n_layers - 1):
            nf_out = int(np.min([nf * 2, 1024]))
            cnn_f += [ActFirstResBlock(nf, nf,     None, 'lrelu', 'none'),
                      ActFirstResBlock(nf, nf_out,  None, 'lrelu', 'none'),
                      nn.ReflectionPad2d(1),
                      nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = int(np.min([nf * 2, 1024]))
        nf_out = int(np.min([nf * 2, 1024]))
        cnn_f += [ActFirstResBlock(nf, nf,    None, 'lrelu', 'none'),
                  ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
        cnn_c = [Conv2dBlock(nf_out, num_writers,
                             IMG_HEIGHT // (2 ** (self.n_layers - 1)),
                             IMG_WIDTH  // (2 ** (self.n_layers - 1)) + 1,
                             norm='none', activation='lrelu', activation_first=True)]
        self.cnn_f       = nn.Sequential(*cnn_f)
        self.cnn_c       = nn.Sequential(*cnn_c)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        out = self.cnn_c(self.cnn_f(x)).squeeze(-1).squeeze(-1)
        return self.cross_entropy(out, y)


class GenModel_FC(nn.Module):
    def __init__(self, text_max_len):
        super().__init__()
        self.enc_image  = ImageEncoder().to(gpu)
        self.enc_text   = TextEncoder_FC(text_max_len).to(gpu)
        self.dec        = Decoder().to(gpu)
        self.linear_mix = nn.Linear(1024, 512)

    def decode(self, content, adain_params):
        assign_adain_params(adain_params, self.dec)
        return self.dec(content)

    def mix(self, feat_xs, feat_embed):
        feat_mix = torch.cat([feat_xs, feat_embed], dim=1)
        f  = feat_mix.permute(0, 2, 3, 1)
        ff = self.linear_mix(f)
        return ff.permute(0, 3, 1, 2)


class TextEncoder_FC(nn.Module):
    def __init__(self, text_max_len):
        super().__init__()
        embed_size = 64
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Sequential(
            nn.Linear(text_max_len * embed_size, 1024),
            nn.LayerNorm(1024), nn.ReLU(inplace=False),  # LayerNorm: batch-size agnostic
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048), nn.ReLU(inplace=False),
            nn.Linear(2048, 4096),
        )
        self.linear = nn.Linear(embed_size, 512)

    def forward(self, x, f_xs_shape):
        xx  = self.embed(x)                          # b, t, embed
        b   = xx.shape[0]
        out = self.fc(xx.reshape(b, -1))             # b, 4096

        xx_new    = self.linear(xx)                  # b, t, 512
        ts        = xx_new.shape[1]
        h_reps    = f_xs_shape[-2]
        w_reps    = f_xs_shape[-1] // ts
        pad_reps  = f_xs_shape[-1] % ts

        tensor_list = [torch.cat([xx_new[:, i:i+1]] * w_reps, dim=1)
                       for i in range(ts)]
        if pad_reps:
            pad_tok = torch.full((1, 1), tokens['PAD_TOKEN'],
                                 dtype=torch.long).to(gpu)
            emb_pad = self.linear(self.embed(pad_tok))  # 1,1,512
            padding = emb_pad.repeat(b, pad_reps, 1)
            tensor_list.append(padding)

        res = torch.cat(tensor_list, dim=1)          # b, W, 512
        res = res.permute(0, 2, 1).unsqueeze(2)      # b, 512, 1, W
        final_res = torch.cat([res] * h_reps, dim=2) # b, 512, H, W
        return out, final_res


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vgg19_bn(False)
        self.output_dim = 512

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=512, out_dim=1,
                 res_norm='adain', activ='relu', pad_type='reflect'):
        super().__init__()
        model = [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        for _ in range(ups):
            model += [nn.Upsample(scale_factor=2),
                      Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                  norm='in', activation=activ, pad_type=pad_type)]
            dim //= 2
        model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                              norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class RecModel(nn.Module):
    def __init__(self, pretrain=False):
        super().__init__()
        hidden = 512
        embed  = 60
        self.enc = rec_encoder(hidden, IMG_HEIGHT, IMG_WIDTH, True, None, False).to(gpu)
        self.dec = rec_decoder(hidden, embed, vocab_size, rec_attention, None).to(gpu)
        self.seq2seq = rec_seq2seq(self.enc, self.dec, OUTPUT_MAX_LEN, vocab_size).to(gpu)
        if pretrain:
            model_file = 'recognizer/save_weights/seq2seq_doha.model'
            print('Loading RecModel', model_file)
            self.seq2seq.load_state_dict(torch.load(model_file))

    def forward(self, img, label, img_width):
        # Respect the outer train/eval context instead of hardcoding train mode.
        # During evaluation this ensures BN uses running stats (not batch stats)
        # giving accurate CER; during training it keeps the original behaviour.
        if self.training:
            self.seq2seq.train()
        else:
            self.seq2seq.eval()
        img = torch.cat([img, img, img], dim=1)   # b,1,H,W → b,3,H,W
        output, _ = self.seq2seq(img, label, img_width,
                                 teacher_rate=False, train=False)
        return output.permute(1, 0, 2)            # t,b,V → b,t,V


class MLP(nn.Module):
    def __init__(self, in_dim=64, out_dim=4096, dim=256, n_blk=3,
                 norm='none', activ='relu'):
        super().__init__()
        model = [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for _ in range(n_blk - 2):
            model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        model += [LinearBlock(dim, out_dim, norm='none', activation='none')]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))