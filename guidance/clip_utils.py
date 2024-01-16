import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
import torchvision.transforms.functional as TF

import clip


class CLIP(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()

        self.device = device

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=self.device, jit=False)

        # image augmentation
        self.aug = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # self.gaussian_blur = T.GaussianBlur(15, sigma=(0.1, 10))

    def get_text_embeds(self, prompt, negative_prompt, **kwargs):
        # NOTE: negative_prompt is ignored for CLIP.

        text = clip.tokenize(prompt).to(self.device)
        text_z = self.clip_model.encode_text(text)
        text_z = text_z / text_z.norm(dim=-1, keepdim=True)

        return text_z

    def get_image_embeds(self, pred_rgb):
        pred_rgb = self.aug(pred_rgb)

        image_z = self.clip_model.encode_image(pred_rgb)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True)

        return image_z

    def train_step(self, text_z, pred_rgb, **kwargs):
        pred_rgb = self.aug(pred_rgb)

        image_z = self.clip_model.encode_image(pred_rgb)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True)  # normalize features

        loss = - (image_z * text_z).sum(-1).mean()

        return loss


if __name__ == '__main__':
    import os
    import numpy as np
    import torch
    from skimage.io import imread, imsave
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='others/data/compare/spiderman')
    parser.add_argument('--data_path', type=str, default='others/output/compare/spiderman')
    parser.add_argument('--long_image', action='store_true')
    args = parser.parse_args()

    model = CLIP("cuda")
    image_size = 256

    input_images = []
    for img_path in os.listdir(args.input_path):
        img = imread(os.path.join(args.input_path, img_path)) / 255
        if img.shape[-1] == 4:
            mask = img[..., 3:]
            img = img[..., :3] * mask + (1 - mask)
        img = torch.tensor(img[None, ..., :3], dtype=torch.float)
        input_images.append(img)

    gen_images = []
    for img_path in os.listdir(args.data_path):
        if '.png' in img_path:
            img = imread(os.path.join(args.data_path, img_path)) / 255
            if args.long_image:
                for index in range(0, 16):
                    rgb = np.copy(img[:, index * image_size:(index + 1) * image_size, :])
                    rgb = torch.tensor(rgb[None, ..., :3], dtype=torch.float)
                    gen_images.append(rgb)
            else:
                rgb = np.copy(img)
                rgb = torch.tensor(rgb[None, ..., :3], dtype=torch.float)
                rgb = F.interpolate(
                    rgb.permute(0, 3, 1, 2), (256, 256), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
                gen_images.append(rgb)

    input_images = torch.cat(input_images, dim=0).permute(0, 3, 1, 2).cuda()
    gen_images = torch.cat(gen_images, dim=0).permute(0, 3, 1, 2).cuda()

    input_z = model.get_image_embeds(input_images)
    gen_z = model.get_image_embeds(gen_images)
    similarity = input_z[:, None, ...] * gen_z[None, :, ...]
    print(similarity.sum(-1).mean())
