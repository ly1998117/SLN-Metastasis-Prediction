import torch
from tqdm import tqdm
import sys
from .breastclip.model.modules import load_image_encoder


def get_encoder(clip_chk_pt_path='/data_smr/liuy/Project/BreastCancer/mammoclip/checkpoints/b5-model-best-epoch-7.tar'):
    ckpt = torch.load(clip_chk_pt_path, map_location="cpu")
    print(ckpt["config"]["model"]["image_encoder"])
    image_encoder = load_image_encoder(ckpt["config"]["model"]["image_encoder"])
    image_encoder_weights = {}
    for k in ckpt["model"].keys():
        if k.startswith("image_encoder."):
            image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
    image_encoder.load_state_dict(image_encoder_weights, strict=True)
    return image_encoder


class MammoCLIP:
    def __init__(self,
                 clip_chk_pt_path='/data_smr/liuy/Project/BreastCancer/mammoclip/checkpoints/b5-model-best-epoch-7.tar',
                 device='cpu', mean=0.3089279, std=0.25053555408335154):
        self.clip_chk_pt_path = clip_chk_pt_path
        self.device = device
        self.mean = mean
        self.std = std
        self.encoder = get_encoder()

    def get_embedding(self, image):
        image = image - image.min()
        image = image / image.max()
        image = (image - self.mean) / self.std
        image_features = self.encoder(image)
        return image_features


if __name__ == '__main__':
    clip = MammoCLIP(
        clip_chk_pt_path='/data_smr/liuy/Project/BreastCancer/mammoclip/checkpoints/b5-model-best-epoch-7.tar',
        device='cpu')
    x = torch.randn(1, 1, 128, 128)
    clip.get_embedding(x)
