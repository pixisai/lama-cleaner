import os

import cv2
import numpy as np
import torch
from loguru import logger

from lama_cleaner.helper import  download_model, norm_img, get_cache_path_by_url
from lama_cleaner.model.base import InpaintModel
from lama_cleaner.schema import Config

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
)


class LaMa(InpaintModel):
    pad_mod = 8

    def init_model(self, device, **kwargs):
        if os.environ.get("LAMA_MODEL"):
            model_path = os.environ.get("LAMA_MODEL")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"lama torchscript model not found: {model_path}"
                )
        else:
            model_path = download_model(LAMA_MODEL_URL)
        logger.info(f"Load LaMa model from: {model_path}")
        model = torch.jit.load(model_path, map_location="cpu")
        model = model.to(device)
        model.eval()
        self.model = model
        self.model_path = model_path

    @staticmethod
    def is_downloaded() -> bool:
        return os.path.exists(get_cache_path_by_url(LAMA_MODEL_URL))

    def forward(self, images, masks, config: Config):
        """Input image and output image have same size
        image: [B, H, W, C] RGB
        mask: [B, H, W]
        return: BGR IMAGE
        """
        
        images = np.asarray([norm_img(image) for image in images])
        masks = np.asarray([norm_img(mask for mask in masks)])

        masks = (masks > 0) * 1
        images = torch.from_numpy(images).to(self.device)
        masks = torch.from_numpy(masks).to(self.device)
        print(f"Image shape = {images.shape}, mask_shape = {masks.shape}")
        inpainted_images = self.model(images, masks)

        cur_res_list = []
        for inpainted_image in inpainted_images:
            cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cur_res_list.append(cur_res)
        return cur_res_list
