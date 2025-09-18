import os
import cv2
import numpy as np
import scanpy as sc
import anndata as ad
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from tqdm import tqdm
from itertools import product

from typing import Literal, Union, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


from transformers import AutoModel
from torchvision import transforms

import hashlib, os, json

class CacheManager:
    def __init__(self, base_dir="~/.panospace_cache"):
        self.base_dir = os.path.expanduser(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def compute_cache_id(self, img_path, params: dict):
        # 生成唯一 hash
        key_str = json.dumps({"img_path": img_path, **params}, sort_keys=True)
        return hashlib.sha1(key_str.encode()).hexdigest()[:12]

    def get_cache_path(self, img_path, params: dict):
        cache_id = self.compute_cache_id(img_path, params)
        cache_path = os.path.join(self.base_dir, cache_id)
        os.makedirs(cache_path, exist_ok=True)
        return cache_path
    
class DINOv2NeighborDataset(Dataset):
    def __init__(self, centers, img_path, label_frame=None, train=True,
                 radius=129, neighb=3):
        self.centers = centers
        self.label_frame = label_frame
        self.train = train
        self.radius = radius
        self.neighb = neighb

        self.image = Image.open(img_path).convert("RGB")
        self.image.load()

        self.transform = ImageTransform(resize=518, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        x, y = self.centers[index]
        r, n = self.radius, self.neighb

        crop = self.image.crop((x - r, y - r, x + r, y + r))
        crop_neighbor = self.image.crop((x - r * n, y - r * n, x + r * n, y + r * n))

        if self.train:
            crop = self.transform(img=crop, phase="valid")
            crop_neighbor = self.transform(img=crop_neighbor, phase="valid")
            label = self.label_frame.iloc[index, :].values.astype(np.float32)
            return crop, crop_neighbor, label
        else:
            crop = self.transform(img=crop, phase="valid")
            crop_neighbor = self.transform(img=crop_neighbor, phase="valid")
            return crop, crop_neighbor

    def __len__(self):
        return len(self.centers)
    

class DINOv2NeighborClassifier(pl.LightningModule):
    def __init__(self, num_classes=9, class_weights=None, learning_rate=1e-4,
                 local_path=".cache/dinov2-base",
                 pretrained_model_name="facebook/dinov2-base"):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.vit = self._load_dinov2_model(local_path, pretrained_model_name)
        # self.vit = AutoModel.from_pretrained('/mnt/Fold/hehf/dinov2-base', local_files_only=True)

        for param in self.vit.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )

    def _load_dinov2_model(self, local_path: str, pretrained_model_name: str):
        """
        Attempt to load DINOv2 from local path first, fallback to online download.
        Raises error with instructions if both fail.
        """
        # 1. Try loading from local path
        try:
            model = AutoModel.from_pretrained(local_path, local_files_only=True)
            print(f"[INFO] Successfully loaded DINOv2 from local path: {local_path}")
            return model
        except Exception as e_local:
            print(f"[WARN] Failed to load DINOv2 locally: {e_local}")

        # 2. Try downloading from Hugging Face
        try:
            print(f"[INFO] Attempting to download DINOv2 model from Hugging Face: {pretrained_model_name}")
            model = AutoModel.from_pretrained(pretrained_model_name)
            print(f"[INFO] Successfully downloaded DINOv2 from Hugging Face")
            return model
        except Exception as e_remote:
            raise RuntimeError(
                f"Failed to load DINOv2 both locally and online.\n"
                f"Local path tried: {local_path}\n"
                f"Hugging Face model: {pretrained_model_name}\n"
                f"Error details:\nLocal load error: {e_local}\nRemote download error: {e_remote}\n\n"
                "Please check your internet connection or manually download the model from:\n"
                "https://huggingface.co/facebook/dinov2-base\n"
                "and place it in the specified local path."
            )
        
    def forward(self, crop, crop_neighbor):
        x1 = self._extract_feature(crop)
        x2 = self._extract_feature(crop_neighbor)
        x = torch.cat((x1, x2), dim=1)  # [B, 1536]
        x = self.classifier(x)
        x = self.softmax(x)
        return x

    def _extract_feature(self, images):
        with torch.no_grad():
            outputs = self.vit(images)
        cls_embedding = outputs.pooler_output  # [B, 768]
        return cls_embedding

    def training_step(self, batch, batch_idx):
        crop, crop_neighbor, label = batch
        pred = self(crop, crop_neighbor)  # [B, num_classes]
        loss = F.kl_div(pred.log(), label, reduction='none')

        if self.class_weights is not None:
            weights = self.class_weights.unsqueeze(0).to(self.device)
            loss = loss * weights

        loss = torch.mean(torch.sum(loss, dim=1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(list(self.classifier.parameters()), lr=self.learning_rate)
        return optimizer
    
    
class DINOv2_superres_deconv(object):
    def __init__(self,
                 deconv_adata,
                 img_dir,
                 radius=129,
                 neighb=2,
                 class_weights=None,
                 learning_rate=1e-4,
                 local_path="~/.panospace_cache/dinov2-base",
                 pretrained_model_name="facebook/dinov2-base",
                 cache_dir="~/.panospace_cache"):
        
        self.img_dir = img_dir
        self.deconv_adata = deconv_adata
        self.cell_type_name = list(self.deconv_adata.uns['celltype'])
        num_classes = len(self.cell_type_name)

        params = {
            "radius": radius,
            "neighb": neighb,
            "num_classes": num_classes,
            "celltypes": self.cell_type_name
        }
        cache_manager = CacheManager(base_dir=cache_dir)
        self.path = cache_manager.get_cache_path(img_dir, params)

        self.num_classes = num_classes
        self.radius=radius
        self.neighb = neighb

        if os.path.exists(os.path.join(self.path,"superres_model.ckpt")):
            print('the checkpoint exists, loading the checkpoint...')
            print('if use the checkpoint, do not execute run_train method')
            self.model = DINOv2NeighborClassifier.load_from_checkpoint(os.path.join(self.path,"superres_model.ckpt"),num_classes=num_classes)
        else:
            self.model = DINOv2NeighborClassifier(num_classes=num_classes,
                                                  class_weights=class_weights,
                                                  learning_rate=learning_rate,
                                                  local_path=local_path,
                                                  pretrained_model_name=pretrained_model_name)
        print('model loaded...')
        print('loading super res data')
        if not os.path.exists(os.path.join(self.path,'sr_adata.h5ad')):
            self.sr_adata = self.make_sr_datalist()
            self.sr_adata.write(os.path.join(self.path,'sr_adata.h5ad'))
        else:
            self.sr_adata = sc.read(os.path.join(self.path,'sr_adata.h5ad'))

    def make_sr_datalist(self):
        # 提取参数
        r = self.deconv_adata.uns['radius']
        spot_centers = self.deconv_adata.obsm['spatial']

        # 构建更高分辨率的网格坐标（subspot）
        axis_x = range(spot_centers[:, 0].min().astype(int),
                    spot_centers[:, 0].max().astype(int), r)
        axis_y = range(spot_centers[:, 1].min().astype(int),
                    spot_centers[:, 1].max().astype(int), r)
        subspot_centers = np.array([*product(axis_x, axis_y)])

        # 读取图像并获取组织轮廓 mask
        img = np.array(Image.open(self.img_dir))
        cnt = cv2_detect_contour(img)  # 需保证返回格式为符合 cv2.drawContours 的格式
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        _ = cv2.drawContours(mask, [cnt], contourIdx=-1, color=255, thickness=-1)
        mask = (mask > 0)  # bool 型掩膜

        # 过滤 subspot：去除掩膜外的点 + 超出图像边界的点
        valid_centers = []
        for x, y in subspot_centers:
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                if mask[y, x]:  # 注意：行是 y，列是 x
                    valid_centers.append((x, y))

        subspot_centers = np.array(valid_centers)

        # 构建高分辨 AnnData（表达矩阵初始化为全 0）
        sr_adata = ad.AnnData(np.zeros((subspot_centers.shape[0], 1)))
        sr_adata.obsm['spatial'] = subspot_centers

        return sr_adata
    
    def pred(self,dataloader,device=torch.device('cuda')):
        self.model.eval()
        model = self.model.to(device)

        pred = np.zeros((len(dataloader.dataset), self.num_classes))

        current_index = 0
        for crop, crop_neighbor in tqdm(dataloader):
            crop = crop.to(device)
            crop_neighbor = crop_neighbor.to(device)
            outputs = model(crop, crop_neighbor)
            batch_size = outputs.shape[0]
            pred[current_index:current_index+batch_size] = outputs.cpu().detach().squeeze(1).numpy()
            current_index += batch_size
        return pred
    
    def run_train(self, epoch=50, batch_size=256, num_workers=4, accelerator='gpu'):
        deconv = self.deconv_adata.obs[self.cell_type_name]
        deconv[deconv < 0] = 0
        deconv = (deconv.T/deconv.sum(1)).T

        dataset = DINOv2NeighborDataset(centers=self.deconv_adata.obsm['spatial'],
                                        img_path=self.img_dir,
                                        label_frame=deconv,
                                        radius=self.radius,
                                        neighb=self.neighb,
                                        )

        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        trainer = pl.Trainer(max_epochs=epoch, accelerator=accelerator, devices=1,logger=False,enable_checkpointing=False)
        trainer.fit(self.model, dataloader)

        trainer.save_checkpoint(os.path.join(self.path,"superres_model.ckpt"))

    def run_superres(self):
        dataset = DINOv2NeighborDataset(centers=self.sr_adata.obsm['spatial'],
                                        img_path=self.img_dir,
                                        label_frame=None,
                                        train=False,
                                        radius=self.radius,
                                        neighb=self.neighb,
                                        )

        dataloader = DataLoader(dataset, batch_size=256, num_workers=4)
        predict = self.pred(dataloader)
        # print(predict)
        self.sr_adata.obs[self.cell_type_name] = predict

        return self.sr_adata


class ImageTransform:
    def __init__(self, resize: int, mean: list[float], std: list[float]):
        self.resize = resize

        self.base_resize = transforms.Compose([
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resize)
        ])

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.augmentations = {
            'flip': transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]),
            'noise': transforms.Compose([
                transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0)),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            ]),
            'blur': transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0)),
            'dist': transforms.Compose([
                transforms.RandomAffine(degrees=30, shear=10),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            ]),
            'contrast': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            'color': transforms.ColorJitter(hue=0.1),
            'crop': transforms.RandomResizedCrop(size=resize, scale=(0.5, 1.0)),
            'random': transforms.Compose([
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0)),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
                ], p=0.5),
                transforms.RandomApply([
                    transforms.RandomAffine(degrees=30, shear=10),
                    transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
                ], p=0.5),
            ])
        }

        self.valid_transform = transforms.Compose([
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img: Image.Image, phase: Literal['train', 'valid'] = 'train', param: str = 'none') -> torch.Tensor:
        if phase == 'train':
            img = self.base_resize(img)

            if param != 'none':
                for para in param.split(','):
                    if para in self.augmentations:
                        img = self.augmentations[para](img)
                    else:
                        raise ValueError(f"Unknown augmentation parameter: {para}")

            img = self.to_tensor(img)

        elif phase == 'valid':
            img = self.valid_transform(img)

        else:
            raise ValueError("phase must be 'train' or 'valid'")

        return img
    
    def transform_batch(self, imgs: Union[List[Image.Image], Image.Image], phase="train", param='none') -> torch.Tensor:
        if isinstance(imgs, Image.Image):
            imgs = [imgs]

        results = [self.__call__(img, phase=phase, param=param) for img in imgs]
        return torch.stack(results)  # shape: [B, C, H, W]
    
def cv2_detect_contour(
    img,
    CANNY_THRESH_1 = 100,
    CANNY_THRESH_2 = 200,
    apertureSize=5,
    L2gradient = True,
    all_cnt_info=False
):
	if len(img.shape)==3:
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	elif len(img.shape)==2:
		gray=(img*((1, 255)[np.max(img)<=1])).astype(np.uint8)
	else:
		print("Image format error!")
	edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2,apertureSize = apertureSize, L2gradient = L2gradient)
	edges = cv2.dilate(edges, None)
	edges = cv2.erode(edges, None)
	cnt_info = []
	cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	for c in cnts:
		cnt_info.append((c,cv2.isContourConvex(c),cv2.contourArea(c),))
	cnt_info = sorted(cnt_info, key=lambda c: c[2], reverse=True)
	cnt=cnt_info[0][0]
	if all_cnt_info:
		return cnt_info
	else:
		return cnt