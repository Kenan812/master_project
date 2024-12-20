import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from io import BytesIO
import pandas as pd
import numpy as np
import requests
from requests.exceptions import RequestException
from tqdm import tqdm
import ast

torch.backends.cudnn.benchmark = True

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

def get_preferred_image_url(images_list):
    if not isinstance(images_list, list):
        return None
    for image_info in images_list:
        if isinstance(image_info, dict):
            if image_info.get('hi_res'):
                return image_info['hi_res']
            elif image_info.get('large'):
                return image_info['large']
    return None

class ProductImageDataset(Dataset):
    def __init__(self, product_ids, image_urls, transform=None):
        self.product_ids = product_ids
        self.image_urls = image_urls
        self.transform = transform

    def __len__(self):
        return len(self.product_ids)

    def __getitem__(self, idx):
        product_id = self.product_ids[idx]
        image_url = self.image_urls[idx]

        if pd.isna(image_url) or not isinstance(image_url, str):
            return None, None

        # Attempt to download the image with a short retry mechanism
        attempts = 2
        img = None
        for _ in range(attempts):
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert('RGB')
                break
            except (RequestException, OSError):
                img = None

        if img is None:
            return None, None

        if self.transform:
            try:
                img = self.transform(img)
            except Exception:
                return None, None

        return product_id, img

def custom_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if not batch:
        return None
    batch_product_ids, batch_imgs = zip(*batch)
    batch_imgs = torch.stack(batch_imgs)
    return batch_product_ids, batch_imgs

if __name__ == "__main__":
    metadata = pd.read_csv('datasets/metadata.csv')
    metadata['images'] = metadata['images'].apply(safe_literal_eval)
    metadata['image_url'] = metadata['images'].apply(get_preferred_image_url)
    metadata = metadata[metadata['image_url'].notna()].reset_index(drop=True)
    product_ids = metadata['parent_asin'].values
    image_urls = metadata['image_url'].values

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Load pre-trained ResNet-50 model and remove the last FC layer
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
    resnet50 = resnet50.to(device)
    resnet50.eval()

    dataset = ProductImageDataset(product_ids, image_urls, transform=preprocess)

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=14,
        pin_memory=True,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    features_dict = {}
    for batch in tqdm(dataloader, total=len(dataloader)):
        if batch is None:
            continue
        batch_product_ids, valid_imgs = batch
        valid_imgs = valid_imgs.to(device, non_blocking=True)

        with torch.no_grad():
            batch_features = resnet50(valid_imgs)
            if batch_features.dim() == 4:
                batch_features = batch_features.squeeze(-1).squeeze(-1)
            batch_features = batch_features.cpu().numpy()

        for product_id, features in zip(batch_product_ids, batch_features):
            features_dict[product_id] = features

    np.savez_compressed('image_features.npz', **features_dict)
    print("Feature extraction completed and saved to 'image_features.npz'.")
