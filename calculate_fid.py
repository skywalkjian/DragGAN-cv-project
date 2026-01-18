import os
import torch
import numpy as np
from scipy import linalg
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def get_inception_features(image_paths, batch_size=32, device='cuda'):
    model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()
    
    # Remove the classification head
    model.fc = torch.nn.Identity()
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    features = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            # Inception returns a tuple in eval mode (output, aux_output)
            # But since we replaced fc, it might just return the tensor
            output = model(batch)
            if isinstance(output, tuple):
                output = output[0]
            features.append(output.cpu().numpy())
            
    return np.concatenate(features, axis=0)

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give a complex number
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def get_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def compute_fid_from_paths(path1_list, path2_list, batch_size=32, device='cuda'):
    print(f"Extracting features for {len(path1_list)} images...")
    feats1 = get_inception_features(path1_list, batch_size, device)
    mu1, sigma1 = get_statistics(feats1)
    
    print(f"Extracting features for {len(path2_list)} images...")
    feats2 = get_inception_features(path2_list, batch_size, device)
    mu2, sigma2 = get_statistics(feats2)
    
    fid_value = calculate_fid(mu1, sigma1, mu2, sigma2)
    return fid_value
