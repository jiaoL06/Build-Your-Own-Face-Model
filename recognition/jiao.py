# -*- coding: utf-8 -*-
import os
import os.path as osp

import torch
import torchvision.transforms as T
import torch.nn as nn
import numpy as np
from PIL import Image

from config import config as conf
from model import FaceMobileNet


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def feature(img):
    img = test_transform(img)
    data = img[:,None,:,:]
    data = data.to(device)
    net = model.to(device)
    with torch.no_grad():
        feature = net(data)
        feature1 = feature[0].cpu().numpy()
    return feature1

if __name__ == '__main__':
    embedding_size = 512
    test_model = "checkpoints/24.pth"
    device = "cuda"
    input_shape = [1, 128, 128]
    test_transform = T.Compose([
        T.Grayscale(),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    model = FaceMobileNet(embedding_size)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(test_model, map_location=device))
    model.eval()

    img_path = "/home/ubuntu/job/arcface/Build-Your-Own-Face-Model/recognition/img/002.png"
    img_path2 = "/home/ubuntu/job/arcface/Build-Your-Own-Face-Model/recognition/img/003.png"
    img = Image.open(img_path)
    img2 = Image.open(img_path2)
    feature1 = feature(img)
    feature2 = feature(img2)
    similarity = cosin_metric(feature1, feature2)
    print(similarity)
