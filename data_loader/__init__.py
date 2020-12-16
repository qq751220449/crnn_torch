# -*- coding: utf-8 -*-
# @Time    : 2020/11/30 13：56
# @Author  : liyujun
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms


class alignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1, interpolation=cv2.INTER_LINEAR):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.interpolation = interpolation

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                h, w = image.shape[:2]
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))      # 保持高度不变
            imgW = max(imgH * self.min_ratio, imgW)     # assure imgH >= imgW

        transform = transforms.ToTensor()
        images = [cv2.resize(image, (imgW, imgH), interpolation=self.interpolation) for image in images]
        images = [transform(image) for image in images]
        for image in images:
            image.sub_(0.5).div_(0.5)
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels