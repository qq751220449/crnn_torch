from torch.utils.data import Dataset
import cv2
import os
import torchvision.transforms as transforms


class ResizeNormalize(object):
    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = cv2.resize(img, dsize=(self.width, self.height), interpolation=self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class HubDataset(Dataset):
    def __init__(self, config, mode="train", transform=None):
        super(HubDataset, self).__init__()
        assert mode in ["train", "eval"]        # 判断数据集类型
        if mode == "train":
            data_path = config.train_data
        else:
            data_path = config.eval_data
        fp = open(data_path, "r", encoding="utf-8")
        self.image_list = []
        for line in fp.readlines():
            if line.strip():
                self.image_list.append(line.strip())

        self.transform = transform
        assert config.img_mode in ["RGB", "GRAY"]
        self.img_mode = config.img_mode

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img = cv2.imread(self.image_list[item], cv2.IMREAD_COLOR)
        if self.img_mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 图片格式转换
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)              # 图片格式转换

        label = str(os.path.basename(self.image_list[item]).split("_")[0])
        if self.transform is not None:
            img = self.transform(img)

        return img, label


