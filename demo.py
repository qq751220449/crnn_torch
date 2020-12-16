import torch
from torch.autograd import Variable
from tools import utils
from data_loader.data_loader import ResizeNormalize
import cv2
import models.crnn as crnn
import argparse
import time
from config.config import Config

config = Config()
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', default="./predict/model.pth", type=str, required=False,
                    help='crnn model path')
parser.add_argument('-i', '--image_path', default="./predict/test.jpg", type=str, required=False,
                    help='demo image path')
args = parser.parse_args()

model_path = args.model_path
image_path = args.image_path

# net init
nclass = len(config.label_classes) + 1
model = crnn.CRNN(config.img_height, config.nc, nclass, config.hidden_size, config.n_layers)
if torch.cuda.is_available():
    model = model.cuda()

# load model
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

beform = time.time()

converter = utils.strLabelConverter(config.label_classes)

transformer = ResizeNormalize(config.img_height, config.img_width)

img = cv2.imread(image_path, cv2.IMREAD_COLOR)
if config.img_mode == "RGB":
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 图片格式转换
else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片格式转换
image = transformer(img)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.LongTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
all_time = time.time() - beform
print(all_time)
