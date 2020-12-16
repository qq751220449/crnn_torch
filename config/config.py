import os
import torch
base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../"))


class Config(object):
    def __init__(self):
        super(Config, self).__init__()
        self.train_data = os.path.abspath(os.path.join(base_dir, "dataset/train.txt"))
        self.eval_data = os.path.abspath(os.path.join(base_dir, "dataset/eval.txt"))
        assert os.path.exists(self.train_data) and os.path.exists(self.eval_data)

        # 定义识别的标签类别
        self.label_classes = "0123456789YAB"

        self.use_cuda = True

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_cuda else "cpu")           # 检测是否存在GPU

        self.train_batch_size = 2                              # 训练batch_size       不定长输入时,这里的batch_size不适合过大
        self.eval_batch_size = 10                               # 校验batch_size

        self.img_height = 64            # 图像高度缩放至固定尺寸       注意设定时，改变下采样的次数
        self.img_width = 320            # 图像宽度缩放至固定尺寸
        self.keep_ratio = True          # 缩放图片是否保持长宽比

        self.hidden_size = 256          # 循环神经网络隐藏层size

        self.img_mode = "RGB"           # 只能是RGB或GRAY中的一种

        self.nc = 3 if self.img_mode == "RGB" else 1                    # 初始图像通道数
        self.n_layers = 1               # 循环神经网络的隐藏层层数

        self.expr_dir = "outputs"
        self.manualSeed = 1234          # 设置随机数种子

        self.pretrained = ""

        self.adam = False               # whether to use adam (default is rmsprop)
        self.adadelta = False           # whether to use adadelta (default is rmsprop)
        self.lr = 0.0001                # learning rate for Critic, not used by adadealta
        self.beta1 = 0.5                # beta1 for adam. default=0.5

        # training process
        self.displayInterval = 20       # interval to be print the train loss
        self.valInterval = 30           # interval to val the model loss and accuray
        self.saveInterval = 20          # interval to save model
        self.n_val_disp = 1             # number of samples to display when val the model

        self.nepoch = 1000              # number of epochs to train for