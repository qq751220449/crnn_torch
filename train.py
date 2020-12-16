from config.config import Config
from data_loader.data_loader import HubDataset, ResizeNormalize
import torch
from data_loader import alignCollate
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from models.crnn import CRNN
from tools import utils
from torch.nn import CTCLoss
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm


def main():
    config = Config()

    if not os.path.exists(config.expr_dir):
        os.makedirs(config.expr_dir)

    if torch.cuda.is_available() and not config.use_cuda:
        print("WARNING: You have a CUDA device, so you should probably set cuda in params.py to True")

    # 加载训练数据集
    train_dataset = HubDataset(config, "train", transform=None)

    train_kwargs = {'num_workers': 2, 'pin_memory': True,
                    'collate_fn': alignCollate(config.img_height, config.img_width, config.keep_ratio)} if torch.cuda.is_available() else {}

    training_data_batch = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=False, **train_kwargs)

    # 加载定长校验数据集
    eval_dataset = HubDataset(config, "eval", transform=transforms.Compose([ResizeNormalize(config.img_height, config.img_width)]))
    eval_kwargs = {'num_workers': 2, 'pin_memory': False} if torch.cuda.is_available() else {}
    eval_data_batch = DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False, drop_last=False, **eval_kwargs)

    # 加载不定长校验数据集
    # eval_dataset = HubDataset(config, "eval")
    # eval_kwargs = {'num_workers': 2, 'pin_memory': False,
    #                'collate_fn': alignCollate(config.img_height, config.img_width, config.keep_ratio)} if torch.cuda.is_available() else {}
    # eval_data_batch = DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False, drop_last=False, **eval_kwargs)

    # 定义网络模型
    nclass = len(config.label_classes) + 1
    crnn = CRNN(config.img_height, config.nc, nclass, config.hidden_size, n_rnn=config.n_layers)
    # 加载预训练模型
    if config.pretrained != '':
        print('loading pretrained model from %s' % config.pretrained)
        crnn.load_state_dict(torch.load(config.pretrained))
    print(crnn)

    # Compute average for `torch.Variable` and `torch.Tensor`.
    loss_avg = utils.averager()

    # Convert between str and label.
    converter = utils.strLabelConverter(config.label_classes)

    criterion = CTCLoss()           # 定义损失函数

    # 设置占位符
    image = torch.FloatTensor(config.train_batch_size, 3, config.img_height, config.img_height)
    text = torch.LongTensor(config.train_batch_size * 5)
    length = torch.LongTensor(config.train_batch_size)

    if config.use_cuda and torch.cuda.is_available():
        criterion = criterion.cuda()
        image = image.cuda()
        crnn = crnn.to(config.device)

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    # 设定优化器
    if config.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    elif config.adadelta:
        optimizer = optim.Adadelta(crnn.parameters())
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=config.lr)

    def val(net, criterion, eval_data_batch):
        print('Start val')
        for p in crnn.parameters():
            p.requires_grad = False
        net.eval()

        n_correct = 0
        loss_avg_eval = utils.averager()
        for data in eval_data_batch:
            cpu_images, cpu_texts = data
            batch_size = cpu_images.size(0)
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts)
            utils.loadData(text, t)
            utils.loadData(length, l)
            preds = crnn(image)
            preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
            cost = criterion(preds, text, preds_size, length) / batch_size
            loss_avg_eval.add(cost)         # 计算loss

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            cpu_texts_decode = []
            for i in cpu_texts:
                cpu_texts_decode.append(i)
            for pred, target in zip(sim_preds, cpu_texts_decode):       # 计算准确率
                if pred == target:
                    n_correct += 1

            raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.n_val_disp]
            for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
                print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

        accuracy = n_correct / float(len(eval_dataset))
        print('Val loss: %f, accuray: %f' % (loss_avg.val(), accuracy))

    # 训练每个batch数据
    def train(net, criterion, optimizer, data):
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)             # 计算当前batch_size大小
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)          # 转换为类别
        utils.loadData(text, t)
        utils.loadData(length, l)
        optimizer.zero_grad()                       # 清零梯度
        preds = net(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        cost.backward()
        optimizer.step()
        return cost

    for epoch in range(config.nepoch):
        i = 0
        for batch_data in training_data_batch:
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()
            cost = train(crnn, criterion, optimizer, batch_data)
            loss_avg.add(cost)
            i += 1

            if i % config.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, config.nepoch, i, len(training_data_batch), loss_avg.val()))
                loss_avg.reset()

            # if i % config.valInterval == 0:
            #     val(crnn, criterion, eval_data_batch)
            #
            # # do checkpointing
            # if i % config.saveInterval == 0:
            #     torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(config.expr_dir, epoch, i))

        val(crnn, criterion, eval_data_batch)
        torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_end.pth'.format(config.expr_dir, epoch))


if __name__ == "__main__":
    main()
