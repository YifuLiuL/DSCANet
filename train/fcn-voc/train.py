import sys
import os
import argparse

import torch
from torch.autograd import Variable
from torch import optim
import torchvision.transforms as stand_transforms
import torch.utils.data as Data
from models import *
from datasets import *
from utils import *
import tqdm
from .test import _val

root = 'D:\\pycharmProject\\pytorch_pro\\segmatic\\semantic-master\\datasets'
parser = argparse.ArgumentParser(description='This is fcn8s train!')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')   # minibatch  内的图片要整合一致
parser.add_argument('--epoches', type=int, default=100, help='epoches count')
parser.add_argument('--lr_decay', type=float, default=0.9, help='lr decay ratio')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum tatio')
parser.add_argument('--restore', type=int, default=100, help='Breakpoint retraining')
parser.add_argument('--print_freq', type=int, default=20, help='print freq num')
parser.add_argument('--weight_decay', type=float, default=2e-5, help='ratio')
parser.add_argument('--snapshot', action='store_true', help='restore?')

cuda = True if torch.cuda.is_available() else False #也可以用to(device)写法
weight = torch.ones([21])


def main(args):
    #print(arg.lr)


    #DA
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_join_transform = Compose([      #这里不能用torchvison自带的 transforms  Compose
        Scale((256, 256), Image.BILINEAR),
        RandomRotate(10),
        RandomFlip()
    ])
    imgs_transform = stand_transforms.Compose([
        stand_transforms.ToTensor(),
        stand_transforms.Normalize(*mean_std)
    ])
    target_transform = Compose([
        Lable2Tentor()
    ])
    vimgs_transform = stand_transforms.Compose([
        Scale((256, 256), Image.BILINEAR),
        stand_transforms.ToTensor(),
        stand_transforms.Normalize(*mean_std)
    ])
    vtarget_transform = stand_transforms.Compose([
        Scale((256, 256), Image.BILINEAR),
        stand_transforms.ToTensor()
    ])
    # dataloader
    train_set = Pascal(root, mode='train', train_join_transform=train_join_transform,  img_transform=imgs_transform,
                       target_transform=target_transform)
    #print(len(train_set))
    train_loader = Data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #1使用多进程，2是否将数据保存在pinmemory区，pin memory中的数据转到GPU会快一些，但是对内存要求较大

    val_dataset = Pascal(root, 'val', img_transform=vimgs_transform, target_transform=vtarget_transform)
    val_dataloader = Data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, )

    ploter = Visualize(env_name='main')
    #loss function
    if cuda:
        net = torch.nn.DataParallel(Fcn8s(num_classes=21)).cuda()
        if args.snapshot:
            net.load_state_dict(torch.load('../../savepth/fcn-deconv-100.pth'))
        # optimizer
        loss_function = CrossEntropyLoss(weight, size_average=True, ignore_index=255).cuda()  #  这里也可以不需要 ig
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)


    #print(net)
    net.train()
    for epoch in range(1, args.epoches+1):
        curr_iter = (epoch - 1) * len(train_loader)
        runing_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(train_loader)):
            #print(type(data))
            #print(data[0].size())

            imgs, labels = data
            input = Variable(imgs.cuda())
            target = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = net(input)

            assert outputs[0].size()[2:] == target.size()[1:]
            assert outputs.size()[1] == 21

            loss = loss_function(outputs[0], target)


            loss.backward()
            optimizer.step()

            curr_iter += 1
            runing_loss += loss.data[0]

            ploter._plot('loss', 'train', curr_iter, runing_loss/(i+1))
            if (i+1) %args.print_freq == 0:
                print("[Epoch %d], [iter %d / %d], [Loss %.4f]" % (
                      epoch, i+1, len(train_loader), loss.data[0]))

        _val(val_dataloader, net, cuda, 21, epoch)
        #runing_loss = 0  #每一个epoch内的loss取平均

        if (epoch+1)% 20 == 0:
            args.lr /= 10
            print(args.lr)
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
            torch.save(net.state_dict(),'../../savepth/fcn-models-%d.pth'% (epoch+1))

    torch.save(net.state_dict(), '../../savepth/fcn-models-%d.pth' % (epoch + 1))









if __name__ == '__main__':
    args = parser.parse_args()
    main(args)