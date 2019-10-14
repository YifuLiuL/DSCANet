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



# mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# imgs_transform = stand_transforms.Compose([
#     Scale((256, 256), Image.BILINEAR),
#     stand_transforms.ToTensor(),
#     stand_transforms.Normalize(*mean_std)
# ])
# target_transform = stand_transforms.Compose([
#     Scale((256, 256), Image.BILINEAR),
#     stand_transforms.ToTensor()
# ])
# val_dataset = Pascal(root, 'val', img_transform=imgs_transform, target_transform=target_transform)
# val_dataloader = Data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4,)

def _val(dataloader, net, cuda, nums, epoch):
    net.eval()

    prediction_all = np.zeros([len(dataloader), 256, 256], dtype=int)
    target_all = np.zeros([len(dataloader), 256, 256], dtype=int)
    for i, data in enumerate(dataloader):
        img, target = data
        with torch.no_grad():
            if cuda:
                img = Variable(img).cuda()
                target = Variable(target).cuda()
            else:
                img = Variable(img)
                target = Variable(target)

        assert img.size()[2:] == target.size()[1:]
        output = net(img)
        assert output[0].size()[2:] == target.size()[1:]
        assert output[0].size()[1] == nums

        target_all[i, :, :] = target[0].data.cpu().numpy()
        prediction_all[i, :, :] = output[0][0].max(0)[1].data.cpu().numpy()
        print('val: %d/%d'%(i+1, len(dataloader)))

    acc, acc_cls, mean_iou, fwavacc = eval._evaluate(prediction_all, target_all, nums)
    print('[epoch %d], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, acc, acc_cls, mean_iou, fwavacc))
    with open('./val_data.txt','a+') as f:
        f.write('[epoch %d], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, acc, acc_cls, mean_iou, fwavacc)+'\n')

    net.train()

#if __name__ == '__main__':
    #print(len(val_dataset))