#encoding=utf-8
import sys
import os
import argparse

import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torchvision.transforms as stand_transforms
import torch.utils.data as Data
from models import *
from datasets import *
from utils import *
import tqdm
# from .test import _val
from option import Options
from torch.nn.parallel.scatter_gather import gather

class Trainer():
    def __init__(self, args):
        self.args = args
        weight = torch.ones([21])
        train_join_transform = Compose([  # 这里不能用torchvison自带的 transforms  Compose
            transform_sync(520, 480)
        ])
        imgs_transform = stand_transforms.Compose([
            stand_transforms.ToTensor(),
            stand_transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        target_transform = stand_transforms.Compose([
            Lable2Tentor()
        ])
        val_join_transform = Compose([
           val_transform_sync(520, 480)
        ])

        train_set = PascalAug(root, mode='train', train_join_transform=train_join_transform,  img_transform=imgs_transform,
                       target_transform=target_transform)
        self.trainloader = Data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        # 1使用多进程，2是否将数据保存在pinmemory区，pin memory中的数据转到GPU会快一些，但是对内存要求较大
        val_set = PascalAug(root, 'val', val_join_transform= val_join_transform, img_transform=imgs_transform,
                                target_transform=target_transform)
        self.val_dataloader = Data.DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4, )
        self.num_class = train_set.num_class
        self.num_class = 21

        net_g = get_generator_net(pretrained=False, init_type='normal', numclasses=self.num_class,
                                  backbone=args.backbone, use_aux=True,
                                  norm_layer=nn.BatchNorm2d)
        net_d = get_discriminator_net(in_channels2=3, init_type='normal', init_gain=0.02, hd_channels=64, patch_size=32,
                                      n_layers=3, norm_layer=nn.BatchNorm2d)

        print(net_g);  print(net_d)

        # optimizer using different LR
        self.optimizer_g = optim.Adam(net_g.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        self.optimizer_d = optim.Adam(net_d.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

        # params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
        # if hasattr(model, 'head'):
        #     params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
        # if hasattr(model, 'auxlayer'):
        #     params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr*10})
        # optimizer = torch.optim.SGD(params_list, lr=args.lr,
        #     momentum=args.momentum, weight_decay=args.weight_decay)
        assert 1 < 0 , 'it is over'

        if args.cuda:
            self.model_g = nn.DataParallel(net_g).cuda()
            self.model_d = nn.DataParallel(net_d).cuda()

            self.criterion_GAN = GANLoss().cuda()
            self.criterion_MCE = CrossEntropyLoss().cuda()
        # Lr Poly
        self.scheduler_g = get_lr_scheduler(self.optimizer_g, args)
        self.scheduler_d = get_lr_scheduler(self.optimizer_d, args)
        self.best_pred = 0.0

    def train(self, epoch):
        # train_loss = 0.0
        self.model_g.train()
        self.model_d.train()
        tbar = tqdm(self.trainloader)
        for i, (image, target) in enumerate(tbar):
            ######
            #self.scheduler(self.optimizer, i, epoch, self.best_pred)
            #####
            image = Variable(image.cuda())
            target = Variable(target.cuda())
            outputs = self.model_g(image)
            output_seg, outpt_fake = tuple(outputs)

            ######################
            # (1) Update D network
            ######################
            self.optimizer_d.zero_grad()
            fake_xg = torch.cat((image, outpt_fake), dim=1)
            output_fake = self.model_d(fake_xg.detach())
            loss_d_fake = self.criterion_GAN(output_fake, False)
            ####
            real_xy = torch.cat((image, target), dim=1)
            outpt_real = self.model_d(real_xy.detach())
            loss_d_real = self.criterion_GAN(outpt_real, True)

            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()
            self.optimizer_d.step()
            ######################
            # (2) Update G network
            ######################
            self.optimizer_g.zero_grad()

            fake_xg = torch.cat((image, outpt_fake), dim=1)
            outpt_fake = self.model_d(fake_xg)
            loss_g_real = self.criterion_GAN(outpt_fake, True)
            loss_seg = self.criterion_MCE(output_seg, target)
            loss_g = loss_seg + 0.4 * loss_g_real

            loss_g.backward()
            self.optimizer_g.step()

            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                epoch, i, len(self.trainloader), loss_d.item(), loss_g.item()))
        update_lr(self.scheduler_g, self.optimizer_g, 'g-')
        update_lr(self.scheduler_d, self.optimizer_d, 'd-')
        # train_loss += loss.item()
        # tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
        # if epoch % 10 == 0:
        #     # save checkpoint every epoch
        #     is_best = False
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': self.model.module.state_dict(),
        #         'optimizer_g': self.optimizer_g.state_dict(),
        #         'optimizer_d': self.optimizer_d.state_dict(),
        #         'best_pred': self.best_pred,
        #     }, self.args, is_best)

    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target):
            outputs = model(image)
            #outputs = gather(outputs, 0, dim=0)      #把 Gathers tensors from different GPUs on a specified device

            pred = outputs[0]

            correct, labeled = batch_pix_accuracy(pred.data, target)
            inter, union = batch_intersection_union(pred.data, target, self.nclass)
            return correct, labeled, inter, union

        is_best = False
        self.model_g.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        tbar = tqdm(self.val_dataloader, desc='\r')
        for i, (image, target) in enumerate(tbar):
            # if torch_ver == "0.3":
            #     image = Variable(image, volatile=True)
            #     correct, labeled, inter, union = eval_batch(self.model, image, target)
            # else:
            with torch.no_grad():
                image = Variable(image.cuda())
                target = Variable(target.cuda())
                correct, labeled, inter, union = eval_batch(self.model_g, image, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            tbar.set_description(
                'pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))

        new_pred = (pixAcc + mIoU)/2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer_g': self.optimizer_g.state_dict(),
                'optimizer_d': self.optimizer_d.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best)


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)