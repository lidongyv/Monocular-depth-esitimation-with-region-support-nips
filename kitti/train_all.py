# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 13:41:34
# @Last Modified by:   yulidong
# @Last Modified time: 2018-07-27 10:22:28
import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
sys.path.append('../')
from rsden.models import get_model
from rsden.loader import get_loader, get_data_path
from rsden.metrics import runningScore
from rsden.loss import *
from rsden.augmentations import *
import os


def train(args):

    # Setup Augmentations
    data_aug = Compose([RandomRotate(10),
                        RandomHorizontallyFlip()])
    loss_rec=[]
    best_error=2
    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True,
                           split='train', img_size=(args.img_rows, args.img_cols),task='all')
    v_loader = data_loader(data_path, is_transform=True,
                           split='eval', img_size=(args.img_rows, args.img_cols),task='all')


    trainloader = data.DataLoader(
        t_loader, batch_size=args.batch_size, num_workers=2, shuffle=True)
    valloader = data.DataLoader(
        v_loader, batch_size=args.batch_size, num_workers=2)



    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()

        loss_window1 = vis.line(X=torch.zeros((1,)).cpu(),
                               Y=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='minibatches',
                                         ylabel='Loss',
                                         title='Training Loss1',
                                         legend=['Loss1']))
        loss_window2 = vis.line(X=torch.zeros((1,)).cpu(),
                               Y=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='minibatches',
                                         ylabel='Loss',
                                         title='Training Loss2',
                                         legend=['Loss']))
        loss_window3 = vis.line(X=torch.zeros((1,)).cpu(),
                               Y=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='minibatches',
                                         ylabel='Loss',
                                         title='Training Loss3',
                                         legend=['Loss3']))                                                 
        pre_window1 = vis.image(
            np.random.rand(args.img_rows, args.img_cols),
            opts=dict(title='predict1!', caption='predict1.'),
        )
        pre_window2 = vis.image(
            np.random.rand(args.img_rows, args.img_cols),
            opts=dict(title='predict2!', caption='predict2.'),
        )
        pre_window3 = vis.image(
            np.random.rand(args.img_rows, args.img_cols),
            opts=dict(title='predict3!', caption='predict3.'),
        )

        ground_window = vis.image(
            np.random.rand(args.img_rows, args.img_cols),
            opts=dict(title='ground!', caption='ground.')
            ),
        region_window = vis.image(
            np.random.rand(args.img_rows, args.img_cols),
            opts=dict(title='region!', caption='region.'),            
        )
    cuda0=torch.device('cuda:0')
    cuda1=torch.device('cuda:1')
    cuda2=torch.device('cuda:2')
    cuda3=torch.device('cuda:3')
    # Setup Model
    rsnet = get_model('rsnet')
    rsnet = torch.nn.DataParallel(rsnet, device_ids=[0,1])
    rsnet.cuda(cuda0)
    drnet=get_model('drnet')
    drnet = torch.nn.DataParallel(drnet, device_ids=[2,3])
    drnet.cuda(cuda2)
    parameters=list(rsnet.parameters())+list(drnet.parameters())

    if hasattr(drnet.module, 'optimizer'):
        optimizer = drnet.module.optimizer
    else:
        optimizer = torch.optim.Adam(
            rsnet.parameters(), lr=args.l_rate,weight_decay=5e-4,betas=(0.9,0.999))
        # optimizer = torch.optim.SGD(
        #     rsnet.parameters(), lr=args.l_rate,momentum=0.99, weight_decay=5e-4)
    if hasattr(rsnet.module, 'loss'):
        print('Using custom loss')
        loss_fn = rsnet.module.loss
    else:
        loss_fn = log_r_kitti
        #loss_fn = region_r
    trained=0
    scale=100

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint=torch.load('/home/lidong/Documents/RSDEN/RSDEN/rsnet_nyu_best_model.pkl')
            rsnet.load_state_dict(checkpoint['model_state'])
            #optimizer.load_state_dict(checkpoint['optimizer_state'])
            trained=checkpoint['epoch']
            best_error=checkpoint['error']
            print('load success from rsnet %.d'%trained)
            checkpoint=torch.load('/home/lidong/Documents/RSDEN/RSDEN/drnet_nyu_best_model.pkl')
            drnet.load_state_dict(checkpoint['model_state'])
            #optimizer.load_state_dict(checkpoint['optimizer_state'])
            trained=checkpoint['epoch']
            print('load success from drnet %.d'%trained)

            #print('load success!')
            loss_rec=np.load('/home/lidong/Documents/RSDEN/RSDEN/kitti/loss.npy')
            loss_rec=list(loss_rec)
            loss_rec=loss_rec[:85898*trained]
            # for i in range(300):
            #     loss_rec[i][1]=loss_rec[i+300][1]
            for l in range(int(len(loss_rec)/85898)):
                if args.visdom:
                    
                    vis.line(
                        X=torch.ones(1).cpu() * loss_rec[l*85898][0],
                        Y=np.mean(np.array(loss_rec[l*85898:(l+1)*85898])[:,1])*torch.ones(1).cpu(),
                        win=old_window,
                        update='append')
            
    else:


        print('Initialize seperately!')
        checkpoint=torch.load('/home/lidong/Documents/RSDEN/RSDEN/kitti/rsnet_kitti_0_27000_model.pkl')
        rsnet.load_state_dict(checkpoint['model_state'])
        trained=0
        best_error=100
        print(best_error)
        print('load success from rsnet %.d'%trained)
        checkpoint=torch.load('/home/lidong/Documents/RSDEN/RSDEN/kitti/drnet_kitti_0_27000_model.pkl')

        drnet.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        trained=0
        print('load success from drnet %.d'%trained)
        #trained=27000
        loss_rec=[]

    for epoch in range(trained, args.n_epoch):


        #trained
        print('training!')
        rsnet.train()
        drnet.train()

        for i, (images, labels) in enumerate(trainloader):
            
            images = images.cuda()
            

            optimizer.zero_grad()

            region_support = rsnet(images)

            coarse_depth=torch.cat([images,region_support],1)
            coarse_depth=torch.cat([coarse_depth,region_support],1)

            outputs=drnet(coarse_depth)



            labels = labels.cuda(cuda2)
            #linear_error=torch.where(target>0,target-pre[0])
            loss = loss_fn(input=outputs, target=labels)
            out=0.2*loss[0]+0.3*loss[1]+0.5*loss[2]
            out.backward()
            optimizer.step()

            if args.visdom:

                vis.line(
                    X=torch.ones(1).cpu() * i+torch.ones(1).cpu() *(epoch-trained)*85898,
                    Y=loss[0].item()*torch.ones(1).cpu(),
                    win=loss_window1,
                    update='append')
                vis.line(
                    X=torch.ones(1).cpu() * i+torch.ones(1).cpu() *(epoch-trained)*85898,
                    Y=loss[1].item()*torch.ones(1).cpu(),
                    win=loss_window2,
                    update='append')
                vis.line(
                    X=torch.ones(1).cpu() * i+torch.ones(1).cpu() *(epoch-trained)*85898,
                    Y=loss[2].item()*torch.ones(1).cpu(),
                    win=loss_window3,
                    update='append')
                pre = outputs[0].data.cpu().numpy().astype('float32')
                pre = pre[0,:, :]

                pre = (np.reshape(pre, [args.img_rows, args.img_cols]).astype('float32')-np.min(pre))/(np.max(pre)-np.min(pre))

                vis.image(
                    pre,
                    opts=dict(title='predict1!', caption='predict1.'),
                    win=pre_window1,
                )
                pre = outputs[1].data.cpu().numpy().astype('float32')
                pre = pre[0,:, :]

                pre = (np.reshape(pre, [args.img_rows, args.img_cols]).astype('float32')-np.min(pre))/(np.max(pre)-np.min(pre))

                vis.image(
                    pre,
                    opts=dict(title='predict2!', caption='predict2.'),
                    win=pre_window2,
                )
                pre = outputs[2].data.cpu().numpy().astype('float32')
                pre = pre[0,:, :]

                pre = (np.reshape(pre, [args.img_rows, args.img_cols]).astype('float32')-np.min(pre))/(np.max(pre)-np.min(pre))

                vis.image(
                    pre,
                    opts=dict(title='predict3!', caption='predict3.'),
                    win=pre_window3,
                )
                ground=labels.data.cpu().numpy().astype('float32')
                #print(ground.shape)
                ground = ground[0, :, :]
                ground = (np.reshape(ground, [args.img_rows, args.img_cols]).astype('float32')-np.min(ground))/(np.max(ground)-np.min(ground))
                vis.image(
                    ground,
                    opts=dict(title='ground!', caption='ground.'),
                    win=ground_window,
                )
                region_vis=region_support.data.cpu().numpy().astype('float32')
                #print(ground.shape)
                region_vis = region_vis[0, :, :]
                region_vis = (np.reshape(region_vis, [args.img_rows, args.img_cols]).astype('float32')-np.min(region_vis))/(np.max(region_vis)-np.min(region_vis))
                vis.image(
                    region_vis,
                    opts=dict(title='region_vis!', caption='region_vis.'),
                    win=region_window,
                )

            loss_rec.append([i+epoch*85898,torch.Tensor([loss[0].item()]).unsqueeze(0).cpu(),torch.Tensor([loss[1].item()]).unsqueeze(0).cpu(),torch.Tensor([loss[2].item()]).unsqueeze(0).cpu()])
            print("data [%d/85898/%d/%d] Loss1: %.4f Loss2: %.4f Loss3: %.4f out:%.4f " % (i+27001, epoch, args.n_epoch,loss[0].item(),loss[1].item(),loss[2].item(),out.item()))
            if i%1000==0:
                i=i+27001
                #best_error = error
                state = {'epoch': epoch+1,
                         'model_state': rsnet.state_dict(),
                         'optimizer_state': optimizer.state_dict(),
                         'error': out.item(),}
                torch.save(state, "{}_{}_{}_{}_model.pkl".format(
                    'rsnet', args.dataset,str(epoch),str(i)))
                state = {'epoch': epoch+1,
                         'model_state': drnet.state_dict(),
                         'optimizer_state': optimizer.state_dict(),
                         'error': out.item(),}
                torch.save(state, "{}_{}_{}_{}_model.pkl".format(
                    'drnet', args.dataset,str(epoch),str(i)))
                print('save success')

        #average_loss=average_loss/816       
        check=1      
        if epoch%check==0:    
            print('testing!')
            rsnet.test()
            drnet.test()
            rmse=[]
            silog=[]
            log_rmse=[]
            for i_val, (images, labels) in tqdm(enumerate(valloader)):
                #print(r'\n')
                images = images.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                print(i_val)
                with torch.no_grad():
                    region_support = rsnet(images)
                    coarse_depth=torch.cat([images,region_support],1)
                    coarse_depth=torch.cat([coarse_depth,region_support],1)
                    outputs=drnet(coarse_depth)
                    #pred = outputs[2].data.cpu().numpy()
                    #gt = labels.data.cpu().numpy()
                    ones=np.ones((gt.shape))
                    zeros=np.zeros((gt.shape))
                    gt=labels
                    pred=outputs[2]
                    num=torch.sum(torch.where(gt>0,ones,zeros))
                    pred=torch.reshape(pred,gt.shape)
                    rmse.append(torch.sum(torch.where(gt>0,torch.pow(gt-pred,2)))/num)
                    gt=torch.where(gt>0,torch.log(gt+1e-6),zeros)
                    pred=torch.where(gt>0,torch.log(pred+1e-6),zeros)
                    silog.append(torch.sum(torch.where(gt>0,torch.pow(gt-pred,2)))/num-torch.pow(torch.sum(torch.where(gt>0,gt-pred)),2)/num/num)
                    log_rmse.append(torch.sum(torch.where(gt>0,torch.pow(gt-pred,2)))/num)
                    print("rmse=%.4f,silog=%.4f,log_rmse=%.4f"%(rmse[i_val],silog[i_val],log_rmse[i_val]))

            rmse=np.mean(rmse)
            silog=np.mean(silog)
            log_rmse=np.mean(log_rmse)
            #error_rate=np.mean(error_rate)
            print("rmse=%.4f,silog=%.4f,log_rmse=%.4f"%(rmse,silog,log_rmse))

            # if error<= best_error:
            #     best_error = error
            #     state = {'epoch': epoch+1,
            #              'model_state': rsnet.state_dict(),
            #              'optimizer_state': optimizer.state_dict(),
            #              'error': error,}
            #     torch.save(state, "{}_{}_best_model.pkl".format(
            #         'rsnet', args.dataset))
            #     state = {'epoch': epoch+1,
            #              'model_state': drnet.state_dict(),
            #              'optimizer_state': optimizer.state_dict(),
            #              'error': error,}
            #     torch.save(state, "{}_{}_best_model.pkl".format(
            #         'drnet', args.dataset))                
            #     print('save success')
            # np.save('/home/lidong/Documents/RSDEN/RSDEN/kitti/loss.npy',loss_rec)
        if epoch%1==0:
            #best_error = error
            state = {'epoch': epoch+1,
                     'model_state': rsnet.state_dict(),
                     'optimizer_state': optimizer.state_dict(),
                     'error': error,}
            torch.save(state, "{}_{}_{}_model.pkl".format(
                'rsnet', args.dataset,str(epoch)))
            state = {'epoch': epoch+1,
                     'model_state': drnet.state_dict(),
                     'optimizer_state': optimizer.state_dict(),
                     'error': error,}
            torch.save(state, "{}_{}_{}_model.pkl".format(
                'drnet', args.dataset,str(epoch)))
            print('save success')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='rsdin',
                        help='Architecture to use [\'region support network\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='kitti',
                        help='Dataset to use [\'sceneflow and kitti etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=360,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=800,
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=40,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=2,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from /home/lidong/Documents/RSDEN/RSDEN/rsnet_nyu_30_model.pkl')
    parser.add_argument('--visdom', nargs='?', type=bool, default=True,
                        help='Show visualization(s) on visdom | False by  default')
    args = parser.parse_args()
    train(args)
