# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 13:41:34
# @Last Modified by:   yulidong
# @Last Modified time: 2018-07-28 16:31:32
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
                           split='train_region', img_size=(args.img_rows, args.img_cols),task='all')
    v_loader = data_loader(data_path, is_transform=True,
                           split='test_region', img_size=(args.img_rows, args.img_cols),task='all')

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader, batch_size=args.batch_size, num_workers=2, shuffle=True)
    valloader = data.DataLoader(
        v_loader, batch_size=args.batch_size, num_workers=2)

    # Setup Metrics
    running_metrics = runningScore(n_classes)

    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()
        # old_window = vis.line(X=torch.zeros((1,)).cpu(),
        #                        Y=torch.zeros((1)).cpu(),
        #                        opts=dict(xlabel='minibatches',
        #                                  ylabel='Loss',
        #                                  title='Trained Loss',
        #                                  legend=['Loss'])
        a_window = vis.line(X=torch.zeros((1,)).cpu(),
                               Y=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='minibatches',
                                         ylabel='Loss',
                                         title='Region Loss1',
                                         legend=['Region']))
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
            np.random.rand(480, 640),
            opts=dict(title='predict1!', caption='predict1.'),
        )
        pre_window2 = vis.image(
            np.random.rand(480, 640),
            opts=dict(title='predict2!', caption='predict2.'),
        )
        pre_window3 = vis.image(
            np.random.rand(480, 640),
            opts=dict(title='predict3!', caption='predict3.'),
        )

        ground_window = vis.image(
            np.random.rand(480, 640),
            opts=dict(title='ground!', caption='ground.')
            ),
        region_window = vis.image(
            np.random.rand(480, 640),
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
    # Check if model has custom optimizer / loss
    # modify to adam, modify the learning rate
    if hasattr(drnet.module, 'optimizer'):
        optimizer = drnet.module.optimizer
    else:
        # optimizer = torch.optim.Adam(
        #     rsnet.parameters(), lr=args.l_rate,weight_decay=5e-4,betas=(0.9,0.999))
        optimizer = torch.optim.SGD(
            rsnet.parameters(), lr=args.l_rate,momentum=0.99, weight_decay=5e-4)
    if hasattr(rsnet.module, 'loss'):
        print('Using custom loss')
        loss_fn = rsnet.module.loss
    else:
        loss_fn = log_r
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
            loss_rec=np.load('/home/lidong/Documents/RSDEN/RSDEN/loss.npy')
            loss_rec=list(loss_rec)
            loss_rec=loss_rec[:1632*trained]
            # for i in range(300):
            #     loss_rec[i][1]=loss_rec[i+300][1]
            for l in range(int(len(loss_rec)/1632)):
                if args.visdom:
                    
                    vis.line(
                        X=torch.ones(1).cpu() * loss_rec[l*1632][0],
                        Y=np.mean(np.array(loss_rec[l*1632:(l+1)*1632])[:,1])*torch.ones(1).cpu(),
                        win=old_window,
                        update='append')
            
    else:

        print("No checkpoint found at '{}'".format(args.resume))
        print('Initialize seperately!')
        checkpoint=torch.load('/home/lidong/Documents/RSDEN/RSDEN/rsnet_nyu_135_model.pkl')
        rsnet.load_state_dict(checkpoint['model_state'])
        trained=checkpoint['epoch']
        best_error=checkpoint['error']
        print(best_error)
        print('load success from rsnet %.d'%trained)
        checkpoint=torch.load('/home/lidong/Documents/RSDEN/RSDEN/drnet_nyu_135_model.pkl')

        # model_dict=drnet.state_dict()            
        # pre_dict={k: v for k, v in checkpoint['model_state'].items() if k in model_dict}

        # model_dict.update(pre_dict)
        # #print(model_dict['module.conv1.weight'].shape)
        # model_dict['module.conv1.weight']=torch.cat([model_dict['module.conv1.weight'],torch.reshape(model_dict['module.conv1.weight'][:,3,:,:],[64,1,7,7])],1)
        # #print(model_dict['module.conv1.weight'].shape)
        # drnet.load_state_dict(model_dict)
        drnet.load_state_dict(checkpoint['model_state'])
        #optimizer.load_state_dict(checkpoint['optimizer_state'])
        trained=checkpoint['epoch']
        print('load success from drnet %.d'%trained)
        #trained=0
        loss_rec=[]
        #loss_rec=np.load('/home/lidong/Documents/RSDEN/RSDEN/loss.npy')
        #loss_rec=list(loss_rec)
        #loss_rec=loss_rec[:1632*trained]
        #average_loss=checkpoint['error']
        # opti_dict=optimizer.state_dict()
        # #pre_dict={k: v for k, v in checkpoint['optimizer_state'].items() if k in opti_dict}
        # pre_dict=checkpoint['optimizer_state']
        # # for k,v in pre_dict.items():
        # #     print(k)
        # #     if k=='state':
        # #         #print(v.type)
        # #         for a,b in v.items():
        # #             print(a)
        # #             print(b['momentum_buffer'].shape)
        # #return 0
        # opti_dict.update(pre_dict)
        # # for k,v in opti_dict.items():
        # #     print(k)
        # #     if k=='state':
        # #         #print(v.type)
        # #         for a,b in v.items():
        # #             if a==140011149405280:
        # #                 print(b['momentum_buffer'].shape)
        # #print(opti_dict['state'][140011149405280]['momentum_buffer'].shape)
        # opti_dict['state'][140011149405280]['momentum_buffer']=torch.cat([opti_dict['state'][140011149405280]['momentum_buffer'],torch.reshape(opti_dict['state'][140011149405280]['momentum_buffer'][:,3,:,:],[64,1,7,7])],1)
        # #print(opti_dict['module.conv1.weight'].shape)
        # optimizer.load_state_dict(opti_dict)
    




    # it should be range(checkpoint[''epoch],args.n_epoch)
    for epoch in range(trained, args.n_epoch):
    #for epoch in range(0, args.n_epoch):

        #trained
        print('training!')
        rsnet.train()
        drnet.train()
        for i, (images, labels,segments) in enumerate(trainloader):
            images = images.cuda()
            labels = labels.cuda(cuda2)
            segments=segments.cuda(cuda2)
            #for error_sample in range(10):
            optimizer.zero_grad()
            #with torch.autograd.enable_grad():
            region_support = rsnet(images)
                #with torch.autograd.enable_grad():
            coarse_depth=torch.cat([images,region_support],1)
            coarse_depth=torch.cat([coarse_depth,region_support],1)
            #with torch.no_grad():
            outputs=drnet(coarse_depth)
            #outputs.append(region_support)
            #outputs=torch.reshape(outputs,[outputs.shape[0],1,outputs.shape[1],outputs.shape[2]])
            #outputs=outputs
            loss = loss_fn(input=outputs, target=labels)
            out=0.2*loss[0]+0.3*loss[1]+0.5*loss[2]
            #out=out
            a=l1(input=region_support,target=labels.to(cuda0))
            #a=region_log(input=region_support,target=labels.to(cuda0),instance=segments.to(cuda0)).to(cuda2)
            b=log_loss(region_support,labels.to(cuda0)).item()
            #out=0.8*out+0.02*a
            #a.backward()
            # print('training:'+str(i)+':learning_rate'+str(loss.data.cpu().numpy()))
            out.backward()
            optimizer.step()
            # print('out:%.4f,error_sample:%d'%(out.item(),error_sample))
            # if i==0:
            #     average_loss=(average_loss+out.item())/2
            #     break
            # if out.item()<average_loss/i:
            #     break
            # print(torch.Tensor([loss.data[0]]).unsqueeze(0).cpu())
            #print(loss.item()*torch.ones(1).cpu())
            #nyu2_train:246,nyu2_all:1632
            if args.visdom:
                vis.line(
                    X=torch.ones(1).cpu() * i+torch.ones(1).cpu() *(epoch-trained)*1632,
                    Y=a.item()*torch.ones(1).cpu(),
                    win=a_window,
                    update='append')
                vis.line(
                    X=torch.ones(1).cpu() * i+torch.ones(1).cpu() *(epoch-trained)*1632,
                    Y=loss[0].item()*torch.ones(1).cpu(),
                    win=loss_window1,
                    update='append')
                vis.line(
                    X=torch.ones(1).cpu() * i+torch.ones(1).cpu() *(epoch-trained)*1632,
                    Y=loss[1].item()*torch.ones(1).cpu(),
                    win=loss_window2,
                    update='append')
                vis.line(
                    X=torch.ones(1).cpu() * i+torch.ones(1).cpu() *(epoch-trained)*1632,
                    Y=loss[2].item()*torch.ones(1).cpu(),
                    win=loss_window3,
                    update='append')
                pre = outputs[0].data.cpu().numpy().astype('float32')
                pre = pre[0,:, :]
                #pre = np.argmax(pre, 0)
                pre = (np.reshape(pre, [480, 640]).astype('float32')-np.min(pre))/(np.max(pre)-np.min(pre))
                #pre = pre/np.max(pre)
                # print(type(pre[0,0]))
                vis.image(
                    pre,
                    opts=dict(title='predict1!', caption='predict1.'),
                    win=pre_window1,
                )
                pre = outputs[1].data.cpu().numpy().astype('float32')
                pre = pre[0,:, :]
                #pre = np.argmax(pre, 0)
                pre = (np.reshape(pre, [480, 640]).astype('float32')-np.min(pre))/(np.max(pre)-np.min(pre))
                #pre = pre/np.max(pre)
                # print(type(pre[0,0]))
                vis.image(
                    pre,
                    opts=dict(title='predict2!', caption='predict2.'),
                    win=pre_window2,
                )
                pre = outputs[2].data.cpu().numpy().astype('float32')
                pre = pre[0,:, :]
                #pre = np.argmax(pre, 0)
                pre = (np.reshape(pre, [480, 640]).astype('float32')-np.min(pre))/(np.max(pre)-np.min(pre))
                #pre = pre/np.max(pre)
                # print(type(pre[0,0]))
                vis.image(
                    pre,
                    opts=dict(title='predict3!', caption='predict3.'),
                    win=pre_window3,
                )
                ground=labels.data.cpu().numpy().astype('float32')
                #print(ground.shape)
                ground = ground[0, :, :]
                ground = (np.reshape(ground, [480, 640]).astype('float32')-np.min(ground))/(np.max(ground)-np.min(ground))
                vis.image(
                    ground,
                    opts=dict(title='ground!', caption='ground.'),
                    win=ground_window,
                )
                region_vis=region_support.data.cpu().numpy().astype('float32')
                #print(ground.shape)
                region_vis = region_vis[0, :, :]
                region_vis = (np.reshape(region_vis, [480, 640]).astype('float32')-np.min(region_vis))/(np.max(region_vis)-np.min(region_vis))
                vis.image(
                    region_vis,
                    opts=dict(title='region_vis!', caption='region_vis.'),
                    win=region_window,
                )
            #average_loss+=out.item()
            loss_rec.append([i+epoch*1632,torch.Tensor([loss[0].item()]).unsqueeze(0).cpu(),torch.Tensor([loss[1].item()]).unsqueeze(0).cpu(),torch.Tensor([loss[2].item()]).unsqueeze(0).cpu()])
            print("data [%d/1632/%d/%d]region:%.4f,%.4f Loss1: %.4f Loss2: %.4f Loss3: %.4f out:%.4f " % (i, epoch, args.n_epoch,a.item(),b,loss[0].item(),loss[1].item(),loss[2].item(),out.item()))

        #average_loss=average_loss/816       
        if epoch>50:
            check=1
        else:
            check=1
        if epoch>70:
            check=1                 
        if epoch%check==0:    
            print('testing!')
            rsnet.train()
            drnet.train()
            error_lin=[]
            error_log=[]
            error_va=[]
            error_rate=[]
            error_absrd=[]
            error_squrd=[]
            thre1=[]
            thre2=[]
            thre3=[]

            for i_val, (images, labels,segments) in tqdm(enumerate(valloader)):
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
                    pred = outputs[2].data.cpu().numpy()
                    gt = labels.data.cpu().numpy()
                    ones=np.ones((gt.shape))
                    zeros=np.zeros((gt.shape))
                    pred=np.reshape(pred,(gt.shape))
                    #gt=np.reshape(gt,[4,480,640])
                    dis=np.square(gt-pred)
                    error_lin.append(np.sqrt(np.mean(dis)))
                    dis=np.square(np.log(gt)-np.log(pred))
                    error_log.append(np.sqrt(np.mean(dis)))
                    alpha=np.mean(np.log(gt)-np.log(pred))
                    dis=np.square(np.log(pred)-np.log(gt)+alpha)
                    error_va.append(np.mean(dis)/2)
                    dis=np.mean(np.abs(gt-pred))/gt
                    error_absrd.append(np.mean(dis))
                    dis=np.square(gt-pred)/gt
                    error_squrd.append(np.mean(dis))
                    thelt=np.where(pred/gt>gt/pred,pred/gt,gt/pred)
                    thres1=1.25
                    
                    thre1.append(np.mean(np.where(thelt<thres1,ones,zeros)))
                    thre2.append(np.mean(np.where(thelt<thres1*thres1,ones,zeros)))
                    thre3.append(np.mean(np.where(thelt<thres1*thres1*thres1,ones,zeros)))
                    #a=thre1[i_val]
                    #error_rate.append(np.mean(np.where(dis<0.6,ones,zeros)))
                    print("error_lin=%.4f,error_log=%.4f,error_va=%.4f,error_absrd=%.4f,error_squrd=%.4f,thre1=%.4f,thre2=%.4f,thre3=%.4f"%(
                        error_lin[i_val],
                        error_log[i_val],
                        error_va[i_val],
                        error_absrd[i_val],
                        error_squrd[i_val],
                        thre1[i_val],
                        thre2[i_val],
                        thre3[i_val]))
                    # if i_val > 219/check:
                    #     break
            error=np.mean(error_lin)
            #error_rate=np.mean(error_rate)
            print("error=%.4f"%(error))

            if error<= best_error:
                best_error = error
                state = {'epoch': epoch+1,
                         'model_state': rsnet.state_dict(),
                         'optimizer_state': optimizer.state_dict(),
                         'error': error,}
                torch.save(state, "{}_{}_best_model.pkl".format(
                    'rsnet', args.dataset))
                state = {'epoch': epoch+1,
                         'model_state': drnet.state_dict(),
                         'optimizer_state': optimizer.state_dict(),
                         'error': error,}
                torch.save(state, "{}_{}_best_model.pkl".format(
                    'drnet', args.dataset))                
                print('save success')
            np.save('/home/lidong/Documents/RSDEN/RSDEN//loss.npy',loss_rec)
        if epoch%3==0:
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
    parser.add_argument('--dataset', nargs='?', type=str, default='nyu',
                        help='Dataset to use [\'sceneflow and kitti etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=480,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=640,
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=4000,
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
