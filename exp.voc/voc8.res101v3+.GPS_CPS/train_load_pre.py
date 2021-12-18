from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
from dataloader import get_train_loader
from network import SingleNetwork
from dataloader import VOC
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP # 추가
import torch.distributed as dist
from utils.pyt_utils import load_model

try:
    from azureml.core import Run
    azure = True
    run = Run.get_context()
except:
    azure = False

parser = argparse.ArgumentParser()

os.environ['MASTER_PORT'] = '169711'

if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False

torch.cuda.empty_cache()

with Engine(custom_parser=parser) as engine:

    args = parser.parse_args()

    cudnn.benchmark = True

    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader + unsupervised data loader
    train_loader, train_sampler = get_train_loader(engine, VOC, train_source=config.train_source, \
                                                   unsupervised=False) # supervised
    unsupervised_train_loader, unsupervised_train_sampler = get_train_loader(engine, VOC, \
                train_source=config.unsup_source, unsupervised=True) # unsupervised

    if engine.distributed and (engine.local_rank == 0):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        logger = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    criterion_csst = nn.MSELoss(reduction='mean')

    BatchNorm2d = nn.BatchNorm2d

    # define and init models
    teacher = SingleNetwork(config.num_classes, criterion=criterion, norm_layer=BatchNorm2d, pretrained_model=config.pretrained_model)
    branch1 = SingleNetwork(config.num_classes, criterion=criterion, norm_layer=BatchNorm2d, pretrained_model=config.pretrained_model)
    branch2 = SingleNetwork(config.num_classes, criterion=criterion, norm_layer=BatchNorm2d, pretrained_model=config.pretrained_model)

    init_weight(branch1.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    init_weight(branch2.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # define the base lr & pretrain_teacher lr
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr * engine.world_size

    params_list_l = []
    params_list_l = group_weight(params_list_l, branch1.backbone,
                               BatchNorm2d, base_lr)
    for module in branch1.business_layer:
        params_list_l = group_weight(params_list_l, module, BatchNorm2d,
                                   base_lr)        # head lr * 10

    optimizer_l = torch.optim.SGD(params_list_l,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    params_list_r = []
    params_list_r = group_weight(params_list_r, branch2.backbone,
                               BatchNorm2d, base_lr)
    for module in branch2.business_layer:
        params_list_r = group_weight(params_list_r, module, BatchNorm2d,
                                   base_lr)        # head lr * 10

    optimizer_r = torch.optim.SGD(params_list_r,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    params_list_t = []
    params_list_t = group_weight(params_list_t, teacher.backbone,
                               BatchNorm2d, base_lr)
    for module in teacher.business_layer:
        params_list_t = group_weight(params_list_t, module, BatchNorm2d,
                                   base_lr)   
    optimizer_t = torch.optim.SGD(params_list_t,
                            lr=base_lr,
                            momentum=config.momentum,
                            weight_decay=config.weight_decay)
    

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)


    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            rank = dist.get_rank()
            teacher.cuda()
            teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
            teacher = DDP(teacher.to(rank), find_unused_parameters=False, device_ids=[rank], output_device=[rank])

            branch1.cuda()
            branch1 = nn.SyncBatchNorm.convert_sync_batchnorm(branch1)
            branch1 = DDP(branch1.to(rank), find_unused_parameters=False, device_ids=[rank], output_device=[rank])

            branch2.cuda()
            branch2 = nn.SyncBatchNorm.convert_sync_batchnorm(branch2)
            branch2 = DDP(branch2.to(rank), find_unused_parameters=False, device_ids=[rank], output_device=[rank])

 
    checkpoint = torch.load('pretrained_t.pt')
    teacher.load_state_dict(checkpoint['model_state_dict'])
    optimizer_t.load_state_dict(checkpoint['optimizer_state_dict']) 
    print("Successfully loaded pretrained teacher")

    start_time = time.time()
    teacher.train()
  
    print('begin training teacher & students')
    branch1.train()
    branch2.train()
    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        if is_debug:
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)


        dataloader = iter(train_loader) # supervised
        unsupervised_dataloader = iter(unsupervised_train_loader) # unsupervised

        sum_loss_sup_l = 0
        sum_loss_sup_r = 0
        sum_cps = 0
        sum_gps = 0
        sum_gps_stu = 0
        sum_loss_sup_t2 = 0

        for idx in pbar:
            optimizer_l.zero_grad()
            optimizer_r.zero_grad()
            optimizer_t.zero_grad()
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next() # supervised
            unsup_minibatch = unsupervised_dataloader.next() # unsupervised
            imgs = minibatch['data']
            gts = minibatch['label']
            unsup_imgs = unsup_minibatch['data']
            imgs = imgs.cuda(non_blocking=True)
            unsup_imgs = unsup_imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)


            b, c, h, w = imgs.shape
            _, pred_sup_l = branch1(imgs)
            _, pred_unsup_l = branch1(unsup_imgs) 
            _, pred_sup_r = branch2(imgs)
            _, pred_unsup_r = branch2(unsup_imgs)
            _, pred_unsup_t = teacher(unsup_imgs)
            _, pred_sup_t = teacher(imgs) # teacher sup


            '''teacher teaches students'''
            ### cps loss ###
            pred_l = torch.cat([pred_unsup_l], dim=0) # only unsupervised data
            pred_r = torch.cat([pred_unsup_r], dim=0) # only unsupervised data
            _, max_l = torch.max(pred_l, dim=1)
            _, max_r = torch.max(pred_r, dim=1)
            max_l = max_l.long()
            max_r = max_r.long()
            cps_loss = criterion(pred_l, max_r) + criterion(pred_r, max_l)
            if engine.distributed:
                dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
                cps_loss = cps_loss / engine.world_size
            cps_loss = cps_loss * config.cps_weight # 1.5

            ### gps loss ###
            pred_t = torch.cat([pred_unsup_t], dim=0) # only unsupervised data
            _, max_t = torch.max(pred_t, dim=1)
            max_t = max_t.long()
            gps_loss = criterion(pred_l, max_t) + criterion(pred_r, max_t)
            if engine.distributed:
                dist.all_reduce(gps_loss, dist.ReduceOp.SUM)
                gps_loss = gps_loss / engine.world_size
            gps_loss = gps_loss * config.gps_weight # 1.5

            ### standard cross entropy loss ###
            loss_sup_l = criterion(pred_sup_l, gts)
            if engine.distributed:
                dist.all_reduce(loss_sup_l, dist.ReduceOp.SUM)
                loss_sup_l = loss_sup_l / engine.world_size

            loss_sup_r = criterion(pred_sup_r, gts)
            if engine.distributed:
                dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
                loss_sup_r = loss_sup_r / engine.world_size

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            # reset the learning rate
            optimizer_l.param_groups[0]['lr'] = lr
            optimizer_l.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_l.param_groups)):
                optimizer_l.param_groups[i]['lr'] = lr
            optimizer_r.param_groups[0]['lr'] = lr
            optimizer_r.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_r.param_groups)):
                optimizer_r.param_groups[i]['lr'] = lr

            loss = loss_sup_l + loss_sup_r + cps_loss + gps_loss
            loss.backward() 
            optimizer_l.step()
            optimizer_r.step()

            sum_loss_sup_l += loss_sup_l.item()
            sum_loss_sup_r += loss_sup_r.item()
            sum_cps += cps_loss.item()
            sum_gps += gps_loss.item()


            '''students teach teacher'''
            ### gps loss from students ###
            _, max_stu = torch.max(pred_l + pred_r, dim=1) # Y_students
            max_stu = max_stu.long()
            gps_stu_loss = criterion(pred_unsup_t, max_stu)
            if engine.distributed:
                dist.all_reduce(gps_stu_loss, dist.ReduceOp.SUM)
                gps_stu_loss = gps_stu_loss / engine.world_size
            gps_stu_loss = gps_stu_loss * config.gps_stu_weight 

            ### standard cross entropy loss ###
            loss_sup_t2 = criterion(pred_sup_t, gts)
            if engine.distributed:
                dist.all_reduce(loss_sup_t2, dist.ReduceOp.SUM)
                loss_sup_t2 = loss_sup_t2 / engine.world_size
 
            # reset the learning rate
            optimizer_t.param_groups[0]['lr'] = config.fixed_lr
            optimizer_t.param_groups[1]['lr'] = config.fixed_lr
            for i in range(2, len(optimizer_t.param_groups)):
                optimizer_t.param_groups[i]['lr'] = config.fixed_lr
            loss = gps_stu_loss + loss_sup_t2
            loss.backward()
            optimizer_t.step()

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup_l=%.2f' % loss_sup_l.item() \
                        + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                        + ' loss_cps=%.4f' % cps_loss.item() \
                        + ' loss_gps=%.4f' % gps_loss.item() \
                        + ' loss_gps_stu=%.4f' % gps_stu_loss.item() \
                        + ' loss_sup_t2=%.4f' % loss_sup_t2.item()

            sum_gps_stu += gps_stu_loss.item()
            sum_loss_sup_t2 += loss_sup_t2.item() ## teacher sup
            pbar.set_description(print_str, refresh=False)


        if engine.distributed and (engine.local_rank == 0):
            logger.add_scalar('train_loss_sup_l', sum_loss_sup_l / len(pbar), epoch)
            logger.add_scalar('train_loss_sup_r', sum_loss_sup_r / len(pbar), epoch)
            logger.add_scalar('train_loss_cps', sum_cps / len(pbar), epoch)
            logger.add_scalar('train_loss_gps', sum_gps / len(pbar), epoch)
            logger.add_scalar('train_loss_gps_stu', sum_gps_stu / len(pbar), epoch)
            logger.add_scalar('train_loss_sup_t2', sum_loss_sup_t2 / len(pbar), epoch)

        if azure and engine.local_rank == 0:
            run.log(name='Supervised Training Loss Left', value=sum_loss_sup_l / len(pbar))
            run.log(name='Supervised Training Loss right', value=sum_loss_sup_r / len(pbar))
            run.log(name='Unsupervised Training Loss CPS', value=sum_cps / len(pbar))
            run.log(name='Unsupervised Training Loss GPS', value=sum_gps / len(pbar))
            run.log(name='Unsupervised Training Loss GPS_stu', value=sum_gps_stu / len(pbar))
            run.log(name='Supervised Training Loss Tchr', value=sum_loss_sup_t2 / len(pbar))

        
        engine.register_state(model=teacher, optimizer_t=optimizer_t)
        if (epoch > config.nepochs // 2) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
        # if (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
        engine.register_state(model=branch1, optimizer_l=optimizer_l)
        if (epoch > config.nepochs // 2) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
        # if (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.b1_snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
        engine.register_state(model=branch2, optimizer_r=optimizer_r)
        if (epoch > config.nepochs // 2) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
        # if (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.b2_snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)

    torch.save({
        'model_state_dict': teacher.state_dict(),
        'optimizer_state_dict': optimizer_t.state_dict()
        }, config.save_t_dir)

    torch.save({
        'model_state_dict': branch1.state_dict(),
        'optimizer_state_dict': optimizer_l.state_dict()
        }, config.save_b1_dir)

    torch.save({
        'model_state_dict': branch2.state_dict(),
        'optimizer_state_dict': optimizer_r.state_dict()
        }, config.save_b2_dir)                                            

    end_time = time.time()
    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total Training Time {:0>2}h {:0>2}m {:05.2f}s".format(int(hours),int(minutes),seconds))