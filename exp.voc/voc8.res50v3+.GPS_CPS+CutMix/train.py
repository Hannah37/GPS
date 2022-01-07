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
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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

'''
For CutMix.
'''
import mask_gen
from custom_collate import SegCollate
mask_generator = mask_gen.BoxMaskGenerator(prop_range=config.cutmix_mask_prop_range, n_boxes=config.cutmix_boxmask_n_boxes,
                                           random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                           prop_by_area=not config.cutmix_boxmask_by_size, within_bounds=not config.cutmix_boxmask_outside_bounds,
                                           invert=not config.cutmix_boxmask_no_invert)

add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
    mask_generator
)
collate_fn = SegCollate()
mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)


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
                                                   unsupervised=False, collate_fn=collate_fn) # supervised
    unsupervised_train_loader_0, unsupervised_train_sampler_0 = get_train_loader(engine, VOC, \
            train_source=config.unsup_source, unsupervised=True, collate_fn=mask_collate_fn) # unsupervised
    unsupervised_train_loader_1, unsupervised_train_sampler_1 = get_train_loader(engine, VOC, \
                train_source=config.unsup_source, unsupervised=True, collate_fn=collate_fn) # unsupervised


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

    init_weight(teacher.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    init_weight(branch1.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    init_weight(branch2.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # define the learning rate
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr * engine.world_size

    # define the 3 optimizers
    params_list_t = []
    params_list_t = group_weight(params_list_t, teacher.backbone,
                               BatchNorm2d, base_lr)
    for module in teacher.business_layer:
        params_list_t = group_weight(params_list_t, module, BatchNorm2d,
                                   base_lr)        # head lr * 10

    optimizer_t = torch.optim.SGD(params_list_t,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

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

    print("ready to train teacher")

    teacher.train()
    print('begin training teacher')
    start_time = time.time()
    
    for epoch in range(config.pretrain_nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        if is_debug:
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

        dataloader = iter(train_loader) # supervised
        sum_loss_sup_t = 0

        ''' supervised part '''
        for idx in pbar:
            optimizer_t.zero_grad()
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next() # supervised
            imgs = minibatch['data']
            gts = minibatch['label']
            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)

            b, c, h, w = imgs.shape
            _, pred_sup_t = teacher(imgs)

            ### standard cross entropy loss ###
            loss_sup_t = criterion(pred_sup_t, gts)
            if engine.distributed:
                dist.all_reduce(loss_sup_t, dist.ReduceOp.SUM)
                loss_sup_t = loss_sup_t / engine.world_size

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            # reset the learning rate
            optimizer_t.param_groups[0]['lr'] = lr
            optimizer_t.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_t.param_groups)):
                optimizer_t.param_groups[i]['lr'] = lr
            loss = loss_sup_t
            loss.backward()
            optimizer_t.step()

            print_str = 'Epoch{}/{}'.format(epoch, config.pretrain_nepochs) \
                + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                + ' lr=%.2e' % lr \
                + ' loss_sup_t=%.2f' % loss_sup_t.item()

            sum_loss_sup_t+= loss_sup_t.item()
            pbar.set_description(print_str, refresh=False)

        if engine.distributed and (engine.local_rank == 0):
            logger.add_scalar('train_loss_sup_t', sum_loss_sup_t / len(pbar), epoch)

        if azure and engine.local_rank == 0:
            run.log(name='Supervised Training Loss of Teacher', value=sum_loss_sup_t / len(pbar))
        
        engine.register_state(model=teacher, optimizer=optimizer_t)
        if (epoch > config.nepochs // 2) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
        # if (epoch % config.snapshot_iter == 0) or (epoch == config.pretrain_nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.pretrain_snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)

    print('end training teacher')

    torch.save({
        'model_state_dict': teacher.state_dict(),
        'optimizer_state_dict': optimizer_t.state_dict()
        }, config.pretrain_dir)

    engine.reset_epoch_iter()

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
        unsupervised_dataloader_0 = iter(unsupervised_train_loader_0) # unsupervised
        unsupervised_dataloader_1 = iter(unsupervised_train_loader_1) # unsupervised

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
            unsup_minibatch_0 = unsupervised_dataloader_0.next() # unsupervised
            unsup_minibatch_1 = unsupervised_dataloader_1.next() # unsupervised

            imgs = minibatch['data']
            gts = minibatch['label']
            unsup_imgs_0 = unsup_minibatch_0['data']
            unsup_imgs_1 = unsup_minibatch_1['data']
            mask_params = unsup_minibatch_0['mask_params']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            unsup_imgs_0 = unsup_imgs_0.cuda(non_blocking=True)
            unsup_imgs_1 = unsup_imgs_1.cuda(non_blocking=True)
            mask_params = mask_params.cuda(non_blocking=True)

            ### unsupervised cps loss ###
            batch_mix_masks = mask_params
            unsup_imgs_mixed = unsup_imgs_0 * (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks
            with torch.no_grad():
                # Estimate the pseudo-label with branch#1 & supervise branch#2
                _, logits_u0_tea_1 = branch1(unsup_imgs_0)
                _, logits_u1_tea_1 = branch1(unsup_imgs_1)
                logits_u0_tea_1 = logits_u0_tea_1.detach()
                logits_u1_tea_1 = logits_u1_tea_1.detach()
                # Estimate the pseudo-label with branch#2 & supervise branch#1
                _, logits_u0_tea_2 = branch2(unsup_imgs_0)
                _, logits_u1_tea_2 = branch2(unsup_imgs_1)
                logits_u0_tea_2 = logits_u0_tea_2.detach()
                logits_u1_tea_2 = logits_u1_tea_2.detach()

            # Mix teacher predictions using the same mask
            # It makes no difference whether we do this with logits or probabilities as
            # the mask pixels are either 1 or 0
            logits_cons_tea_1 = logits_u0_tea_1 * (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
            _, ps_label_1 = torch.max(logits_cons_tea_1, dim=1)
            ps_label_1 = ps_label_1.long()
            logits_cons_tea_2 = logits_u0_tea_2 * (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
            _, ps_label_2 = torch.max(logits_cons_tea_2, dim=1)
            ps_label_2 = ps_label_2.long()

            # Get student#1 prediction for mixed image
            _, logits_cons_stu_1 = branch1(unsup_imgs_mixed)
            # Get student#2 prediction for mixed image
            _, logits_cons_stu_2 = branch2(unsup_imgs_mixed)

            cps_loss = criterion(logits_cons_stu_1, ps_label_2) + criterion(logits_cons_stu_2, ps_label_1)
            if engine.distributed:
                dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
                cps_loss = cps_loss / engine.world_size
            cps_loss = cps_loss * config.cps_weight 

            ### gps loss ###
            with torch.no_grad():
                _, logits_u0_tea_t = teacher(unsup_imgs_0)
                _, logits_u1_tea_t = teacher(unsup_imgs_1)
                logits_u0_tea_t = logits_u0_tea_t.detach()
                logits_u1_tea_t = logits_u1_tea_t.detach()

            logits_cons_tea_t = logits_u0_tea_t * (1 - batch_mix_masks) + logits_u1_tea_t * batch_mix_masks
            _, ps_label_t = torch.max(logits_cons_tea_t, dim=1) # pesudo supervision from teacher model
            ps_label_t = ps_label_t.long()

            gps_loss = criterion(logits_cons_stu_1, ps_label_t) + criterion(logits_cons_stu_2, ps_label_t)
            if engine.distributed:
                dist.all_reduce(gps_loss, dist.ReduceOp.SUM)
                gps_loss = gps_loss / engine.world_size
            gps_loss = gps_loss * config.gps_weight 

            ### standard cross entropy loss ###
            _, sup_pred_l = branch1(imgs)
            _, sup_pred_r = branch2(imgs)

            loss_sup_l = criterion(sup_pred_l, gts)
            if engine.distributed:
                dist.all_reduce(loss_sup_l, dist.ReduceOp.SUM)
                loss_sup_l = loss_sup_l / engine.world_size

            loss_sup_r = criterion(sup_pred_r, gts)
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
            # Get teacher prediction for mixed image
            _, logits_cons_stu_t = teacher(unsup_imgs_mixed)
            _, max_stu = torch.max(logits_cons_tea_1 + logits_cons_tea_2, dim=1) # Y_students
            max_stu = max_stu.long()
            gps_stu_loss = criterion(logits_cons_stu_t, max_stu)
            if engine.distributed:
                dist.all_reduce(gps_stu_loss, dist.ReduceOp.SUM)
                gps_stu_loss = gps_stu_loss / engine.world_size
            gps_stu_loss =  gps_stu_loss * config. gps_stu_weight 
            
            ### standard cross entropy loss ###
            _, sup_pred_t = teacher(imgs)
            loss_sup_t2 = criterion(sup_pred_t, gts)
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


        engine.register_state(model=teacher, optimizer=optimizer_t)
        if (epoch > config.nepochs // 2) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
        # if (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
        engine.register_state(model=branch1, optimizer=optimizer_l)
        if (epoch > config.nepochs // 2) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
        # if (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.b1_snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
        engine.register_state(model=branch2, optimizer=optimizer_r)
        if (epoch > config.nepochs // 2) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
        # if (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.b2_snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
    
    torch.save({
    'model_state_dict': teacher.state_dict(),
    'optimizer_state_dict': optimizer_t.state_dict()
    }, config.t_dir)

    torch.save({
        'model_state_dict': branch1.state_dict(),
        'optimizer_state_dict': optimizer_l.state_dict()
        }, config.b1_dir)

    torch.save({
        'model_state_dict': branch2.state_dict(),
        'optimizer_state_dict': optimizer_r.state_dict()
        }, config.b2_dir)

    end_time = time.time()
    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total Training Time {:0>2}h {:0>2}m {:05.2f}s".format(int(hours),int(minutes),seconds))
