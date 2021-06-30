import torch
import time
import os
import sys

import torch
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy

from advertorch.attacks import LinfPGDAttack, L2PGDAttack
import numpy as np

def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              device,
              logger,
              tb_writer=None,
              distributed=False,
              attack_type="clean",
              eps=4,
              step_size=1,
              attack_iter=5):
    
    print('validation at epoch {}'.format(epoch))
    
    model.eval()
    
    eps_rvs = None
    if isinstance(eps, (list, np.ndarray)): 
        eps_rvs = eps
        
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        
        if eps_rvs is not None:
            eps = eps_rvs[i]
        
        data_time.update(time.time() - end_time)
        inputs = inputs.to(device)
        targets = targets.to(device, non_blocking=True)
        
        if attack_type == "clean":
            outputs = model(inputs)
        elif attack_type == "pgd_inf":
            adversary = LinfPGDAttack(predict=model, loss_fn=criterion,
                                         eps=float(eps/255), nb_iter=attack_iter, eps_iter=float(step_size/255))
            adv_inputs = adversary.perturb(inputs, targets)
            
            outputs = model(adv_inputs)
            
        loss = criterion(outputs, targets)
            
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)
        acc_sum = torch.tensor([accuracies.sum],
                               dtype=torch.float32,
                               device=device)
        acc_count = torch.tensor([accuracies.count],
                                 dtype=torch.float32,
                                 device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()

    if logger is not None:
        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        tb_writer.add_scalar('val/acc', accuracies.avg, epoch)

    return losses.avg
