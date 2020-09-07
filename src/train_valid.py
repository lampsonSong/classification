import torch 
import os
import shutil
from tqdm import tqdm
from src.utils import AverageMeter, accuracy, Calculate


def train(train_loader, model, criterion, optimizer, epoch, writer=None, scheduler=None):
    model.train()
    num_iter_per_epoch = len(train_loader)

    pbar = tqdm(train_loader)
    for i, (images, target) in enumerate(pbar):

        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)
        
        if writer:
            writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + i)

        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            if writer:
                writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch * num_iter_per_epoch + i)
            scheduler.step()
        else:
            tmp_lr = 0
            for param_group in optimizer.param_groups:
                tmp_lr = param_group['lr']
                break
            if writer:
                writer.add_scalar('LR', tmp_lr, epoch * num_iter_per_epoch + i)
        pbar.set_description("Train batch loss: %f" % loss)

def validate_with_sigmoid(val_loader, model, criterion, epoch, category, writer=None, threshold=0.5):

    # switch to evaluate mode
    model.eval()
    valid = Calculate('valid', categories=category, thresh=threshold)

    losses = 0.
    # num_tp = 0
    # num_fp = 0

    # batch = 0
    pbar = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(pbar):
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()
            # if batch == 0:
            #     batch = images.shape[0]
            # compute output
            output = model(images)
            loss = criterion(output, target)

            losses += loss.item()

            threshed_output = torch.sigmoid(output) > threshold

            valid.update(threshed_output.cpu().numpy(), target.cpu().numpy())

            # # tp and fp
            # tp = threshed_output * target 
            # fp = threshed_output * ((target + 1) % 2)
            # # print (sum(sum(fp)), batch - sum(sum(tp)))
            # num_tp += sum(sum(tp))
            # num_fp += sum(sum(fp))

            pbar.set_description('Validation')
            


    avg_loss = losses / len(val_loader)
    if writer:
        writer.add_scalar('Test/Loss', avg_loss, epoch)
    
    # precision = num_tp / (num_tp+num_fp)
    # recall = num_tp / (len(val_loader) * batch)

    precision, recall, f1_score = valid.result()

    if writer:
        writer.add_scalar('Test/precision', precision, epoch)
        writer.add_scalar('Test/RECALL', recall, epoch)
    print ('Avg Loss {}, mAP {}, mAR {}'.format(avg_loss, precision, recall))

    return avg_loss, precision, recall, f1_score

def validate_with_softmax(val_loader, model, criterion, epoch, writer=None, threshold=0.5):

    # switch to evaluate mode
    model.eval()

    losses = AverageMeter('Loss', ":.4e")
    top1 = AverageMeter('Acc@1', ':6.2f')

    pbar = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(pbar):
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()
        
            # compute output
            output = model(images)
            loss = criterion(output, target)

            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0][0], images.size(0))

            pbar.set_description('Validation')
            
        print(" * Acc@1 {top1.avg:.3f}".format(top1=top1))
        if writer:
            writer.add_scalar('Test/Loss', losses.avg, epoch)
            writer.add_scalar('Test/Top1_acc', top1.avg, epoch)

    return top1.avg
