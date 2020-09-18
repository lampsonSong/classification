"""
@author: Signatrix GmbH
Implementation of paradigm described in paper: Designing Network Design Spaces published by Facebook AI Research (FAIR)
"""
import argparse
import os
import shutil
import time
from pthflops import count_ops
from torchsummary import summary
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter

from src.networks.regnets.regnet import RegNetY
from src.networks.mobilenet.mobilenetv3 import MobileNetV3Tiny, MobileNetV3
# from src.datasets.face_mask import FaceMaskDataset as UsedDataset
# from src.datasets.car_classify import CarDataset as UsedDataset
from src.datasets.classification import ClassifyDataset as UsedDataset

from src.networks.regnets.config import TRAIN_IMAGE_SIZE

from src.train_valid import train, validate_with_sigmoid, validate_with_softmax
# from src.losses.anchor_loss import AnchorLoss

import torch.multiprocessing as mp
import apex
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

hyp = {
        'rotate_degrees' : [0, 360],
        'hsv_h' : 0.0138,
        'hsv_s' : 0.678,
        'hsv_v' : 0.36,
        'category_names' : ['face', 'face_mask', 'face_mask_half']
    }

def get_args():
    parser = argparse.ArgumentParser(
        description="Implementation of paradigm described in paper: Designing Network Design Spaces published by Facebook AI Research (FAIR)")

    parser.add_argument("-d", "--data_path", type=str, default="/home/hsw/VehicleClassification_train", help="the root folder of dataset")
    parser.add_argument("-e", "--epochs", default=100, type=int, help="number of total epochs to run")
    parser.add_argument("-b", "--batch_size", default=1024, type=int)
    parser.add_argument("-l", "--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("-m", "--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("-w", "--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--log_path", type=str, default="tensorboard/test")
    parser.add_argument("--saved_path", type=str, default="models/test")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--nodes", default=1, type=int)
    parser.add_argument("--node_rank", default=0, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--mixed_precision", action='store_true')
    parser.add_argument("--loss", type=str, default='sigmoid', help='you can choose either sigmoid or softmax')
    parser.add_argument("--optimizer", type=str, default='sgd')
    parser.add_argument('--path', type=str, default=None, required=True)

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    args.log_path = 'car/tensorboard/' + args.path
    args.saved_path = 'car/models/' + args.path

    return args

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warm_up(optimizer, epoch, lr):
    lr = epoch / 10 * lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_dataloader(opt, local_rank):
    #training_set = UsedDataset(root_dir=opt.data_path, category_names=['Bus', 'Car', 'Truck'], in_size=opt.size, hyp=hyp, mode="train", loss_type=opt.loss)
    training_set = UsedDataset(root_dir=opt.data_path, category_names=hyp['category_names'], in_size=opt.size, hyp=hyp, mode="train", loss_type=opt.loss)
    training_sampler = DistributedSampler(
            training_set,
            num_replicas = opt.world_size,
            rank = local_rank
            )
    if opt.gpus > 1:
        training_generator = DataLoader(
                dataset = training_set,
                batch_size = opt.batch_size,
                num_workers = opt.num_workers,
                shuffle = False,
                pin_memory = True,
                drop_last = True,
                sampler = training_sampler
                )
    else:
        training_generator = DataLoader(
                dataset = training_set,
                batch_size = opt.batch_size,
                num_workers = opt.num_workers,
                shuffle = True,
                pin_memory = True,
                drop_last = True,
                )

    test_generator = None
    if local_rank % opt.gpus == 0:
        test_params = {"batch_size": opt.batch_size//10,
                       "shuffle": False,
                       "drop_last": False,
                       "num_workers": opt.num_workers}

        #test_set = UsedDataset(root_dir=opt.data_path, category_names=['Bus', 'Car', 'Truck'], in_size=opt.size, hyp=hyp, mode="val", loss_type=opt.loss)
        test_set = UsedDataset(root_dir=opt.data_path, category_names=hyp['category_names'], in_size=opt.size, hyp=hyp, mode="val", loss_type=opt.loss)
        test_generator = DataLoader(test_set, **test_params)

    return training_generator, test_generator


def main(gpu, opt):
    print ('Opt: ', opt)
    local_rank = opt.node_rank * opt.gpus + gpu

    torch.manual_seed(123)

    training_generator, test_generator = get_dataloader(opt, local_rank)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if os.path.isdir(opt.saved_path):
        shutil.rmtree(opt.saved_path)
    os.makedirs(opt.saved_path)


    writer = SummaryWriter(opt.log_path)

    num_categories = len(hyp['category_names'])
    
    model = MobileNetV3Tiny(n_class=num_categories, version=2)
    #model = MobileNetV3(n_class=3)

    dummy_input = torch.randn((1, 3, opt.size, opt.size))
    writer.add_graph(model, dummy_input)
    
    # Calculate model FLOPS and number of parameters
    # count_ops(model, dummy_input, verbose=False)
    # summary(model, (3, opt.size, opt.size), device="cpu")

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

       
    # criterion = nn.CrossEntropyLoss()
    # weight = torch.tensor([1., 1.5]).cuda() # no_mask mask
    weight = None
    if opt.loss == 'softmax':
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.BCEWithLogitsLoss(weight=weight)

    if opt.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True)
    else:
        optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    if opt.mixed_precision:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    if opt.world_size > 1:
        dist.init_process_group(
                backend = 'nccl',
                init_method = 'env://',
                world_size = opt.world_size,
                rank = local_rank
                )
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[gpu],
                find_unused_parameters=True
                )

    best_acc1 = 0
    best_recall = 0
    best_f1 = 0

    # set warmup && cosine learning rate decay
    total_epochs = opt.epochs 
    epoch_steps = len(training_generator)
    warm_epochs = 10
    
    warm_steps = warm_epochs * epoch_steps
    total_steps = total_epochs * epoch_steps
    lr_lambda = lambda epoch : (epoch / warm_steps + 0.0001) if epoch < warm_steps else 0.5 * (math.cos((epoch - warm_steps)/( total_steps - warm_steps) * math.pi) + 1)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # scheduler = None


    for epoch in range(opt.epochs):
        print ('Start running epoch.......{} / {} '.format(epoch, opt.epochs))

        train(training_generator, model, criterion, optimizer, epoch, writer, scheduler=scheduler)
        
        # validate
        if(local_rank % opt.gpus == 0):
            if opt.loss == 'sigmoid':
                val_loss, precision, recall, f1_score = validate_with_sigmoid(test_generator, model, criterion, epoch, writer, category=num_categories, threshold=0.8)
                
                if math.isnan(precision) or math.isnan(recall):
                    print ('Meet Nan. continue')
                    continue

                is_best = precision > best_acc1
                best_acc1 = max(precision, best_acc1)

                is_best_recall = recall > best_recall
                best_recall = max(recall, best_recall)

                is_best_all = f1_score > best_f1
                best_f1 = max(f1_score, best_f1)
                save_checkpoint({
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict() if opt.gpus < 2 else model.module.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                }, is_best, is_best_recall, is_best_all, opt.saved_path)
            else:
                top1_accu = validate_with_softmax(test_generator, model, criterion, epoch, writer, threshold=0.8)
                is_best = top1_accu > best_acc1        

                save_checkpoint({
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict() if opt.gpus < 2 else model.module.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                }, is_best, False, False, opt.saved_path)

            # adjust_learning_rate(optimizer, epoch, opt.lr)

def save_checkpoint(state, is_best_accu, is_best_recall, is_best_f1, saved_path, filename="checkpoint.pth"):
    file_path = os.path.join(saved_path, filename)
    torch.save(state['state_dict'], file_path)

    if is_best_accu:
        shutil.copyfile(file_path, os.path.join(saved_path, "precision_best_checkpoint.pth"))
    if is_best_recall:
        shutil.copyfile(file_path, os.path.join(saved_path, "recall_best_checkpoint.pth"))
    if is_best_f1:
        shutil.copyfile(file_path, os.path.join(saved_path, "f1_best_checkpoint.pth"))

if __name__ == "__main__":
    opt = get_args()
    if opt.world_size > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        # os.environ['MASTER_PORT'] = '9999'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(main, nprocs=opt.gpus, args=(opt,))
    else:
        main(0, opt)
