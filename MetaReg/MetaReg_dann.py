from meta_layers import MetaModule
from resnet_meta import resnet_meta50
from tools.lr_scheduler import StepwiseLR
from tools.transforms import ResizeImage
from tools.utils import AverageMeter, ProgressMeter, accuracy, ForeverDataIterator, MinHeap
import dalib.vision.models as models
import dalib.vision.datasets as datasets
from dalib.adaptation.dann import DomainAdversarialLoss, ImageClassifier
from dalib.modules.domain_discriminator import DomainDiscriminator
import random
import time
import warnings
import sys
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.append('.')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args: argparse.Namespace):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    #cudnn.benchmark = True

    dataset = datasets.__dict__[args.data]
    train_source_dataset = dataset(
        root=args.root, task=args.source, download=False, transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = dataset(
        root=args.root, task=args.target, download=False, transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, task=args.target,
                          download=False, transform=val_transform)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.data == 'DomainNet':
        test_dataset = dataset(root=args.root, task=args.target,
                               evaluate=True, download=False, transform=val_transform)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = val_loader

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # extra loader
    pseudo_selection_dataset = dataset(
        root=args.root, task=args.target, download=False, transform=val_transform)
    pseudo_selection_loader = DataLoader(
            pseudo_selection_dataset, batch_size=args.batch_size*2, shuffle=False, num_workers=args.workers)
    unbias_selection_dataset = dataset(
        root=args.root, task=args.source, download=False, transform=val_transform)
    unbias_selection_loader = DataLoader(
        unbias_selection_dataset, batch_size=args.batch_size*2, shuffle=False, num_workers=args.workers)

    # create model
    class_num = train_source_dataset.num_classes
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    classifier = ImageClassifier(
        backbone, train_source_dataset.num_classes).to(device)
    domain_discri = DomainDiscriminator(
        in_feature=classifier.features_dim, hidden_size=1024).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = StepwiseLR(
        optimizer, init_lr=args.lr, gamma=0.001, decay_rate=0.75)

    # define loss function
    domain_adv = DomainAdversarialLoss(domain_discri).to(device)

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, pseudo_selection_loader, unbias_selection_loader,
              classifier, domain_discri, domain_adv, optimizer, lr_scheduler, epoch, class_num, args)

        # evaluate on validation set
        if args.data == 'VisDA2017':
            acc1 = validate_visda(val_loader, classifier, args)
        else:
            acc1 = validate(val_loader, classifier, args)

        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(classifier.state_dict())
        best_acc1 = max(acc1, best_acc1)
    
    print("best_acc1 = {:3.1f}".format(best_acc1))

    #evaluate on test set
    classifier.load_state_dict(best_model)
    if args.data == 'VisDA2017':
        acc1 = validate_visda(test_loader, classifier, args)
    else:
        acc1 = validate(test_loader, classifier, args)
    print("test_acc1 = {:3.1f}".format(acc1))

    #save the best model
    if args.save:
        model_path = 'meta/dann/models/' + time.strftime('%Y-%m-%d')+' '+args.log_filename+'/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(best_model, model_path)

def getMetaWeight(x_p: torch.FloatTensor, labels_p: torch.LongTensor,
                  x_u: torch.FloatTensor, labels_u: torch.LongTensor,
                  model: ImageClassifier, lr:float, class_num: int):
    x_p = x_p.to(device)
    x_u = x_u.to(device)
    labels_p = labels_p.to(device)
    labels_u = labels_u.to(device)
    meta_net = resnet_meta50(class_num)
    meta_net = meta_net.to(device)
    meta_dict = meta_net.state_dict()

    '''
    for param in meta_net.state_dict():
        print(param,'\t',meta_net.state_dict()[param].size())
    print("\nbase\n")
    for param in model_dict:
        print(param,'\t',model_dict[param].size())
    '''

    model_dict = {k.replace('backbone.', '', 1): v for k,
                  v in model.state_dict().items()}  # remove the prefix
    base_model_dict = {k: v for k, v in model_dict.items() if k in meta_dict}
    meta_dict.update(base_model_dict)
    meta_net.load_state_dict(meta_dict)
    # initial forward pass
    _, y_f_hat_target = meta_net(x_p)
    meta_loss = nn.CrossEntropyLoss(reduction='none')(y_f_hat_target, labels_p)
    eps = Variable(torch.zeros(meta_loss.size()).cuda(), requires_grad=True)
    
    l_f_meta = torch.sum(meta_loss*(eps + torch.ones(meta_loss.size(0)).cuda()/ meta_loss.size(0)))
    meta_net.zero_grad()

    # perform a parameter update
    meta_lr = lr
    grads = torch.autograd.grad(
        l_f_meta, (meta_net.params()), create_graph=True)
    meta_net.update_params(meta_lr, source_params=grads)
    del grads
    # 2nd forward pass and getting the gradients with respect to epsilon
    _, y_g_hat_source = meta_net(x_u)

    l_g_meta = nn.CrossEntropyLoss()(y_g_hat_source, labels_u)
    grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

    # computing the weights
    w_tilde = -grad_eps
    init_w = torch.ones(w_tilde.size(0)).cuda()/w_tilde.size(0)
    w = init_w + 0.01*w_tilde
    
    return w.to(device)

def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          pseudo_selection_loader: DataLoader, unbias_selection_loader: DataLoader,
          model: ImageClassifier, domain_discri: DomainDiscriminator, domain_adv: DomainAdversarialLoss,
          optimizer: SGD, lr_scheduler: StepwiseLR, epoch: int, class_num: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()

    end = time.time()

    phase = 'pretrain'
    pseudo_bs = args.pseudo_batchsize
    if epoch >= args.pre_epochs:
        phase = 'meta st'

        # select pseudo label by confidence
        pseudo_file, len_train_pseudo, correct = select_pseudo_label_by_threshold(
            pseudo_selection_loader, model, epoch)
        log_str = "size: {} correct:{}".format(len_train_pseudo, correct)
        print(log_str)
        # select unbiased source set
        unbias_file = select_unbiased_set(
            unbias_selection_loader, model, domain_discri, epoch, class_num)

        # load meta training data
        dataset = datasets.__dict__[args.data]
        meta_dataset = datasets.__dict__['ImageList']
        pseudo_training_dataset = meta_dataset(
            root="", classes=dataset.CLASSES, data_list_file=pseudo_file, transform=train_transform)  # prep for training
        pseudo_training_loader = DataLoader(pseudo_training_dataset,
                                            batch_size=pseudo_bs, shuffle=True, num_workers=args.workers, drop_last=True)
        unbias_training_dataset = meta_dataset(
            root="", classes=dataset.CLASSES, data_list_file=unbias_file, transform=val_transform)  # prep for testing
        unbias_training_loader = DataLoader(unbias_training_dataset,
                                            batch_size=pseudo_bs, shuffle=True, num_workers=args.workers, drop_last=True)

        pseudo_train_iter = ForeverDataIterator(pseudo_training_loader)
        unbias_train_iter = ForeverDataIterator(unbias_training_loader)

    for i in range(args.iters_per_epoch):
        lr_scheduler.step()

        # measure data loading time
        data_time.update(time.time() - end)

        x_s, labels_s, _ = next(train_source_iter)
        x_t, _, _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        if phase == 'meta st':
            x_p, labels_p, path_p = next(pseudo_train_iter)
            x_u, labels_u, _ = next(unbias_train_iter)
            model.eval()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            x_p = x_p.to(device)
            labels_p=labels_p.to(device)
            w = getMetaWeight(x_p, labels_p, x_u, labels_u, model, lr_scheduler.get_lr(), class_num)
            model.train()

        torch.cuda.empty_cache()
        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        if phase == 'meta st':
            y_p, f_p = model(x_p)
            #y_l = torch.cat((y_s, y_p),dim=0)
            target_loss = nn.CrossEntropyLoss(reduction='none')(y_p, labels_p)
            weighted_target_loss = torch.sum(
                target_loss*w) * args.meta_trade_off  # add: meta trade-off
            cls_loss += weighted_target_loss
        transfer_loss = domain_adv(f_s, f_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, _) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def validate_visda(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    dict = {0: "plane", 1: "bcybl", 2: "bus", 3: "car", 4: "horse", 5: "knife", 6: "mcyle", 7: "person", 8: "plant", \
            9: "sktb", 10: "train", 11: "truck"}
    model.eval()
    with torch.no_grad():
        # tick = 0
        subclasses_correct = np.zeros(12)
        subclasses_bins = np.zeros(12)
        for i, (images, target, _) in enumerate(val_loader):
            # tick += 1
            images = images.to(device)
            output, _ = model(images)
            output = nn.Softmax(dim=1)(output)
            _, pred = torch.max(output.cpu(), dim=1)
            pred = pred.numpy()
            target = target.numpy()
            for i in range(pred.size):
                subclasses_bins[target[i]] += 1
                if pred[i] == target[i]:
                    subclasses_correct[target[i]] += 1
        subclasses_acc = np.divide(subclasses_correct, subclasses_bins)
        for i in range(12):
            log_str1 = '\t{}----------({:.4f})'.format(dict[i], subclasses_acc[i] * 100.0)
            print(log_str1)
            #config["out_file"].write(log_str1 + "\n")
        cls_avg = subclasses_acc.mean()
        cls_avg = cls_avg * 100.0
        avg = np.sum(subclasses_correct)*100.0/np.sum(subclasses_bins)
        print(' * Acc_cls {:.3f} Acc_all {:.3f}'
              .format(cls_avg, avg))
        #config["out_file"].write(log_avg + "\n")
        #config["out_file"].flush()
    return avg

def select_pseudo_label_by_threshold(pseudo_loader, model, iteration):
    model.eval()

    pseudo_path = 'meta/dann/pseudo/' + \
        time.strftime('%Y-%m-%d')+' '+args.log_filename+'/'
    if not os.path.exists(pseudo_path):
        os.makedirs(pseudo_path)
    pseudo_file = pseudo_path+'epoch-'+str(iteration)+'.txt'
    pseudo = open(pseudo_file, "w")
    with torch.no_grad():
        correct = 0
        pseudo_size = 0
        for i, (inputs, label, path) in enumerate(pseudo_loader):
            inputs = inputs.to(device)
            label = label.to(device)
            outputs, _ = model(inputs)
            predict_out = nn.Softmax(dim=1)(outputs)
            confidence, predict = torch.max(predict_out, 1)
            for j, conf in enumerate(confidence):
                if conf.item() > args.pseudo_threshold:
                    pseudo.write(path[j] + " ")
                    pseudo.write("{:d}\n".format(predict[j].item()))
                    pseudo.flush()
                    pseudo_size += 1
                    if predict[j].item() == label[j].item():
                        correct += 1
    pseudo.close()
    model.train()
    return pseudo_file, pseudo_size, correct


def select_unbiased_set(unbias_loader, model, discriminator, iteration, class_num, m=2):
    model.eval()
    discriminator.eval()

    class_sample = [0]*class_num
    distance_heap = MinHeap(key=lambda item: item[0])
    unbias_path = 'meta/dann/unbias/' + \
        time.strftime('%Y-%m-%d')+' '+args.log_filename+'/'
    if not os.path.exists(unbias_path):
        os.makedirs(unbias_path)
    unbias_file = unbias_path+'epoch-'+str(iteration)+'.txt'
    unbias = open(unbias_file, "w")
    per_class_size = m
    with torch.no_grad():
        for i, (inputs, label, path) in enumerate(unbias_loader):
            inputs = inputs.to(device)
            label = label.to(device)
            _, features = model(inputs)

            ad_out = discriminator(features)
            for j, d_score in enumerate(ad_out):
                ad_score = abs(0.5-d_score.item())
                distance_heap.push([ad_score, path[j], label[j]])

        for i in range(len(distance_heap)):
            data = distance_heap.pop()
            class_label = data[2].item()
            path = data[1]
            if class_sample[class_label] < per_class_size:
                unbias.flush()
                unbias.write(path + " ")
                unbias.write("{:d}\n".format(class_label))
                unbias.flush()
                class_sample[class_label] += 1
        unbias.close()
    model.train()
    discriminator.train()
    return unbias_file


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--meta_lr', '--meta-learning-rate', default=0.01, type=float,
                        metavar='MLR', help='meta learning rate.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--trade_off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--meta_trade_off', default=0.1, type=float,
                        help='the trade-off hyper-parameter for target CE loss')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--log_filename', default='log', type=str,
                        help='output file name. ')
    parser.add_argument('--model_name', default='model', type=str,
                        help='output model name. ')
    parser.add_argument('--pre_epochs', default=1, type=int,
                        help='epoch number for pretraining phase. ')
    parser.add_argument('--pseudo_batchsize', default=8, type=int,
                        help='batchsize for pseudo training. ')
    parser.add_argument('--pseudo_threshold', default=0.85, type=float,
                        help='confidence threshold for pseudo label selection. ')
    parser.add_argument('-m', '--per_class', default=2, type=int)
    parser.add_argument('--save', default=False, action='store_true',
                        help='save the model. ')

    args = parser.parse_args()
    print(args)

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        ResizeImage(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        ResizeImage(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    main(args)
