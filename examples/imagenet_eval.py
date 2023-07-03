from __future__ import print_function, division, absolute_import
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sys

sys.path.append('.')
import pretrainedmodels
import pretrainedmodels.utils

try:
    from context_func import context_func
except ModuleNotFoundError as e:
    print("!!!pls check how to add context_func.py from launch_benchmark.sh")
    sys.exit(0)

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default="path_to_imagenet",
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='nasnetamobile',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: fbresnet152)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=True,
                    action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', default='imagenet', help='use pre-trained model')
parser.add_argument('--do-not-preserve-aspect-ratio',
                    dest='preserve_aspect_ratio',
                    help='do not preserve the aspect ratio when resizing an image',
                    action='store_false')
# OOB
parser.add_argument('--device', default="cpu", choices=["cpu", "xpu", "cuda"])
parser.add_argument('--jit', action='store_true', default=False,
                    help='enable Intel_PyTorch_Extension JIT path')
parser.add_argument('--nv_fuser', action='store_true', default=False)
parser.add_argument('-i', '--num_iter', default=0, type=int, metavar='N',
                    help='number of total iterations to run')
parser.add_argument('-w', '--num_warmup', default=0, type=int, metavar='N',
                    help='number of warmup iterations to run')
parser.add_argument('--precision', type=str, default="float32",
                    help='precision, float32, int8, float16')
parser.add_argument("-t", "--profile", action='store_true',
                    help="Trigger profile on current topology.")
parser.add_argument("--performance", action='store_true',
                    help="measure performance only, no accuracy.")
parser.add_argument("--dummy", action='store_true',
                    help="using  dummu data to test the performance of inference")
parser.add_argument('--channels_last', type=int, default=1,
                    help='use channels last format')
parser.add_argument('--config_file', type=str, default="./conf.yaml",
                    help='config file for int8 tuning')

parser.set_defaults(preserve_aspect_ratio=True)
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    if args.device == "xpu":
        import intel_extension_for_pytorch
    elif args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.pretrained.lower() not in ['false', 'none', 'not', 'no', '0']:
        print("=> using pre-trained parameters '{}'".format(args.pretrained))
        model = pretrainedmodels.__dict__[args.arch](num_classes=1000,
                                                     pretrained=args.pretrained)
    else:
        model = pretrainedmodels.__dict__[args.arch](pretrained=None)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.device == "cuda":
        cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(traindir, transforms.Compose([
    #         transforms.RandomSizedCrop(max(model.input_size)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)



    # if 'scale' in pretrainedmodels.pretrained_settings[args.arch][args.pretrained]:
    #     scale = pretrainedmodels.pretrained_settings[args.arch][args.pretrained]['scale']
    # else:
    #     scale = 0.875
    scale = 0.875
    if args.arch == "vggm":
        from pretrainedmodels.models import vggm
        pretrainedmodels.pretrained_settings = vggm.pretrained_settings
    opt = pretrainedmodels.pretrained_settings[args.arch]["imagenet"]

    print('Images transformed from size {} to {}'.format(
        int(round(max(opt["input_size"]) / scale)),
        opt["input_size"]))
    #print('Images transformed from size {} to {}'.format(
    #    int(round(max(model.input_size) / scale)),
    #    model.input_size))

    val_tf = pretrainedmodels.utils.TransformImage(
        opt,
        scale=scale,
        preserve_aspect_ratio=args.preserve_aspect_ratio
    )

    if not args.dummy:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, val_tf),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        val_loader = ""

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    #model = torch.nn.DataParallel(model).cuda()
    model = model.to(args.device)

    model.eval()
    if args.device == "xpu":
        datatype = torch.float16 if args.precision == "float16" else torch.bfloat16 if args.precision == "bfloat16" else torch.float
        model = torch.xpu.optimize(model=model, dtype=datatype)
        print("---- enable xpu optmize")

    if args.channels_last or args.device == "cuda":
        try:
            model = model.to(memory_format=torch.channels_last)
            print("---- use NHWC format")
        except RuntimeError as e:
            print("---- use normal format")
            print("failed to enable NHWC: ", e)
    if args.nv_fuser:
       args.fuser_mode = "fuser2"
    else:
       args.fuser_mode = "none"
    print("---- fuser mode:", args.fuser_mode)

    image_size = pretrainedmodels.pretrained_settings[args.arch]["imagenet"]["input_size"]
    images = torch.randn(args.batch_size, *image_size).to(args.device)
    if args.jit:
        try:
            model = torch.jit.trace(model, images, check_trace=False)
            print("---- JIT trace enable.")
        except (RuntimeError, TypeError) as e:
            print("---- JIT trace disable.")
            print("failed to use PyTorch jit mode due to: ", e)

    if args.evaluate:
        if args.precision == "float16":
            amp_dtype = torch.float16
            amp_enabled = True
        elif args.precision == "bfloat16":
            amp_dtype = torch.bfloat16
            amp_enabled = True
        else:
            amp_dtype = torch.float32
            amp_enabled = False
        print("---- amp_enable:{}, amp_dtype:{}".format(amp_enabled, amp_dtype))

        if args.device == "xpu":
            model = torch.xpu.optimize(model, dtype=amp_dtype)
            print("---- enable xpu optimize")

        with torch.autocast(device_type=args.device, enabled=amp_enabled, dtype=amp_dtype):
            validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(args.device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        if args.dummy:
            profile_len = args.num_iter // 2
            loopper = zip(range(args.num_iter), range(args.num_iter))
        else:
            profile_len = min(len(val_loader), args.num_iter) // 2
            loopper = val_loader
        for i, (input, target) in enumerate(loopper):
            if i >= args.num_iter:
                break
            if args.dummy:
                image_size = pretrainedmodels.pretrained_settings[args.arch]["imagenet"]["input_size"]
                input = torch.randn(args.batch_size, *image_size)
            input = input.to(args.device)

            if args.channels_last or args.device == "cuda":
                input = input.to(args.device)

            start = time.time()
            # compute output
            with context_func(args.profile if i == profile_len else False, args.device, args.fuser_mode) as prof:
                output = model(input)
                #loss = criterion(output, target)
            if args.device == "xpu":
                torch.xpu.synchronize()
            elif args.device == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            print("Iteration: {}, inference time: {} sec.".format(i, end - start), flush=True)
            if i >= args.num_warmup:
                batch_time.update(end - start)

        batch_size = args.batch_size
        latency = batch_time.avg / batch_size * 1000
        perf = batch_size/batch_time.avg
        print('inference latency: %3.3f ms'%latency)
        print('inference Throughput: %3.3f fps'%perf)

            # measure accuracy and record loss
            #prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            #losses.update(loss.data.item(), input.size(0))
            #top1.update(prec1.item(), input.size(0))
            #top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            #batch_time.update(start - end)

            #if i % args.print_freq == 0:
            #    print('Test: [{0}/{1}]\t'
            #          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            #          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            #           i, len(val_loader), batch_time=batch_time, loss=losses,
            #           top1=top1, top5=top5))

        #print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #      .format(top1=top1, top5=top5))

        #return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
