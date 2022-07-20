# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Based on DINO library:
https://github.com/facebookresearch/dino
"""

import os
import argparse
import copy
import torch
import torch.backends.cudnn as cudnn

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from attmask import AttMask
import utils
import models

from pathlib import Path
from torch import nn
from torchvision import transforms as pth_transforms
from loader import ImageFolder

from models.vision_transformer import vit_base

def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    
    # fix the seed for reproducibility 
    utils.fix_random_seeds(args.seed)

    # ============ preparing data ... ============
    if args.arch == 'dalle_encoder':

        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(128, interpolation=3),
            pth_transforms.CenterCrop(112),
            pth_transforms.ToTensor(),
        ])
    else:

        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    valdir=os.path.join(args.data_path, "val")
    dataset_val = ImageFolder(valdir, transform=val_transform)
    

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if 'swin' in args.arch:
        args.patch_size = 4
        model = models.__dict__[args.arch](
            window_size=args.window_size,
            patch_size=args.patch_size,
            num_classes=0)
        embed_dim = model.num_features
    else:
        model = models.__dict__[args.arch](
            patch_size=args.patch_size, 
            num_classes=0,
            use_mean_pooling=args.avgpool_patchtokens==1,
            masked_im_modeling=False) # CHANGE THIS FOR MIM
        embed_dim = model.embed_dim
    model.cuda()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    
    if 'swin' in args.arch:
        num_features = []
        for i, d in enumerate(model.depths):
            num_features += [int(model.embed_dim * 2 ** i)] * d
        feat_dim = sum(num_features[-args.n_last_blocks:])
    else:
        feat_dim = embed_dim * (args.n_last_blocks * int(args.avgpool_patchtokens != 1) + \
            int(args.avgpool_patchtokens > 0))
    linear_classifier = LinearClassifier(feat_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # Load Attentioner DINO model
    attentioner = vit_base(patch_size=16, num_classes=0)
    print("We load the reference pretrained DINO weights to extract self-attention for fair comparison.")
    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth")
    attentioner.load_state_dict(state_dict, strict=True)
    attentioner.cuda()
    attentioner.eval()
    
    # set optimizer
    parameters = linear_classifier.parameters()

    optimizer = torch.optim.SGD(
        parameters,
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            state_dict=linear_classifier,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    best_acc = to_restore["best_acc"]

    model.eval()
    linear_classifier.eval()
    test_stats = validate_network(val_loader, model, linear_classifier, attentioner, args.n_last_blocks, args.avgpool_patchtokens)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            
    best_acc = max(best_acc, test_stats["acc1"])
    print(f'Max accuracy so far: {best_acc:.2f}%')
            
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))

@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, attentioner, n, avgpool):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            
            # Mask from Attentioner
            attentions = attentioner.get_last_selfattention(inp)
            # Get mean [CLS] token attention
            cls_attention = attentions[:, :, 0, 1:].mean(1).detach().clone()

            # Get AttMask. cls_attention should be in shape (batch_size, number_of_tokens)
            mask = AttMask( cls_attention,
                            1.0,
                            'attmask_high',
                            args.masking_ratio,
                            None
                            )
            
            mask = mask.reshape(-1, 224//args.patch_size, 224//args.patch_size)
            mask_up = nn.functional.interpolate(mask.float().unsqueeze(1), scale_factor=16, mode="nearest").bool().squeeze(1)
            for s in range(mask_up.shape[0]):
                inp[s][:,mask_up[s]] = torch.tensor([-2.1179, -2.0357, -1.8044], device = inp.device).unsqueeze(1) # Replace with Zero values - black pixels (before normalization)

            if avgpool == 0:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = [x[:, 0] for x in intermediate_output]
                output = torch.cat(output, dim=-1)
            else:
                assert False, "Unkown avgpool type {}".format(avgpool)
        
        output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if linear_classifier.module.num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if linear_classifier.module.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base/Large.""")
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0, 1, 2], type=int,
        help="""Whether or not to use global average pooled features or the [CLS] token.
        We typically set this to 1 for BEiT and 0 for models with [CLS] token (e.g., DINO).
        we set this to 2 for base/large-size models with [CLS] token when doing linear classification.""")
    parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base', 
        'vit_large', 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'resnet50', 'resnet101', 'dalle_encoder'], help='Architecture.')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--window_size', default=7, type=int, help='Window size of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str,
        help='Please specify path to the ImageNet data.')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--load_from', default=None, help='Path to load checkpoints to resume training')
    parser.add_argument("--subset", default=-1, type=int, help="The number of images per class that they would be use for "
                        "training (default -1). If -1, then all the availabe images are used.")
    parser.add_argument('--backend', default='nccl', type=str, help='Specify backend nccl or gloo')
    
    # Attention parameters
    parser.add_argument('--masking_ratio', type=float, default=0.3, help="Perform token masking based on attention")
    
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for checkpoint_key in args.checkpoint_key.split(','):
        print("Starting evaluating {}.".format(checkpoint_key))
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = checkpoint_key
        eval_linear(args_copy)
