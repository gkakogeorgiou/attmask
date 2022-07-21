import os
import sys
import pickle
import argparse
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np
import utils
import models
import torchvision
import scipy
import pandas as pd
import torch.nn.functional as F

from scipy import io
from torch import nn
from PIL import Image, ImageFile
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms
from eval_knn import extract_features

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform = None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        def img_load(index):
            im = Image.open(self.im_paths[index])
            # convert gray to rgb
            if len(list(im.split())) == 1 : im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
            
            return im

        im = img_load(index)
        target = self.ys[index]

        return im, target, index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]

class CUBirds(BaseDataset):
    def __init__(self, root, mode, transform = None):
        self.root = root + '/CUB_200_2011'
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0,100)
        elif self.mode == 'eval':
            self.classes = range(100,200)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0
        for i in torchvision.datasets.ImageFolder(root=os.path.join(self.root, 'images')).imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(os.path.join(self.root, i[0]))
                index += 1

class Cars(BaseDataset):
    def __init__(self, root, mode, transform = None):
        self.root = root + '/cars196'
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0,98)
        elif self.mode == 'eval':
            self.classes = range(98,196)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        annos_fn = 'cars_annos.mat'
        cars = scipy.io.loadmat(os.path.join(self.root, annos_fn))
        ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]
        im_paths = [a[0][0] for a in cars['annotations'][0]]
        index = 0
        for im_path, y in zip(im_paths, ys):
            if y in self.classes: # choose only specified classes
                self.im_paths.append(os.path.join(self.root, im_path))
                self.ys.append(y)
                self.I += [index]
                index += 1

class SOP(BaseDataset):
    def __init__(self, root, mode, transform = None):
        self.root = root + '/Stanford_Online_Products'
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0,11318)
        elif self.mode == 'eval':
            self.classes = range(11318,22634)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        metadata = open(os.path.join(self.root, 'Ebay_train.txt' if self.classes == range(0, 11318) else 'Ebay_test.txt'))
        for i, (image_id, class_id, _, path) in enumerate(map(str.split, metadata)):
            if i > 0:
                if int(class_id)-1 in self.classes:
                    self.ys += [int(class_id)-1]
                    self.I += [int(image_id)-1]
                    self.im_paths.append(os.path.join(self.root, path))

class Inshop(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform = None):
        self.root = root + '/Inshop_Clothes'
        self.mode = mode
        self.transform = transform
        self.train_ys, self.train_im_paths = [], []
        self.query_ys, self.query_im_paths = [], []
        self.gallery_ys, self.gallery_im_paths = [], []
                    
        data_info = np.array(pd.read_table(self.root +'/Eval/list_eval_partition.txt', header=1, delim_whitespace=True))[:,:]
        #Separate into training dataset and query/gallery dataset for testing.
        train, query, gallery = data_info[data_info[:,2]=='train'][:,:2], data_info[data_info[:,2]=='query'][:,:2], data_info[data_info[:,2]=='gallery'][:,:2]

        #Generate conversions
        lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in train[:,1]])))}
        train[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in train[:,1]])

        lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in np.concatenate([query[:,1], gallery[:,1]])])))}
        query[:,1]   = np.array([lab_conv[int(x.split('_')[-1])] for x in query[:,1]])
        gallery[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in gallery[:,1]])

        #Generate Image-Dicts for training, query and gallery of shape {class_idx:[list of paths to images belong to this class] ...}
        for img_path, key in train:
            self.train_im_paths.append(os.path.join(self.root, 'Img', img_path))
            self.train_ys += [int(key)]

        for img_path, key in query:
            self.query_im_paths.append(os.path.join(self.root, 'Img', img_path))
            self.query_ys += [int(key)]

        for img_path, key in gallery:
            self.gallery_im_paths.append(os.path.join(self.root, 'Img', img_path))
            self.gallery_ys += [int(key)]
            
        if self.mode == 'train':
            self.im_paths = self.train_im_paths
            self.ys = self.train_ys
        elif self.mode == 'query':
            self.im_paths = self.query_im_paths
            self.ys = self.query_ys
        elif self.mode == 'gallery':
            self.im_paths = self.gallery_im_paths
            self.ys = self.gallery_ys

    def nb_classes(self):
        return len(set(self.ys))
            
    def __len__(self):
        return len(self.ys)
            
    def __getitem__(self, index):
        
        def img_load(index):
            im = Image.open(self.im_paths[index])
            # convert gray to rgb
            if len(list(im.split())) == 1 : im = im.convert('RGB') 
            if self.transform is not None:
                im = self.transform(im)
            return im
        
        im = img_load(index)
        target = self.ys[index]

        return im, target, index


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def config_imname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])


def config_qimname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Metric Learning on CUB, CARS, SOP, In-Shop')
    parser.add_argument('--n_last_blocks', default=1, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=1` all the time for k-NN evaluation.""")
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0, 1, 2], type=int,
        help="""Whether or not to use global average pooled features or the [CLS] token.
        We typically set this to 1 for BEiT and 0 for models with [CLS] token (e.g., DINO).
        we set this to 2 for base-size models with [CLS] token when doing linear classification.""")
    parser.add_argument('--data_path', default='/path/to/revisited_paris_oxford/', type=str)
    parser.add_argument('--dataset', default='cub', type=str, choices=['cub', 'cars', 'sop', 'inshop'])
    parser.add_argument('--multiscale', default=False, type=utils.bool_flag)
    parser.add_argument('--imsize', default=224, type=int, help='Image size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag)
    parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base', 
        'vit_large'], help='Architecture.')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--backend", default="gloo", type=str, help="Backend to use.")
    parser.add_argument('--save_features', default=False, type=utils.bool_flag)
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        pth_transforms.Resize(256),
        pth_transforms.CenterCrop(224),
    ])
    
    if args.dataset == 'cub':
        dataset_train = CUBirds(args.data_path, mode="train", transform=transform)
        dataset_query = CUBirds(args.data_path, mode="eval", transform=transform)
    elif args.dataset == 'cars':
        dataset_train = Cars(args.data_path, mode="train", transform=transform)
        dataset_query = Cars(args.data_path, mode="eval", transform=transform)
    elif args.dataset == 'sop':
        #dataset_train = SOP(args.data_path, mode="train", transform=transform)
        dataset_query = SOP(args.data_path, mode="eval", transform=transform)
    elif args.dataset == 'inshop':
        dataset_query = Inshop(args.data_path, mode="query", transform=transform)
        dataset_gallery = Inshop(args.data_path, mode="gallery", transform=transform)

    #sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    '''
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler,
        batch_size = 1,
        num_workers=args.num_workers,
        pin_memory = True,
        drop_last = True,
    )
    '''
    
    if args.dataset != 'inshop':
        dataloader_query = torch.utils.data.DataLoader(
            dataset_query,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        dataloader_query = torch.utils.data.DataLoader(
            dataset_query,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        dataloader_gallery = torch.utils.data.DataLoader(
            dataset_gallery,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    
    #print(f"train: {len(dataset_train)} imgs / query: {len(dataset_query)} imgs")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = models.__dict__[args.arch](
            patch_size=args.patch_size, 
            num_classes=0,
            use_mean_pooling=args.avgpool_patchtokens==1)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    if args.use_cuda:
        model.cuda()
    model.eval()

    # load pretrained weights
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
            # print("Since no pretrained weights have been provided, we load pretrained DINO weights on Google Landmark v2.")
            # model.load_state_dict(torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/dino_vitsmall16_googlelandmark_pretrain/dino_vitsmall16_googlelandmark_pretrain.pth"))
        else:
            print("There is no reference weights available for this model => We use random weights.")

    ############################################################################
    # Step 1: extract features
    #train_features, train_labels = extract_features(model, dataloader_train, args.n_last_blocks, args.avgpool_patchtokens, args.use_cuda, multiscale=args.multiscale)
    query_features, query_labels = extract_features(model, dataloader_query, args.n_last_blocks, args.avgpool_patchtokens, args.use_cuda, multiscale=args.multiscale)

    if args.save_features:
        if not os.path.exists('features'):
            os.makedirs('features')
        np.save('features/iboff_{}_query_features'.format(args.dataset), query_features.float().cpu().numpy())
        np.save('features/iboff_{}_query_labels'.format(args.dataset), query_labels.float().cpu().numpy())

    if args.dataset == 'inshop':
        gallery_features, gallery_labels = extract_features(model, dataloader_gallery, args.n_last_blocks, args.avgpool_patchtokens, args.use_cuda, multiscale=args.multiscale)
        if args.save_features:
            np.save('features/iboff_{}_gallery_features'.format(args.dataset), gallery_features.float().cpu().numpy())
            np.save('features/iboff_{}_gallery_labels'.format(args.dataset), gallery_labels.float().cpu().numpy())

    if utils.get_rank() == 0:  # only rank 0 will work from now on
        # normalize features
        query_features = nn.functional.normalize(query_features, dim=1, p=2)        
        if args.dataset == 'cub' or args.dataset == 'cars':
            K = 32
            Y = []
            cos_sim = torch.mm(query_features, query_features.T)
            Y = query_labels[cos_sim.topk(1 + K)[1][:,1:]]
            Y = Y.float().cpu()
            query_labels = query_labels.float().cpu()
            recall = []
            for k in [1, 2, 4, 8, 16, 32]:
                r_at_k = calc_recall_at_k(query_labels, Y, k)
                recall.append(r_at_k)
                print("R@{} : {:.3f}".format(k, 100 * r_at_k))
        
        elif args.dataset == 'sop':
            K = 1000
            Y = []
            xs = []
            for x in query_features:
                if len(xs)<10000:
                    xs.append(x)
                else:
                    xs.append(x)
                    import pdb;pdb.set_trace()
                    xs = torch.stack(xs, dim=0)
                    cos_sim = torch.mm(xs, query_features.T)
                    #cos_sim = F.linear(xs, query_features.T)
                    y = query_labels[cos_sim.topk(1 + K)[1][:,1:]]
                    Y.append(y.float().cpu())
                    xs = []
            
            xs = torch.stack(xs, dim=0)
            #cos_sim = torch.mm(xs, query_features.T)
            cos_sim = torch.mm(xs, query_features.T)
            y = query_labels[cos_sim.topk(1 + K)[1][:,1:]]
            Y.append(y.float().cpu())
            Y = torch.cat(Y, dim=0)

            recall = []
            for k in [1, 10, 100, 1000]:
                r_at_k = calc_recall_at_k(query_labels, Y, k)
                recall.append(r_at_k)
                print("R@{} : {:.3f}".format(k, 100 * r_at_k))
        
        elif args.dataset == 'inshop':
            gallery_features = nn.functional.normalize(gallery_features, dim=1, p=2)
            K = 50
            Y = []
            xs = []
            
            cos_sim = F.linear(query_features, gallery_features)

            def recall_k(cos_sim, query_T, gallery_T, k):
                m = len(cos_sim)
                match_counter = 0

                for i in range(m):
                    pos_sim = cos_sim[i][gallery_T == query_T[i]]
                    neg_sim = cos_sim[i][gallery_T != query_T[i]]

                    thresh = torch.max(pos_sim).item()

                    if torch.sum(neg_sim > thresh) < k:
                        match_counter += 1
                    
                return match_counter / m
            
            recall = []
            for k in [1, 10, 20, 30, 40, 50]:
                r_at_k = recall_k(cos_sim, query_labels, gallery_labels, k)
                recall.append(r_at_k)
                print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    dist.barrier()



#AttMask CUB
'''
R@1 : 57.174
R@2 : 69.379
R@4 : 80.250
R@8 : 87.779
R@16 : 92.556
R@32 : 96.320
'''

#iBOT CUB
'''
R@1 : 51.384
R@2 : 63.842
R@4 : 74.983
R@8 : 84.065
R@16 : 90.766
R@32 : 94.986
'''

#AttMask Cars 
'''
R@1 : 39.798
R@2 : 50.387
R@4 : 61.419
R@8 : 71.504
R@16 : 81.306
R@32 : 89.300
'''

#iBOT Cars
'''
R@1 : 35.641
R@2 : 45.960
R@4 : 56.389
R@8 : 67.630
R@16 : 78.072
R@32 : 86.767
'''

#AttMask SOP 
'''
R@1 : 59.043
R@10 : 73.925
R@100 : 85.432
R@1000 : 94.721
'''

#iBOT SOP
'''
R@1 : 57.358
R@10 : 72.209
R@100 : 84.086
R@1000 : 93.918
'''

#AttMask In-Shop
'''
R@1 : 40.737
R@10 : 63.694
R@20 : 70.333
R@30 : 74.005
R@40 : 76.649
R@50 : 78.372
'''

#iBOT In-Shop
'''
R@1 : 39.091
R@10 : 61.879
R@20 : 68.160
R@30 : 71.719
R@40 : 74.237
R@50 : 76.150
'''