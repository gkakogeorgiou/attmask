# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from iBOT library:
https://github.com/bytedance/ibot
"""

import torch
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--checkpoint_pretraining', required=True, type=str)
parser.add_argument('--checkpoint_linear', required=True, type=str)
parser.add_argument('--output_file', required=True, type=str)
parser.add_argument('--checkpoint_key', required=False, type=str, default = 'teacher')

if __name__ == "__main__":
    args = parser.parse_args()
    backbone = torch.load(args.checkpoint_pretraining)[args.checkpoint_key]
    model = torch.load(args.checkpoint_linear)
    linear = model['state_dict']
    head_index = model['epoch']

    new_linear = {}
    for key, val in linear.items():
        splits = key.split('.')
        new_linear['.'.join(splits[1:])] = val
    backbone.update(new_linear)
    backbone = {k.replace('backbone.', ''):v for k, v in backbone.items()}
    model['state_dict'] = backbone
    
    print(f"save {head_index}th head with acc {model['best_acc']}")
    torch.save(model, args.output_file)