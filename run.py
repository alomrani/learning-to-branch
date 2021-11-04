import torch.nn as nn
import torch
from options import get_options
import os
from itertools import product
import json


def train(opts):
  torch.manual_seed(opts.seed)
  if not os.path.exists(opts.save_dir):
    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), "w") as f:
        json.dump(vars(opts), f, indent=True)
  pretrained_model = "./target_model_param/lenet_mnist_model.pth"
  batch_size = opts.batch_size
  device = opts.device
  # TODO: Add rest of code here, this is the entry file (file that will be run)
  return
