import torch
import torch.nn
import torch.nn as nn
from libs.GANet.functions.GANet import SgaFunction
from libs.sync_bn.modules.sync_bn import BatchNorm2d, BatchNorm3d

import torch.nn.functional as F
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == "__main__":
    input = torch.rand(1, 3, 5, 10, 10).cuda()
    g0 = torch.rand(1, 3, 10, 10).cuda()
    g1 = torch.rand(1, 3, 10, 10).cuda()
    g2 = torch.rand(1, 3, 10, 10).cuda()
    g3 = torch.rand(1, 3, 10, 10).cuda()

    out = SgaFunction.apply(input, g0, g1, g2, g3)
