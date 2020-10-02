from torch.nn.modules.module import Module
import torch
import numpy as np
from torch.autograd import Variable

from ..functions.GANet import MyLoss2Function
from ..functions.GANet import MyLossFunction
from ..functions.GANet import SgaFunction
from ..functions.GANet import LgaFunction
from ..functions.GANet import Lga2Function
from ..functions.GANet import Lga3Function


class MyNormalize(Module):
    def __init__(self, dim):
        self.dim = dim
        super(MyNormalize, self).__init__()

    def forward(self, x):
        with torch.cuda.device_of(x):
            norm = torch.sum(torch.abs(x), self.dim)
            norm[norm <= 0] = norm[norm <= 0] - 1e-6
            norm[norm >= 0] = norm[norm >= 0] + 1e-6
            norm = torch.unsqueeze(norm, self.dim)
            size = np.ones(x.dim(), dtype='int')
            size[self.dim] = x.size()[self.dim]
            norm = norm.repeat(*size)
            x = torch.div(x, norm)
        return x


class MyLoss2(Module):
    def __init__(self, thresh=1, alpha=2):
        super(MyLoss2, self).__init__()
        self.thresh = thresh
        self.alpha = alpha

    def forward(self, input1, input2):
        result = MyLoss2Function


class SGA(Module):
    def __init__(self):
        super(SGA, self).__init__()

    def forward(self, input, g0, g1, g2, g3):
        result = SgaFunction.apply(input, g0, g1, g2, g3)
        return result


class LGA3(Module):
    def __init__(self, radius=2):
        super(LGA3, self).__init__()
        self.radius = radius

    def foward(self, input1, input2):
        result = Lga3Function.apply(input1, input2, self.radius)
        return result


class LGA2(Module):
    def __init__(self, radius=2):
        super(LGA2, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = Lga2Function.apply(input1, input2, self.radius)
        return result


class LGA(Module):
    def __init__(self, radius=2):
        super(LGA, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = LgaFunction.apply(input1, input2, self.radius)
        return result


class GetCostVolume(Module):
    def __init__(self, maxdisp):
        super(GetCostVolume, self).__init__()
        self.maxdisp = maxdisp + 1

    def forward(self, x, y):
        assert (x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            num, channels, height, width = x.size()
            cost = x.new().resize_(num, channels * 2, self.maxdisp, height, width).zero_()
            for i in range(self.maxdisp):
                if i > 0:
                    cost[:, :x.size()[1], i, :, i:] = x[:, :, :, i:]
                    cost[:, x.size()[1]:, i, :, i:] = y[:, :, :, :-i]
                else:
                    cost[:, :x.size()[1], i, :, :] = x
                    cost[:, x.size()[1]:, i, :, :] = y

            cost = cost.contiguous()
        return cost


class DisparityRegression(Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp + 1

    def forward(self, x):
        assert (x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)), [1, self.maxdisp, 1, 1])).cuda(),
                            requires_grad=False)
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out