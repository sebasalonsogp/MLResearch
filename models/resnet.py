import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

import math
from scipy.special import binom


class LSoftmaxLinear(nn.Module):

    def __init__(self, input_features, output_features, margin, device):
        #super().__init__()
        super(LSoftmaxLinear,self).__init__()
        self.input_dim = input_features  # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.margin = margin  # m
        self.beta = 100
        self.beta_min = 0
        self.scale = 0.99

        self.device = device  # gpu or cpu

        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).to(device)  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).to(device)  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).to(device)  # n
        self.signs = torch.ones(margin // 2 + 1).to(device)  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

        self.training = 1

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta**2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    #def reset_parameters(self):
    #    nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            #print (torch.max(self.weight))
            x, w = input, self.weight
            beta = max(self.beta, self.beta_min)
            logit = x.mm(w)
            indexes = range(logit.size(0))
            logit_target = logit[indexes, target]

            # cos(theta) = w * x / ||w||*||x||
            w_target_norm = w[:, target].norm(p=2, dim=0)
            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            # equation 7
            cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

            # find k in equation 6
            k = self.find_k(cos_theta_target)

            # f_y_i
            logit_target_updated = (w_target_norm *
                                    x_norm *
                                    (((-1) ** k * cos_m_theta_target) - 2 * k))
            logit_target_updated_beta = (logit_target_updated + beta * logit[indexes, target]) / (1 + beta)

            #final logit update
            logit[indexes, target] = logit_target_updated_beta
            self.beta *= self.scale

            #modified effective logit during training
            return logit
        else:
            assert target is None
            
            return input.mm(self.weight)

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, nclass=10, **kwargs):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, nclass)

        self.multi_out = 0
        self.proto_layer = kwargs['proto_layer']
        self.proto_pool = kwargs['proto_pool']
        self.proto_norm = kwargs['proto_norm']
        if self.proto_pool == "max":
            self.proto_pool_f = nn.AdaptiveMaxPool2d((1,1))
        elif self.proto_pool == "ave":
            self.proto_pool_f = nn.AdaptiveAvgPool2d((1,1))

    def define_proto(self,features):
        
        if self.proto_pool in ['max','ave']:
            features = self.proto_pool_f(features)
            
        if self.proto_norm:
            return F.normalize(features.view(features.shape[0],-1))
        else:
            return features.view(features.shape[0],-1)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.proto_layer == 3:
            p = self.define_proto(out)
        out = self.layer4(out)
        if self.proto_layer == 4:
            p = self.define_proto(out)
        out = F.avg_pool2d(out, 4)
        if self.proto_layer == 5:
            p = self.define_proto(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if (self.multi_out):
            return p, out
        else:
            return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, HW, stride=1):
        super(BasicBlock, self).__init__()

        #self.kwargs = kwargs
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # if self.kwargs['norm_type'] == 'batch':
        #     self.bn1 = nn.BatchNorm2d(planes)
        # elif self.kwargs['norm_type'] == 'layer':
        #     self.bn1 = nn.LayerNorm([planes,2048//planes,2048//planes], elementwise_affine=True)
        # elif self.kwargs['norm_type'] == 'instance':
        #     self.bn1 = nn.InstanceNorm2d(planes, affine=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # if self.kwargs['norm_type'] == 'batch':
        #     self.bn2 = nn.BatchNorm2d(planes)
        # elif self.kwargs['norm_type'] == 'layer':
        #     self.bn2 = nn.LayerNorm([planes,2048//planes,2048//planes], elementwise_affine=True)
        # elif self.kwargs['norm_type'] == 'instance':
        #     self.bn2 = nn.InstanceNorm2d(planes, affine=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlockLayer(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, HW, stride=1, drp=0):
        super(BasicBlockLayer, self).__init__()

        #self.kwargs = kwargs
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.LayerNorm([planes,HW,HW], elementwise_affine=False)

        # if self.kwargs['norm_type'] == 'batch':
        #     self.bn1 = nn.BatchNorm2d(planes)
        # elif self.kwargs['norm_type'] == 'layer':
        #     self.bn1 = nn.LayerNorm([planes,2048//planes,2048//planes], elementwise_affine=True)
        # elif self.kwargs['norm_type'] == 'instance':
        #     self.bn1 = nn.InstanceNorm2d(planes, affine=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.LayerNorm([planes,HW,HW], elementwise_affine=False)

        # if self.kwargs['norm_type'] == 'batch':
        #     self.bn2 = nn.BatchNorm2d(planes)
        # elif self.kwargs['norm_type'] == 'layer':
        #     self.bn2 = nn.LayerNorm([planes,2048//planes,2048//planes], elementwise_affine=True)
        # elif self.kwargs['norm_type'] == 'instance':
        #     self.bn2 = nn.InstanceNorm2d(planes, affine=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True),
                nn.LayerNorm([self.expansion * planes,HW,HW], elementwise_affine=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, HW, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nclass=10, scale=64, channels=3, bn=1, **kwargs):
        super(ResNet, self).__init__()
        #self.kwargs = kwargs
        self.scale = int(scale)
        self.in_planes = int(scale)
        self.channels = channels
        self.player = kwargs['proto_layer']
        self.ln = kwargs['layer_norm']
        self.entry = kwargs['entry_stride']
        #self.drp = drp
        print (self.in_planes)



        if bn:
            if self.entry == 1:
                self.conv1 = nn.Conv2d(self.channels, int(scale), kernel_size=3, stride=1, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(int(scale))
                self.relu1 = nn.ReLU(inplace=True)
                self.block0 = nn.Sequential(self.conv1, self.bn1, self.relu1)
            else: 
                self.conv1 = nn.Conv2d(self.channels, int(scale), kernel_size=7, stride=2, padding=3, bias=False)
                self.bn1 = nn.BatchNorm2d(int(scale))
                self.relu1 = nn.ReLU(inplace=True)
                self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.block0 = nn.Sequential(self.conv1, self.bn1, self.relu1, self.pool1)
        else:
            self.conv1 = nn.Conv2d(self.channels, int(scale), kernel_size=3, stride=1, padding=1, bias=True)
            self.bn1 = nn.LayerNorm([int(scale),32,32], elementwise_affine=False)



        # if self.kwargs['norm_type'] == 'batch':
        #     self.bn1 = nn.BatchNorm2d(64)
        # elif self.kwargs['norm_type'] == 'layer':
        #     self.bn1 = nn.LayerNorm([self.in_planes,32,32], elementwise_affine=True)
        # elif self.kwargs['norm_type'] == 'instance':
        #     self.bn1 = nn.InstanceNorm2d(64, affine=True)
        self.layer1 = self._make_layer(block, int(scale), num_blocks[0], HW=32, stride=1)
        self.layer2 = self._make_layer(block, int(2 * scale), num_blocks[1], HW=16, stride=2)
        self.layer3 = self._make_layer(block, int(4 * scale), num_blocks[2], HW=8, stride=2)
        self.layer4 = self._make_layer(block, int(8 * scale), num_blocks[3], HW=4, stride=2)
        self.linear = nn.Linear(int(8 * scale) * block.expansion, nclass)

        self.multi_out = 1 #NOTE for logits + fv?
        #self.proto_layer = kwargs['proto_layer']
        #self.proto_pool = kwargs['proto_pool']
        #self.proto_norm = kwargs['proto_norm']
        
        self.proto_pool_f = nn.AdaptiveAvgPool2d((1,1))

        self.params_conv = nn.ModuleList([self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4])
        self.params_fc = nn.ModuleList([self.linear])

            
    def _make_layer(self, block, planes, num_blocks, HW, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, HW, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = self.block0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        p3_ = F.adaptive_avg_pool2d(out,(1,1))
        p3 = p3_.view(p3_.size(0),-1)

        out = self.layer4(out)

        p4_ = F.adaptive_avg_pool2d(out,(1,1))
        if self.ln:
            p4_ = F.layer_norm(p4_,[int(8*self.scale),1,1])
            #std, mean = torch.std_mean(p4_,dim=1)
            #print (mean[0])
            #print (std[0])
        p4 = p4_.view(p4_.size(0),-1)

        #out = self.proto_pool_f(out)
        
        #p = out.view(out.size(0), -1)

        out = self.linear(p4)
        
        if (self.multi_out):
            if self.player==34:
                return p3, p4, out
            elif self.player==3:
                return p3, out
            else:
                return p4, out
        else:
            return out

class ResNetLargeMargin(nn.Module):
    def __init__(self, block, num_blocks, device, nclass=10, scale=64, channels=3, bn=1, **kwargs):
        super(ResNetLargeMargin, self).__init__()
        #self.kwargs = kwargs
        self.scale = int(scale)
        self.in_planes = int(scale)
        self.channels = channels
        self.player = kwargs['proto_layer']
        self.ln = kwargs['layer_norm']
        self.margin = kwargs['margin']
        self.entry = kwargs['entry_stride']
        self.device = device
        #self.drp = drp
        #print (self.in_planes)

        if bn:
            if self.entry == 1:
                self.conv1 = nn.Conv2d(self.channels, int(scale), kernel_size=3, stride=1, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(int(scale))
                self.relu1 = nn.ReLU(inplace=True)
                self.block0 = nn.Sequential(self.conv1, self.bn1, self.relu1)
            else: 
                self.conv1 = nn.Conv2d(self.channels, int(scale), kernel_size=7, stride=2, padding=3, bias=False)
                self.bn1 = nn.BatchNorm2d(int(scale))
                self.relu1 = nn.ReLU(inplace=True)
                self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.block0 = nn.Sequential(self.conv1, self.bn1, self.relu1, self.pool1)
        else:
            self.conv1 = nn.Conv2d(self.channels, int(scale), kernel_size=3, stride=1, padding=1, bias=True)
            self.bn1 = nn.LayerNorm([int(scale),32,32], elementwise_affine=False)

        # if self.kwargs['norm_type'] == 'batch':
        #     self.bn1 = nn.BatchNorm2d(64)
        # elif self.kwargs['norm_type'] == 'layer':
        #     self.bn1 = nn.LayerNorm([self.in_planes,32,32], elementwise_affine=True)
        # elif self.kwargs['norm_type'] == 'instance':
        #     self.bn1 = nn.InstanceNorm2d(64, affine=True)
        self.layer1 = self._make_layer(block, int(scale), num_blocks[0], HW=32, stride=1)
        self.layer2 = self._make_layer(block, int(2 * scale), num_blocks[1], HW=16, stride=2)
        self.layer3 = self._make_layer(block, int(4 * scale), num_blocks[2], HW=8, stride=2)
        self.layer4 = self._make_layer(block, int(8 * scale), num_blocks[3], HW=4, stride=2)
        #self.linear = nn.Linear(int(8 * scale) * block.expansion, nclass)

        self.multi_out = 0
        #self.proto_layer = kwargs['proto_layer']
        #self.proto_pool = kwargs['proto_pool']
        #self.proto_norm = kwargs['proto_norm']
        
        self.proto_pool_f = nn.AdaptiveAvgPool2d((1,1))

        self.linear = LSoftmaxLinear(
            input_features=int(8 * scale), output_features=nclass, margin=self.margin, device=self.device)

        #self.params_conv = nn.ModuleDict(nn.ModuleList([self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]))
        #self.params_fc = nn.ModuleDict(nn.ModuleList([self.linear]))

            
    def _make_layer(self, block, planes, num_blocks, HW, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, HW, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, target=None):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = self.block0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        p3_ = F.adaptive_avg_pool2d(out,(1,1))
        p3 = p3_.view(p3_.size(0),-1)

        out = self.layer4(out)

        p4_ = F.adaptive_avg_pool2d(out,(1,1))
        if self.ln:
            p4_ = F.layer_norm(p4_,[int(8*self.scale),1,1])
            #std, mean = torch.std_mean(p4_,dim=1)
            #print (mean[0])
            #print (std[0])
        p4 = p4_.view(p4_.size(0),-1)

        #out = self.proto_pool_f(out)
        
        #p = out.view(out.size(0), -1)

        out = self.linear(p4, target=target)
        
        if (self.multi_out):
            if self.player==34:
                return p3, p4, out
            elif self.player==3:
                return p3, out
            else:
                return p4, out
        else:
            return out

class ResNetIN(nn.Module):
    def __init__(self, block, num_blocks, nclass=10, scale=1, **kwargs):
        super(ResNetIN, self).__init__()
        #self.kwargs = kwargs        
        self.in_planes = 64 // scale

        self.conv1 = nn.Conv2d(3, 64 // scale, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64 // scale)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64//scale, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//scale, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256//scale, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512//scale, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512//scale * block.expansion, nclass)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.multi_out = 0
        self.proto_layer = kwargs['proto_layer']
        self.proto_pool = kwargs['proto_pool']
        self.proto_norm = kwargs['proto_norm']

        
        if self.proto_pool == "max":
            self.proto_pool_f = nn.AdaptiveMaxPool2d((1,1))
        elif self.proto_pool == "ave":
            self.proto_pool_f = nn.AdaptiveAvgPool2d((1,1))

    def define_proto(self,features):
        if self.proto_pool in ['max','ave']:
            features = self.proto_pool_f(features)

        #if self.proto_norm:
        #    return F.normalize(features.view(features.shape[0],-1))
        #else:
        return features.view(features.shape[0],-1)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.proto_layer == 3:
            p = self.define_proto(out)
            
        out = self.layer4(out)
        
        if self.proto_layer == 4:
            p = self.define_proto(out)
            
        #out = self.avgpool(out)

        if self.proto_pool == 'ave':
            out = F.avg_pool2d(out, 4)

        if self.proto_pool == 'max':
            out = F.max_pool2d(out, 4)

        
        #if self.proto_layer == 5:
        #    p = self.define_proto(out)
            
        out = out.view(out.size(0), -1)

        if self.proto_norm:
            out = F.normalize(out)

        
        out = self.linear(out)
        
        if (self.multi_out):
            return p, out
        else:
            return out

class ResNetTiny(nn.Module):
    def __init__(self, block, num_blocks, nclass=10, scale=1, channels=3, bn=1, **kwargs):
        super(ResNetTiny, self).__init__()
        #self.kwargs = kwargs        
        self.in_planes = int(scale)
        self.channels = channels
        self.player = kwargs['proto_layer']
        self.ln = kwargs['layer_norm']

        #self.conv1 = nn.Conv2d(self.channels, int(64 * scale), kernel_size=7, stride=2, padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(int(64 * scale))

        if bn:
            self.conv1 = nn.Conv2d(self.channels, int(scale), kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(int(scale))
        else:
            self.conv1 = nn.Conv2d(self.channels, int(scale), kernel_size=7, stride=2, padding=3, bias=True)
            self.bn1 = nn.LayerNorm([int(scale),32,32], elementwise_affine=False)

        self.layer1 = self._make_layer(block, int(scale), num_blocks[0], HW=32, stride=1)
        self.layer2 = self._make_layer(block, int(2*scale), num_blocks[1], HW=16, stride=2)
        self.layer3 = self._make_layer(block, int(4*scale), num_blocks[2], HW=8, stride=2)
        self.layer4 = self._make_layer(block, int(8*scale), num_blocks[3], HW=4, stride=2)
        self.linear = nn.Linear(int(8*scale) * block.expansion, nclass)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)

        self.multi_out = 0
        #self.proto_layer = kwargs['proto_layer']
        #self.proto_pool = kwargs['proto_pool']
        #self.proto_norm = kwargs['proto_norm']
        self.proto_pool_f = nn.AdaptiveAvgPool2d((1,1))
        
        #if self.proto_pool == "max":
        #    self.proto_pool_f = nn.AdaptiveMaxPool2d((1,1))
        #elif self.proto_pool == "ave":
        #    self.proto_pool_f = nn.AdaptiveAvgPool2d((1,1))
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, num_blocks, HW, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, HW, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        p3_ = F.adaptive_avg_pool2d(out,(1,1))
        p3 = p3_.view(p3_.size(0),-1)
        #if self.proto_layer == 3:
        #    p = self.define_proto(out)
            
        out = self.layer4(out)
        p4_ = F.adaptive_avg_pool2d(out,(1,1))
        if self.ln:
            p4_ = F.layer_norm(p4_,[int(8*self.scale),1,1])
            #std, mean = torch.std_mean(p4_,dim=1)                                                                                                                                                                                    
            #print (mean[0])                                                                                                                                                                                                          
            #print (std[0])                                                                                                                                                                                                           
        p4 = p4_.view(p4_.size(0),-1)

        
        #if self.proto_layer == 4:
        #p = self.define_proto(out)
            
        #out = self.avgpool(out)

        #if self.proto_pool == 'ave':
        #out = self.proto_pool_f(out)

        #if self.proto_pool == 'max':
        #out = F.max_pool2d(out, 4)

        
        #if self.proto_layer == 5:
        #    p = self.define_proto(out)
            
        #p = out.view(out.size(0), -1)

        #if self.proto_norm:
        #    out = F.normalize(out)

        
        out = self.linear(p4)

        if (self.multi_out):
            if self.player==34:
                return p3, p4, out
            elif self.player==3:
                return p3, out
            else:
                return p4, out
        else:
            return out

class TransferWrapper(nn.Module):
    def __init__(self, extractor, nftr, nclass, fcdepth, fcratio, **kwargs):
        super(TransferWrapper, self).__init__()
        #self.kwargs = kwargs        vbvbvvvb
        self.extractor = extractor
        self.nclass = nclass
        self.fcdepth = fcdepth
        self.fcratio = fcratio
        self.nftr = nftr
        self.multi_out = 0


        curmaps = self.nftr
        fclyrs=[]
        for fclyr in range(1,self.fcdepth):
            fclyrs.append(nn.Linear(curmaps, int(self.fcratio*curmaps)))
            fclyrs.append(nn.Tanh())
            curmaps = int(self.fcratio*curmaps)

        fclyrs.append(nn.Linear(curmaps, nclass))

        self.linear = nn.Sequential(*fclyrs)


    def forward(self, x):
        p = self.extractor(x)
        
        out = self.linear(p)
        
        if (self.multi_out):
            return p, out
        else:
            return out


def PreActResNet18(nclass, **kwargs):
    return PreActResNet(PreActBlock, [2,2,2,2], nclass, **kwargs)

def ResNet18(nclass, scale, channels, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclass, scale, channels, 1, **kwargs)

def ResNet34(nclass, scale, channels, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], nclass, scale, channels, 1, **kwargs)


def ResNet18Large(nclass, scale, channels, device, **kwargs):
    return ResNetLargeMargin(BasicBlock, [2, 2, 2, 2], device, nclass, scale, channels, 1, **kwargs)

def ResNet34Large(nclass, scale, channels, device, **kwargs):
    return ResNetLargeMargin(BasicBlock, [3, 4, 6, 3], device, nclass, scale, channels, 1, **kwargs)


def ResNet18Tiny(nclass, scale, channels, **kwargs):
    return ResNetTiny(BasicBlock, [2, 2, 2, 2], nclass, scale, channels, 1, **kwargs)

def ResNet50Tiny(nclass, scale, channels, **kwargs):
    return ResNetTiny(Bottleneck, [3, 4, 6, 3], nclass, scale, channels, 1, **kwargs)



def ResNet18L(nclass, scale, channels, **kwargs):
    return ResNet(BasicBlockLayer, [2, 2, 2, 2], nclass, scale, channels, 0, **kwargs)

def ResNet18TinyL(nclass, scale, channels, **kwargs):
    return ResNetTiny(BasicBlockLayer, [2, 2, 2, 2], nclass, scale, channels, 0, **kwargs)

def TransferWrapperA(extractor, nftr, nclass, fcdepth, fcratio, **kwargs):
    return TransferWrapper(extractor, nftr, nclass, fcdepth=1, fcratio=1, **kwargs)


def ResNet18IN(nclass, scale, **kwargs):
    return ResNetIN(BasicBlock, [2, 2, 2, 2], nclass, scale, **kwargs)


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()
        m.training=False
        #m.weight.requires_grad = True
        #m.bias.requires_grad = True
        #print (torch.mean(m.bias))
        #print (m.running_mean)

def set_bn_train(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()
