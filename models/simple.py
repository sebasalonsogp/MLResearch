import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

class SmallNetA(nn.Module):
    def __init__(self, nclass=10, scale=64, channels=3, drp=0, **kwargs):
        super(SmallNetA, self).__init__()
        #self.kwargs = kwargs
        self.in_planes = scale
        self.channels = channels
        self.drp = drp
        self.player = kwargs['proto_layer']
        print (int(scale))

        #32x32
        self.block1 = nn.Sequential(
            nn.Conv2d(self.channels, int(scale), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(scale)),
            nn.ReLU(True),
            nn.Conv2d(int(scale), int(scale), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(scale)),
            nn.ReLU(True),
            nn.AvgPool2d(2)
            )
        #16x16

        self.block2 = nn.Sequential(
            nn.Conv2d(int(scale), int(2*scale), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(2*scale)),
            nn.ReLU(True),
            nn.Conv2d(int(2*scale), int(2*scale), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(2*scale)),
            nn.ReLU(True),
            nn.AvgPool2d(2)
            )
        #8x8

        self.block3 = nn.Sequential(
            nn.Conv2d(int(2*scale), int(4*scale), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(4*scale)),
            nn.ReLU(True),
            nn.Conv2d(int(4*scale), int(4*scale), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(4*scale)),
            nn.ReLU(True),
            nn.Conv2d(int(4*scale), int(4*scale), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(4*scale)),
            nn.ReLU(True),
            nn.AvgPool2d(2)
            )
        #4x4

        self.block4 = nn.Sequential(
            nn.Conv2d(int(4*scale), int(8*scale), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(8*scale)),
            nn.ReLU(True),
            nn.Conv2d(int(8*scale), int(8*scale), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(8*scale)),
            nn.ReLU(True),
            nn.Conv2d(int(8*scale), int(8*scale), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(8*scale)),
            nn.ReLU(True),
            nn.AvgPool2d(2)
            )
        #2x2

        self.block5 = nn.Sequential(
            nn.Conv2d(int(8*scale), int(8*scale), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(8*scale)),
            nn.ReLU(True),
            nn.Conv2d(int(8*scale), int(8*scale), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(8*scale)),
            nn.ReLU(True),
            nn.Conv2d(int(8*scale), int(8*scale), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(8*scale)),
            nn.ReLU(True),
            nn.AvgPool2d(2)
            )



        #c x 32 x 32
        #self.conv1 = nn.Conv2d(self.channels, int(scale), kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(int(scale))
        #self.pool1 = nn.AvgPool2d(2)
        #c x 16 x 16
        
        #self.conv2 = nn.Conv2d(int(scale), int(2*scale), kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(int(2*scale))
        #self.pool2 = nn.AvgPool2d(2)
        #c x 8 x 8

        #self.conv3 = nn.Conv2d(int(2*scale), int(4*scale), kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn3 = nn.BatchNorm2d(int(4*scale))
        #self.pool3 = nn.AvgPool2d(2)
        #c x 4 x 4

        #self.conv4 = nn.Conv2d(int(4*scale), int(8*scale), kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn4 = nn.BatchNorm2d(int(8*scale))
        #self.pool4 = nn.AvgPool2d(2)
        #c x 2 x 2
        
        #self.conv5 = nn.Conv2d(int(8*scale), int(8*scale), kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn5 = nn.BatchNorm2d(int(8*scale))
        #self.conv6 = nn.Conv2d(int(8*scale), int(8*scale), kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn6 = nn.BatchNorm2d(int(8*scale))
        #self.pool5 = nn.AvgPool2d(2)
        #c x 1 x 1

        #self.linear1 = nn.Linear(int(8*scale) , int(8*scale))
        #self.nl = nn.Tanh()
        #self.linear2 = nn.Linear(int(8*scale), nclass)

        if self.drp:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(int(8*scale), int(8*scale)),
                nn.ReLU(True),
                nn.Linear(int(8*scale), nclass),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(int(8*scale), int(8*scale)),
                nn.ReLU(True),
                nn.Linear(int(8*scale), nclass),
            )


        self.multi_out = 0
        self.linear1 = nn.Linear(int(8*scale),int(8*scale))
        self.nl = nn.ReLU(True)
        self.linear2 = nn.Linear(int(8*scale),nclass)

    def transfer_param(self):
        #print (self.classifier)
        #print (self.classifier[0].weight.shape)
        #print (self.classifier[2].weight.shape)
        
        self.linear1.weight.data = self.classifier[0].weight.data.clone()
        self.linear1.bias.data = self.classifier[0].bias.data.clone()
        self.linear2.weight.data = self.classifier[2].weight.data.clone()
        self.linear2.bias.data = self.classifier[2].bias.data.clone()
        

            
    def forward(self, x):
        #out = self.pool1(F.relu(self.bn1(self.conv1(x))))
        #out = self.pool2(F.relu(self.bn2(self.conv2(out))))
        #out = self.pool3(F.relu(self.bn3(self.conv3(out))))
        #out = self.pool4(F.relu(self.bn4(self.conv4(out))))
        #out = F.relu(self.bn5(self.conv5(out)))
        #out = self.pool5(F.relu(self.bn6(self.conv6(out))))
        

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        p3_ = F.adaptive_avg_pool2d(out,(1,1))
        p3 = p3_.view(p3_.size(0),-1)

        out = self.block4(out)
        
        p4_ = F.adaptive_avg_pool2d(out,(1,1))
        p4 = p4_.view(p4_.size(0),-1)

        out = self.block5(out)
        
        p5 = out.view(out.size(0), -1)

        #out = self.classifier(out)

        p7 = self.linear1(p5)
        p6 = self.nl(p7)
        out = self.linear2(p6)
        
        if (self.multi_out):

            if self.player == 4:
                return p4, out
            elif self.player == 5:
                return p5, out
            elif self.player ==6:
                return p6, out
            elif self.player ==3:
                return p3, p4, out
            elif self.player ==7:
                return p7, out
        else:
            return out


class LogNetBaselineA(nn.Module):
    def __init__(self, nclass=10, scale=1, channels=3, **kwargs):
        super(LogNetBaselineA, self).__init__()
        #self.kwargs = kwargs                                                                                                                                                                                                          
        self.in_planes = int(32 * scale)
        self.channels = channels
        self.droprate = kwargs['droprate']
        self.drop0 = nn.Dropout(p=self.droprate[0])


        #c x 32 x 32                                                                                                                                                                                                                   
        self.conv1 = nn.Conv2d(self.channels, int(32 * scale), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.LayerNorm([int(32*scale),32,32], elementwise_affine=False)
    
        #c x 16 x 16                                                                                                                                                                                                                   

        self.conv2 = nn.Conv2d(int(32*scale), int(32*scale), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.LayerNorm([int(32*scale),32,32],elementwise_affine=False)
        self.pool2 = nn.AvgPool2d(2)
        #c x 8 x 8                                                                                                                                                                                                                     

        self.conv3 = nn.Conv2d(int(32*scale), int(64*scale), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.LayerNorm([int(64*scale),16,16])
        self.pool3 = nn.AvgPool2d(2)
        #c x 4 x 4

        self.conv4 = nn.Conv2d(int(64*scale), int(128*scale), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.LayerNorm([int(128*scale),8,8])
        self.pool4 = nn.AvgPool2d(2)

        self.conv5 = nn.Conv2d(int(128*scale), int(256*scale), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.LayerNorm([int(256*scale),4,4])
        self.pool5 = nn.AvgPool2d(2)



        self.linear = nn.Linear(int(256 * scale) , nclass)

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
        #normalize in loss function                                                                                                                                                        
        return features.view(features.shape[0],-1)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        if self.droprate[0] > 0.0:
            out = self.drop0(out)
        
        out = F.relu(self.pool2(self.bn2(self.conv2(out))))
        out = F.relu(self.pool3(self.bn3(self.conv3(out))))
        out = F.relu(self.pool4(self.bn4(self.conv4(out))))
        out = F.relu(self.pool5(self.bn5(self.conv5(out))))

        if self.proto_layer == 4:
            p = self.define_proto(out)

        if self.proto_pool == 'ave':
            out = F.avg_pool2d(out, 2)

        if self.proto_pool == 'max':
            out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)

        if self.proto_norm:
            out = F.normalize(out)

        out = self.linear(out)

        if (self.multi_out):
            return p, out
        else:
            return out



def SmallNet(nclass, scale, channels, drop=0, **kwargs):
    
    return SmallNetA(nclass, scale, channels, drop, **kwargs)

def LogNetBaseline(nclass, scale, channels, **kwargs):
    return LogNetBaselineA(nclass,scale, channels, **kwargs)
