from __future__ import print_function
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import datetime
from datetime import datetime
import numpy as np

from torchvision import datasets, transforms, models
from torch.hub import load_state_dict_from_url
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchsummary import summary

from models.resnet import *
from models.simple import *
from models.densenet import *
from loss_utils import *
from utils import *
from utils_proto import *

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# import augmentations
# from color_jitter import *
# from diffeomorphism import *
# from rand_filter import *

# from torch.distributions import Dirichlet, Beta
# from einops import rearrange, repeat
# from opt_einsum import contract

#from utils_confusion import *
#from utils_augmix import *
#from utils_prime import *
#from trades import trades_loss
#from foolbox import PyTorchModel, accuracy, samples
#from foolbox.attacks import L2DeepFoolAttack
#from create_data import compute_smooth_data, merge_data, CustomDataSet

#from robustness.datasets import CustomImageNet
#from robustness.datasets import DATASETS, DataSet, CustomImageNet
#import smoothers

parser = argparse.ArgumentParser(description='PyTorch CIFAR + proximity training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--beta', default=0.0, type=float,
                    help='loss weight for proximity')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many epochs to wait before logging training status')
parser.add_argument('--model-dir', default='../ProtoRuns/metric-cifar10-',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--model',default="ResNet18",
                    help='network to use')
parser.add_argument('--restart',default=0, type=int,
                    help='restart training, make sure to specify directory')
parser.add_argument('--restart-epoch',default=0, type=int,
                    help='epoch to restart from')
parser.add_argument('--norm-type', default='batch',
                    help='batch, layer, or instance')
parser.add_argument('--par-servant', default=0, type=int,
                    help='whether normalization is learnable')
parser.add_argument('--par-sparse', default=0, type=int,
                    help='force L1 sparsity on prototype images')

parser.add_argument('--zetaCov', default=1.0, type=float,
                    help='zeta for covariance')
parser.add_argument('--zetaCS', default=1.0, type=float,
                    help='zeta for proto cs')
#parser.add_argument('--expand-data-epoch', default=0, type=int,
#                    help='start mixing data with misclassified combinations with parametric images')
#parser.add_argument('--expand-interval', default=5, type=int,
#                    help='number of epochs to wait before re-expanding')
#parser.add_argument('--kldiv', default=0, type=int,
#                    help='enforce kldiv match between prototype and examples')
#parser.add_argument('--gamma', default=0.0, type=float,
#                    help='mult loss for kldiv match')
#parser.add_argument('--mixup', default=0, type=int,
#                    help='augment data with mixup par class x to examples not x')
#parser.add_argument('--alpha-mix', default=0.7, type=float,
#                    help='alpha for mixup')
#parser.add_argument('--mix-interval',default=3, type=int,
#                    help='how often to intra class mix')
parser.add_argument('--par-grad-mult', default=10.0, type=float,
                    help='boost image gradients if desired')
parser.add_argument('--par-grad-clip', default=0.01, type=float,
                    help='max magnitude per update for proto image updates')
#parser.add_argument('--class-centers', default=1, type=int,
#                    help='number of parametric centers per class')
parser.add_argument('--dataset', default="CIFAR10",
                    help='which dataset to use, CIFAR10, CIFAR100, IN100')
parser.add_argument('--image-train', default=0, type=int,
                    help='train parametric images on frozen model')
parser.add_argument('--norm-data', default=0, type=int,
                    help='normalize data')
parser.add_argument('--anneal', default="stairstep", 
                    help='type of LR schedule stairstep, cosine, or cyclic')
#parser.add_argument('--inter-mix', default=0, type=int,
#                    help='fill in holes within same class')
# parser.add_argument('--augmix', default=0, type=int,
#                     help='use augmix data augmentation')
# parser.add_argument('--prime', default=0, type=int,
#                     help='use PRIME data augmentation')
# parser.add_argument('--confusionmix', default=0, type=int,
#                     help='use confusionmix data augmentation')
parser.add_argument('--js-loss', default=0, type=int,
                    help='use jensen shannon divergence for augmix')
#parser.add_argument('--pipeline', nargs='+',default=[],
#                    help='augmentation pipeline')
parser.add_argument('--grad-clip', default = 0, type=int,
                    help='clip model weight gradients by 0.5')
#parser.add_argument('--confusion-mode', default = 2, type=int,
#                    help='0 = (mode0,mode0), 1 = (mode1,mode1), 2= (mode0,mode1) 3= (random,random)')
#parser.add_argument('--mode0rand', default = 0, type=int,
#                    help='randomly switch between window crop size 3 and 5 in mode 0')
parser.add_argument('--channel-norm', default = 0, type=int,
                    help='normalize each channel by training set mean and std')
parser.add_argument('--channel-swap', default = 0, type=int,
                    help='randomly permute channels augmentation')
#parser.add_argument('--window', nargs='+', default=[], type=int,
#                    help='possible windows for cutouts')
#parser.add_argument('--counts', nargs='+', default=[], type=int,
#                    help='possible counts for windows')
parser.add_argument('--proto-layer', default = 4, type=int,
                    help='after which block to compute prototype loss')
parser.add_argument('--proto-pool', default ="none",
                    help='whether to adaptive pool proto vector to Cx1 and how')
parser.add_argument('--proto-norm', default = 0, type=int,
                    help='normalize vectors before prototype loss computed')
parser.add_argument('--proto-aug', nargs='+',default=[],
                    help='augmentations for prototype image')
parser.add_argument('--k', default=0, type=int,
                    help='consider only top +k or bottom -k elements of prototype vector, sorting based on prototype')
#parser.add_argument('--decay_pow', default=0.0, type=float,
#                    help='reduce loss by magnitude of prototype')
#parser.add_argument('--decay_const', default=1.0, type=float,
#                    help='reduce loss by magnitude of prototype')
parser.add_argument('--renorm-prox', default=0, type=int,
                    help='set to 1 if proto-norm =0')
parser.add_argument('--psi', default=0.0, type=float,
                    help='weight for proxcos contribution, multiplied by beta')
parser.add_argument('--latent-proto', default=0, type=int,
                    help='whether prototypes should be held in latent space as opposed to image space')
parser.add_argument('--kprox', default=1, type=int,
                    help='topk of each row to consider in proto cosine sim loss')
parser.add_argument('--maxmean', default=1, type=int,
                    help='if 1, will use topk maxes from each row, if 0, topk means from cossim matrix')
parser.add_argument('--proxpwr', default=1.0, type=float,
                    help='power of the L2 dist on data to prototype')
parser.add_argument('--topkprox', default=0, type=int,
                    help='if not 0, will select only topk maxes from kprox selection ie top10 of top5 maxes')
parser.add_argument('--hsphere', default=0, type=int,
                    help='shrink variance on magnitudes to speed convergence')
parser.add_argument('--wfactor', default=2.0, type=float,
                    help='weight multipler for outlier points')
parser.add_argument('--sfactor', default=3.0, type=float,
                    help='sigma factor to identify outliers')
parser.add_argument('--model-scale', default=64, type=int,
                    help='width scale of network off of baseline resnet18')
parser.add_argument('--wxent', default=0, type=int,
                    help='whether to apply boosted weights to xent loss')
parser.add_argument('--boost', default=0, type=int,
                    help='whether to boost after each fold')
parser.add_argument('--droprate', nargs='+', default=[0.0],
                    help='drop out rate')

parser.add_argument('--continuous', default=0, type=int,
                    help='whether to do continuous training of the same model')

parser.add_argument('--lr-fine', default=0.1, type=float,
                    help='lr multiplier on args lr for finetune')
parser.add_argument('--datasplits', nargs='+', default=[1.0], type=float,
                    help='split of data less than 100 pct')
parser.add_argument('--modelinitials', default=5, type=int,
                    help='number of model initializations for each datasplit')
parser.add_argument('--flipcrop', default=1, type=int,
                    help='whether to use flipping augmentation')
parser.add_argument('--image-step', default=-0.1, type=float,
                    help='learning rate for train image no data')
parser.add_argument('--train-cheb', default=0, type=int,
                    help='train on chebyshev risk')
parser.add_argument('--nsample', default=250, type=int,
                    help='num of perturbations')
parser.add_argument('--onehot', default=0, type=int,
                    help='whether to use onehot protos or data protos')
parser.add_argument('--tweak', default=0, type=int,
                    help='whether to augment input data to accomodate losses')
parser.add_argument('--warmup', default=25, type=int,
                    help='warmup period')
parser.add_argument('--layer-norm', default=0, type=int,
                    help="apply layer norm to prototype before loss")
parser.add_argument('--cov-sign', default=-1.0, type=float,
                    help="whether to control negative cov or positive cov")
parser.add_argument('--semi', default=0, type=int,
                    help='whether to draw different random training sets per model initial')
parser.add_argument('--ftrcnt', default=256, type=int,
                    help='number of features in feature layer')
parser.add_argument('--orthoreg', default=0, type=int,
                    help='use orthoreg')
parser.add_argument('--decov', default=0, type=int,
                    help='use decov loss')
parser.add_argument('--w-decov', default=0.1, type=float,
                    help='decov loss weighting')
parser.add_argument('--cheb-cs', default=0, type=int,
                    help='min the cheb cs term')
parser.add_argument('--sq-cs', default=0, type=int,
                    help='min sq cs')
parser.add_argument('--fc-freeze', default=0, type=int,
                    help='augment algorithms with fc orthogonality monitoring')
parser.add_argument('--fc-limit', default=0.0002, type=float,
                    help='criteria for mean CS weight to freeze FC weights and biases')
parser.add_argument('--squentropy', default=0, type=int,
                    help='use squentropy loss')
parser.add_argument('--largemarginCE', default=0, type=int,
                    help='use large margin softmax loss')
parser.add_argument('--margin', default=4, type=int,
                    help='angular margin for large softmax loss')
parser.add_argument('--entry-stride', default=1, type=int,
                    help='defines whether to use a CIFAR or ImageNet style entry in ResNet')
parser.add_argument('--rmin', default=1, type=int,
                    help="minimum shift in CPR algorithm")
parser.add_argument('--rmax', default=10, type=int,
                    help="maximum shift in CPR algorithm")
parser.add_argument('--verbose', default=0, type=int,
                    help="whether to print to std out")

# AugMix options
#parser.add_argument(
#    '--mixture-width',
#    default=3,
#    type=int,
#    help='Number of augmentation chains to mix per augmented example')
#parser.add_argument(
#    '--mixture-depth',
#    default=-1,
#    type=int,
#    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
#parser.add_argument(
#    '--aug-severity',
#    default=3,
#    type=int,
#    help='Severity of base augmentation operators')
# parser.add_argument(
#     '--no-jsd',
#     '-nj',
#     action='store_true',
#     help='Turn off JSD consistency loss.')
#parser.add_argument(
#    '--all-ops',
#    default=1,
#    type=int,
#    help='Turn on all operations (+brightness,contrast,color,sharpness).')



args = parser.parse_args()

kwargsUser = {}
kwargsUser['norm_type'] = args.norm_type
#kwargsUser['augmix'] = args.augmix
#kwargsUser['prime'] = args.prime
kwargsUser['js_loss'] = args.js_loss
kwargsUser['proto_aug'] = args.proto_aug
#kwargsUser['pipeline'] = args.pipeline
#kwargsUser['augmix'] = "augmix" in args.pipeline
#kwargsUser['prime'] = "prime" in args.pipeline
#kwargsUser['confusion'] = "confusion" in args.pipeline
#kwargsUser['pipelength'] = len(args.pipeline)   

kwargsUser['proto_layer'] = args.proto_layer
kwargsUser['proto_pool'] = args.proto_pool
kwargsUser['proto_norm'] = args.proto_norm
#kwargsUser['renorm_prox'] = args.renorm_prox
kwargsUser['psi']= args.psi
kwargsUser['latent_proto'] = args.latent_proto
kwargsUser['kprox'] = args.kprox
kwargsUser['maxmean'] = args.maxmean
kwargsUser['proxpwr'] = args.proxpwr
kwargsUser['topkprox'] = args.topkprox
kwargsUser['hsphere'] = args.hsphere
kwargsUser['wxent'] = args.wxent
kwargsUser['droprate'] = args.droprate
kwargsUser['layer_norm'] = args.layer_norm
kwargsUser['entry_stride'] = args.entry_stride
kwargsUser['margin'] = args.margin
#kwargsUser['scale'] = args.model_scale

assert (args.proto_pool in ['none','max','ave'])



# settings
if (args.model == "ResNet18"):
    network_string = "ResNet18"
elif (args.model == "ResNet34"):
    network_string = "ResNet34"
elif (args.model == "ResNet18L"):
    network_string = "ResNet18L"
elif (args.model == "LogNetBaseline"):
    network_string = "LogNet"
elif (args.model == "PreActResNet18"):
    network_string = "PreActResNet18"
elif (args.model == "SmallNet"):
    network_string = "SmallNet"
elif ("WRN" in args.model):
    network_string = args.model
elif (args.model == "DenseNet"):
    network_string = "DenseNet"
elif (args.model == "ResNeXt"):
    network_string = "ResNeXt"
elif (args.model == "AllConv"):
    network_string = "AllConv"
else:
    print ("Invalid model architecture")
    
def get_datetime():
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H_%M_%S")
    return dt_string



#model_dir = ("{}_{}_beta_{}_k_{}_pool_{}_norm_{}_{}".format("../ProtoRuns/metric-{}".format(args.dataset),network_string,args.beta,args.k,args.proto_pool, args.proto_norm,get_datetime()))

#/home/lab/nxd551/Desktop/PrototypeImage/
targ = "/home/lab/nxd551/Desktop/ProtoRuns"
plottarg = "/home/lab/nxd551/Desktop/PrototypeImage/metric_plots"
dir_suffix = "metric-{}_{}_wd_{}_aug_{}_scale_{}_beta{}_k{}_zCv{}_zCS{}_{}".format(args.dataset,network_string,args.weight_decay,args.flipcrop,args.model_scale,args.beta,args.k,args.zetaCov,args.zetaCS,get_datetime())
#need local global results file                                                                                                                                                                             
#local_results_name = "../ProtoRuns/quicklook_{}_{}.txt".format(args.dataset,get_datetime())                                                                                                                        

#full_dir = os.path.join(targ, model_dir)
model_dir = os.path.join(targ, dir_suffix)
full_dir_plot = os.path.join(plottarg,dir_suffix)

#print (full_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


if not os.path.exists(full_dir_plot):
    os.makedirs(full_dir_plot)


with open('{}/commandline_args.txt'.format(model_dir), 'a') as f:
    json.dump(args.__dict__, f, indent=2)
f.close()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
torch.cuda.empty_cache()

print ("cuda: ", use_cuda)


def train_cheb(args, model, device, cur_loader, optimizer, epoch, par_images, proto_targ, scheduler=0.0, max_steps = 0, transformDict={}, **kwargs):

    model.train()
    model.multi_out=1
    print ('Training model')

    image_lr = abs(args.image_step)
    nclass = len(par_images)
    covsign = args.cov_sign
    zCov = args.zetaCov
    zCS = args.zetaCS
    rmin_cur = args.rmin
    rmax_cur = args.rmax
    verb_cur = args.verbose
    
    prox_criterion = Proximity(device,
                                num_classes=len(par_images),
                                num_ftrs=args.ftrcnt,
                                k=args.k,
                                protonorm=args.proto_norm)


    with torch.no_grad():
       targets_onehot = torch.arange(nclass, dtype=torch.long, device=device) 


    #weight=torch.tensor(0.0)
    for batch_idx, data_all in enumerate(cur_loader):

        for o in optimizer:
            o.zero_grad()

        model.zero_grad()

        model.apply(set_bn_train)

        if args.tweak:
            data, target, noise, indexes = data_all[0].to(device), data_all[1].to(device), data_all[2].to(device), data_all[3]

            noise_batch = noise.clone().detach().requires_grad_(True).to(device)

            _inputs = data + noise_batch
            _inputs.clamp_(0.0, 1.0)

        else:
            _inputs, target = data_all[0].to(device), data_all[1].to(device)
        
        bs = len(target)
        
        data_norm = transformDict['norm'](_inputs)

        
        # lets make par_servant useful again
        #if (args.par_servant):
        #    image_model.load_state_dict(model.state_dict())
        #    image_model.eval()

        #if args.beta > 0.0:
        #    _par_images_opt = par_images.clone().detach().requires_grad_(True).to(device)
        #else:
        #    _par_images_opt = 0.0
        loss = 0.0
        loss_cov = torch.tensor(0.0)
        loss_proto = torch.tensor(0.0)
        loss_cs = torch.tensor(0.0)

        L2_nat, Z_nat = model(data_norm)    #[batch, 256]
        loss = F.cross_entropy(Z_nat, target)

        #if epoch >=10:
        #    _par_images_opt = par_images.clone().detach().requires_grad_(True).to(device)
        #    loss_proto = args.beta*prox_criterion(L2_nat, target, _par_images_opt)
        #    loss += loss_proto
        #else:
        #    _par_images_opt = 0.0
        #    loss_proto = torch.tensor(0.0)


        if epoch >= args.warmup:

            if args.onehot:

                model.apply(set_bn_eval)

                _par_images_opt = par_images.clone().detach().requires_grad_(True).to(device)
                _par_images_opt_norm = transformDict['norm'](_par_images_opt)
                L2_proto, logits_proto = model(_par_images_opt_norm)

                loss_proto = args.beta*F.cross_entropy(logits_proto, targets_onehot)
                loss += loss_proto

                L2_proto_unit = F.normalize(L2_proto,dim=1)
                #L2_proto_unit = F.normalize(par_images.clone().detach())

                
                L2_proto_unit_sort, L2_proto_unit_sort_ind = torch.sort(L2_proto_unit, dim=1)

            else:
                _par_images_opt = par_images.clone().detach().requires_grad_(True).to(device)
                loss_proto = args.beta*prox_criterion(L2_nat, target, _par_images_opt)
                loss += loss_proto
                L2_proto_unit = F.normalize(_par_images_opt,dim=1)

                _par_images_detach = par_images.clone().detach().to(device)
                L2_proto_unit_detach = F.normalize(_par_images_detach,dim=1)

                L2_proto_unit_sort, L2_proto_unit_sort_ind = torch.sort(L2_proto_unit_detach, dim=1)



            #cov loss
            #latent_X_sorted = latent_X[:,class_proto_data_idx[lbl]].clone()
            #print (L2_proto_unit_sort_ind[target].shape)
            #torch.arange(3).unsqueeze(1)
            #with torch.no_grad():
            #    proto_sort_vals = L2_proto_unit_sort[target]
            #    proto_sort_inds = L2_proto_unit_sort_ind[target]
            #print ("proto sort inds shape ", proto_sort_inds.shape)
            #L2_nat_unit = F.normalize(L2_nat,dim=1)
            #L2_nat_sorted = L2_nat_unit[torch.arange(bs).unsqueeze(1),proto_sort_inds]
            #print (L2_nat_sorted.shape)

            if args.proto_norm:
                #L2_proto_unit_sort, L2_proto_unit_sort_ind = torch.sort(L2_proto_unit)
                proto_sort_vals = L2_proto_unit_sort[target]
                proto_sort_inds = L2_proto_unit_sort_ind[target]
                L2_nat_unit = F.normalize(L2_nat,dim=1)
                L2_nat_sorted = L2_nat_unit[torch.arange(bs).unsqueeze(1),proto_sort_inds]
                #L2_nat_diffs = L2_nat_sorted - proto_sort_vals
                L2_nat_diffs = proto_sort_vals*(L2_nat_sorted - proto_sort_vals)
            else:
                L2_proto_sort, L2_proto_sort_ind = torch.sort(_par_images_opt)
                proto_sort_vals = L2_proto_sort[target]
                proto_sort_inds = L2_proto_sort_ind[target]
                L2_nat_sorted = L2_nat[torch.arange(bs).unsqueeze(1),proto_sort_inds]
                #L2_nat_diffs = L2_nat_sorted - proto_sort_vals
                L2_nat_diffs = proto_sort_vals*(L2_nat_sorted - proto_sort_vals)

            
            #L2_nat_tpose = torch.transpose(L2_nat,0,1)    #ftrs, obs
            #L2_nat_cov = torch.triu(torch.cov(L2_nat_tpose), diagonal=1)
            #L2_nat_cov = 1.0 - torch.triu(torch.corrcoef(L2_nat_tpose),diagonal=1)
            
            #shift
            choice = np.random.randint(rmin_cur,rmax_cur + 1)
            L2_nat_sorted_shift_left = F.pad(L2_nat_diffs,(0,choice))
            L2_nat_sorted_shift_right = F.pad(L2_nat_diffs,(choice,0))

            if covsign == 0.0:
                loss_cov = zCov*torch.mean(torch.abs(L2_nat_sorted_shift_left*L2_nat_sorted_shift_right))
            else:
                if choice ==0:
                    loss_cov = zCov*torch.mean(L2_nat_sorted_shift_left*L2_nat_sorted_shift_right)
                else:
                    loss_cov = zCov*torch.mean(F.relu(covsign*(L2_nat_sorted_shift_left*L2_nat_sorted_shift_right)))
            #loss_cov = args.zetaCov*torch.mean(L2_nat_cov)
            loss += loss_cov

            #proto training
            #loss_proto_xent = 0.01*F.cross_entropy(logits_proto, proto_targ)
            #loss += loss_proto_xent

            CS_mat = L2_proto_unit @ L2_proto_unit.t()
            if args.cheb_cs:
                CS_mat_2 = CS_mat**2.0 #[nclass,nclass]
                loss_cs = zCS*torch.mean((2.0*CS_mat - CS_mat_2).masked_select(~torch.eye(nclass, dtype=bool, device=device)).view(nclass,nclass-1))
            elif args.sq_cs:
                CS_mat_2 = CS_mat**2.0
                loss_cs = zCS*torch.mean(CS_mat_2.masked_select(~torch.eye(nclass, dtype=bool, device=device)).view(nclass,nclass-1))
            else:
                loss_cs = zCS*torch.mean(CS_mat.masked_select(~torch.eye(nclass, dtype=bool, device=device)).view(nclass,nclass-1))
            loss += loss_cs

        loss.backward()


        if epoch >= args.warmup:
            if args.tweak:
                with torch.no_grad():
                    pert_vector = torch.sign(noise_batch.grad)
                    noise += 0.03*pert_vector.data
                    noise.clamp_(-0.03,0.03)

                    cur_loader.dataset.update_noise(noise, indexes)

                    noise_batch.grad.zero_()
            #with torch.no_grad():
            #    gradients_unscaled = _par_images_opt.grad
            #    grad_mag = gradients_unscaled.view(gradients_unscaled.shape[0], -1).norm(2, dim=-1)
            #    image_gradients = image_lr*gradients_unscaled  / (grad_mag.view(-1, 1, 1, 1) + 1.e-6)  
            
            #    par_images.add_(-image_gradients)
            #    par_images.clamp_(0.0,1.0)


            #    _par_images_opt.grad.zero_()
            with torch.no_grad():
                if kwargsUser['latent_proto']:
                    latent_gradients = 0.1*_par_images_opt.grad
                    par_images.add_(-latent_gradients)
                    par_images.clamp_(0.0,1e6)
                    _par_images_opt.grad.zero_()
                else:
                    image_gradients = args.par_grad_mult*_par_images_opt.grad
                    image_gradients.clamp_(-args.par_grad_clip,args.par_grad_clip)
                    #image_gradients =_par_images_opt.grad.clamp_(-0.002,0.002)                                                                                                                                                        
                    #print (torch.mean(image_gradients))                                                                                                                                                                               
                    #print ("image gradients are ", image_gradients)                                                                                                                                                                   
                    par_images.add_(-image_gradients)
                    par_images.clamp_(0.0,1.0)
                    _par_images_opt.grad.zero_()





        # if args.beta > 0.0 and epoch >=20:
        #     with torch.no_grad():
        #         if kwargsUser['latent_proto']:
        #             latent_gradients = _par_images_opt.grad
        #             par_images.add_(-latent_gradients)
        #             par_images.clamp_(0.0,1e6)
        #             _par_images_opt.grad.zero_()
        #         else:
        #             image_gradients = args.par_grad_mult*_par_images_opt.grad
        #             image_gradients.clamp_(-args.par_grad_clip,args.par_grad_clip)
        #             #image_gradients =_par_images_opt.grad.clamp_(-0.002,0.002)
        #             #print (torch.mean(image_gradients))
        #             #print ("image gradients are ", image_gradients)
        #             par_images.add_(-image_gradients)
        #             par_images.clamp_(0.0,1.0)
        #             _par_images_opt.grad.zero_()

        if (args.grad_clip):
            nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

        for o_num, o in enumerate(optimizer):
            o.step()
            # if not args.fc_freeze:
            #     o.step()
            # else:
            #     pass
            #     #compute fc orthogonality
            #     optimizer[o_num].step()


        if not args.fc_freeze:
            if epoch >= args.warmup and args.orthoreg:
                apply_orthoreg(model, lr=scheduler.get_last_lr()[0], beta=0.001, lambd=10., epsilon=1e-6)


        if args.anneal == "cyclic" or args.anneal == "cosine":
            if batch_idx < max_steps:
                for s in scheduler:
                    s.step()
                #scheduler.step()

        #loss_onehot = 0.0

        #_par_images_opt = par_images.clone().detach().requires_grad_(True).to(device)
        #_par_images_opt_norm = transformDict['norm'](_par_images_opt)
        #L2_proto, logits_proto = model(_par_images_opt_norm)
        #proto training
        #loss_onehot += F.cross_entropy(logits_proto, proto_targ)
        #L2_proto_unit = F.normalize(L2_proto)               #[nclass, numftr]
        #CS_mat = (L2_proto_unit @ L2_proto_unit.t())**2.0   #[nclass, nclass]
        #loss_onehot += args.zeta*torch.mean(CS_mat.masked_select(~torch.eye(nclass, dtype=bool, device=device)).view(nclass,nclass-1))

        #loss_onehot.backward()

        #with torch.no_grad():
        #    gradients_unscaled = _par_images_opt.grad
        #    grad_mag = gradients_unscaled.view(gradients_unscaled.shape[0], -1).norm(2, dim=-1)
        #    image_gradients = image_lr*gradients_unscaled  / (grad_mag.view(-1, 1, 1, 1) + 1.e-6)  
            
        #    par_images.add_(-image_gradients)
        #    par_images.clamp_(0.0,1.0)


        #    _par_images_opt.grad.zero_()

        #only update images, not network weights
        #optimizer.zero_grad()
        #model.zero_grad()
     
        # print progress
        if verb_cur:
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f} \t LossProto: {:.6f} \t LossCov: {:.6f} \t LossCS: {:.6f}'.format(
                    epoch, batch_idx * bs, len(cur_loader.dataset),
                       100. * batch_idx / len(cur_loader), loss.item(), loss_proto.item(), loss_cov.item(), loss_cs.item()))


def train(args, model, device, cur_loader, optimizer, epoch, par_images, scheduler=0.0, max_steps = 0, transformDict={}, **kwargs):

    model.train()
    model.multi_out=1
    print ('Training model')

    prox_criterion = Proximity(device,
                                num_classes=len(par_images),
                                num_ftrs=args.ftrcnt,
                                k=args.k,
                                protonorm=args.proto_norm)

    cur_decov = args.decov
    w_decov = args.w_decov
    cur_squentropy = args.squentropy
    cur_orthoreg = args.orthoreg
    cur_largemargin = args.largemarginCE
    verb_cur = args.verbose

    
    num_classes = len(par_images)

    #glob_k = 0
    #if (args.k != 0 and epoch > 0):
    #    glob_k = int(args.k * args.model_scale)

    #weight=torch.tensor(0.0)
    for batch_idx, (data, target) in enumerate(cur_loader):

        data, target = data.to(device), target.to(device)
        #print (batch_idx)
        #print (torch.min(data))
        #print ("mean ", MEAN)
        for o in optimizer:
            o.zero_grad()

        model.zero_grad()

        #channel normalize data
        data_norm = transformDict['norm'](data)

        #print (weight)

        # lets make par_servant useful again
        #if (args.par_servant):
        #    image_model.load_state_dict(model.state_dict())
        #    image_model.eval()

        #if args.beta > 0.0:
        #    _par_images_opt = par_images.clone().detach().requires_grad_(True).to(device)
        #else:
        #    _par_images_opt = 0.0

        if cur_largemargin:
            #Z_nat are modified with additional angular margin
            L2_nat, Z_nat = model(data_norm, target)
        else:
            L2_nat, Z_nat = model(data_norm)
        
        loss = F.cross_entropy(Z_nat, target)

        if args.beta > 0.0 and epoch >= args.warmup:
            _par_images_opt = par_images.clone().detach().requires_grad_(True).to(device)
            loss_proto = args.beta*prox_criterion(L2_nat, target, _par_images_opt)
            loss += loss_proto
        else:
            _par_images_opt = 0.0
            loss_proto = torch.tensor(0.0)

        if epoch >= args.warmup and cur_decov:
            L2_nat_tpose = torch.transpose(L2_nat,0,1)    #ftrs, obs
            cur_cov_mat_sq = torch.cov(L2_nat_tpose, correction=0)**2.0
            loss_proto = w_decov*(torch.sum(cur_cov_mat_sq) - torch.sum(torch.diagonal(cur_cov_mat_sq)))
            loss += loss_proto
            #L2_nat_cov = 1.0 - torch.triu(torch.corrcoef(L2_nat_tpose),diagonal=1)

        if epoch >= args.warmup and cur_squentropy:
            target_final = torch.zeros([target.size()[0], num_classes], device=device).scatter_(1, target.reshape(target.size()[0], 1), 1)
            #All logits squared - correct logit squared = incorrect logit squared contributions
            loss_proto = (torch.sum(Z_nat** 2)-torch.sum((Z_nat[target_final == 1]) ** 2))/(num_classes-1)/target_final.size()[0]
            loss += loss_proto



        loss.backward()

        if args.beta > 0.0 and epoch >= args.warmup:
            with torch.no_grad():
                if kwargsUser['latent_proto']:
                    latent_gradients = 0.1*_par_images_opt.grad
                    par_images.add_(-latent_gradients)
                    par_images.clamp_(0.0,1e6)
                    _par_images_opt.grad.zero_()
                else:
                    image_gradients = args.par_grad_mult*_par_images_opt.grad
                    image_gradients.clamp_(-args.par_grad_clip,args.par_grad_clip)
                    #image_gradients =_par_images_opt.grad.clamp_(-0.002,0.002)
                    #print (torch.mean(image_gradients))
                    #print ("image gradients are ", image_gradients)
                    par_images.add_(-image_gradients)
                    par_images.clamp_(0.0,1.0)
                    _par_images_opt.grad.zero_()

        if (args.grad_clip):
            nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

        for o_num, o in enumerate(optimizer):
            o.step()
            #Do the checking once per epoch (outside of training loop)
            # if not args.fc_freeze:
            #     o.step()
            # else:
            #     pass
            #     #compute fc orthogonality
            #     optimizer[o_num].step()

        if epoch >= args.warmup and cur_orthoreg:
            apply_orthoreg(model, lr=scheduler[0].get_last_lr()[0], beta=0.001, lambd=10., epsilon=1e-6)

        if args.anneal == "cyclic" or args.anneal == "cosine":
            if batch_idx < max_steps:
                for s in scheduler:
                    s.step()
                #scheduler.step()
                
        # print progress
        if verb_cur:
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLossProto: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(cur_loader.dataset),
                       100. * batch_idx / len(cur_loader), loss.item(), loss_proto.item()))



def eval_train(args, model, device, train_loader, transformDict):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
        #for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = transformDict['norm'](data)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    if args.verbose:
        print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(args, model, device, test_loader, transformDict):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
        #for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = transformDict['norm'](data)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    if args.verbose:
        print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= (0.5*args.epochs):
        lr = args.lr * 0.1
    if epoch >= (0.75*args.epochs):
        lr = args.lr * 0.01
    if epoch >= (0.9*args.epochs):
        lr = args.lr * 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # setup data loader
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    MEAN = [0.5]*3
    STD = [0.5]*3

    
    if (args.dataset == "CIFAR10"):
        if args.channel_norm:
            MEAN = [0.4914, 0.4822, 0.4465]
            STD = [0.2471, 0.2435, 0.2616] 
    elif(args.dataset == "CIFAR100"):
        if args.channel_norm:
            MEAN = [0.5071, 0.4865, 0.4409]
            STD = [0.2673, 0.2564, 0.2762]
    elif  (args.dataset == "IN100"):
        if args.channel_norm:
            MEAN = [0.485, 0.456, 0.406]
            STD  = [0.229, 0.224, 0.225]

    elif (args.dataset == "TINYIN"):
        if args.channel_norm:
            MEAN = [0.4802, 0.4481, 0.3975]
            STD  = [0.2302, 0.2265, 0.2262]

    elif (args.dataset == "FASHION"):
        MEAN = [0.5]
        STD = [0.5]
    else:
        print ("ERROR dataset not found")

    gen_transform_train = transforms.Compose([transforms.ToTensor()])
    #gen_transform_test = transforms.Compose([transforms.ToTensor()])

    #first augmentation in pipeline gets [Tensor, Flip, Crop] by default
    if args.dataset in ["CIFAR10","CIFAR100"]:
        if args.flipcrop:
            train_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4)])
        else:
            train_transform = transforms.Compose(
                [transforms.ToTensor()])


        train_transform_tensor = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])

        gen_transform_test = transforms.Compose(
            [transforms.ToTensor()])

    elif args.dataset in ["STL10"]:

        if args.flipcrop:
            train_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(96, padding=4)])
        else:
            train_transform = transforms.Compose(
                [transforms.ToTensor()])


        train_transform_tensor = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96, padding=4)])

        gen_transform_test = transforms.Compose(
            [transforms.ToTensor()])


    elif args.dataset in ["FASHION"]:

        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Pad(2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])

        train_transform_tensor = transforms.Compose(
            [transforms.Pad(2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])

        gen_transform_test = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Pad(2)])
        
    elif args.dataset in ["TINYIN"]:
        if args.flipcrop:
            train_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(64, padding=4)])
        else:
            train_transform = transforms.Compose(
                [transforms.ToTensor()])

        train_transform_tensor = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4)])
        gen_transform_test = transforms.Compose(
            [transforms.ToTensor()])

    elif args.dataset in ["IN100", "PETS", "CALTECH100", "CALTECH256"]:
        #train_transform = transforms.Compose(
        #    [transforms.ToTensor(),
        #    transforms.RandomResizedCrop(224, antialias=True),
        #     transforms.RandomHorizontalFlip()])

        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(256, antialias=True),
             transforms.CenterCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(224, padding=4)]) 

        train_transform_tensor = transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip()])
        gen_transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(256, antialias=True),
             transforms.CenterCrop(224)])
    else:
        print ("ERROR setting transforms")


    #comp_list_test = [transforms.ToTensor()]
    
    if (args.dataset == "CIFAR10"):

        #trainset_basic = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=gen_transform_train)
        #train_loader_basic = torch.utils.data.DataLoader(trainset_basic, batch_size=args.batch_size, shuffle=True, **kwargs)

        #both augmix and PRIME want [crop, flip] before their augmentations
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)


        #cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)


        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=gen_transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        #num_classes = 10
        kwargsUser['num_classes'] = 10
        nclass=10
        nchannels = 3
        H, W = 32, 32
        targs_ds = trainset.targets
            
    elif (args.dataset == "CIFAR100"):

        #trainset_basic = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=gen_transform_train)
        #train_loader_basic = torch.utils.data.DataLoader(trainset_basic, batch_size=args.batch_size, shuffle=True, **kwargs)


        #trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
        trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        #cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=gen_transform_test)
        #test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        #num_classes = 100
        kwargsUser['num_classes'] = 100
        nclass=100
        nchannels = 3
        H, W = 32, 32
        targs_ds = trainset.targets

    elif (args.dataset == "STL10"):

        #trainset_basic = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=gen_transform_train)
        #train_loader_basic = torch.utils.data.DataLoader(trainset_basic, batch_size=args.batch_size, shuffle=True, **kwargs)


        #trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
        trainset = torchvision.datasets.STL10(root='../data', split='train', download=True, transform=train_transform)
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        #cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.STL10(root='../data', split='test', download=True, transform=gen_transform_test)
        #test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        #num_classes = 100
        kwargsUser['num_classes'] = 10
        nclass=10
        nchannels = 3
        H, W = 96, 96
        targs_ds = trainset.labels

    elif (args.dataset == "CALTECH256"):
        alldata = torchvision.datasets.Caltech256(root='../data', download=True)
        kwargsUser['num_classes'] = 256
        nclass=256
        nchannels=3
        H,W = 224, 224
        # divide the data into train, validation, and test set                                                                                                                                                                        
        x_base, x_test , y_base, y_test = train_test_split(alldata.index, alldata.y, test_size=0.2, stratify=alldata.y,random_state=42)
        #(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.25, random_state=42)                                                                                                                                 
        #torch.utils.data.Subset                                                                                                                                                                                                      
        trainset = torch.utils.data.Subset(alldata, x_base)
        #val_sub = torch.utils.data.Subset(alldata, x_val)                                                                                                                                                                            
        testset = torch.utils.data.Subset(alldata, x_test)

    elif (args.dataset == "PETS"):
        trainset = torchvision.datasets.OxfordIIITPet(root='../data', split='trainval', download=True, transform=train_transform)
        testset = torchvision.datasets.OxfordIIITPet(root='../data', split='test', download=True, transform=gen_transform_test)
        kwargsUser['num_classes'] = 37
        nclass=37
        nchannels=3
        H,W = 224, 224
        targs_ds = trainset._labels

    elif (args.dataset == "FASHION"):

        #trainset_basic = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=gen_transform_train)
        #train_loader_basic = torch.utils.data.DataLoader(trainset_basic, batch_size=args.batch_size, shuffle=True, **kwargs)


        #trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
        trainset = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=train_transform)
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        #cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=gen_transform_test)
        #test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        #num_classes = 100
        kwargsUser['num_classes'] = 10
        nclass=10
        nchannels = 1
        H, W = 32, 32

    elif (args.dataset == "IN100"):
        
        #trainset_basic = datasets.ImageFolder(
        #    './Data_ImageNet/train_100',
        #    transform=gen_transform_train)
        #train_loader_basic = torch.utils.data.DataLoader(trainset_basic, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)


        trainset = datasets.ImageFolder(
            './Data_ImageNet/train_100',
            transform=train_transform)
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        #cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        testset = datasets.ImageFolder(
            './Data_ImageNet/val_100',
            transform=gen_transform_test)
        #test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        kwargsUser['num_classes'] = 100
        nclass = 100
        nchannels = 3
        H, W = 224, 224
    
    elif (args.dataset == "TINYIN"):
        
        #trainset_basic = datasets.ImageFolder(
        #    './Data_ImageNet/train_100',
        #    transform=gen_transform_train)
        #train_loader_basic = torch.utils.data.DataLoader(trainset_basic, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)


        trainset = datasets.ImageFolder(
            './tiny-imagenet-200/train',
            transform=train_transform)
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        #cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        testset = datasets.ImageFolder(
            './tiny-imagenet-200/val/images',
            transform=gen_transform_test)
        #test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        kwargsUser['num_classes'] = 200
        nclass = 200
        nchannels = 3
        H, W = 64, 64
        targs_ds = trainset.targets

    else:
          
        print ("Error getting dataset")


    transformDict = {}

    transformDict['basic'] = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(H, padding=4),transforms.Normalize(MEAN, STD)])
    transformDict['flipcrop'] = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(H, padding=4)])
    transformDict['norm'] = transforms.Compose([transforms.Normalize(MEAN, STD)])
    transformDict['mean'] = MEAN
    transformDict['std'] = STD

    # if kwargsUser['augmix']:
    #     transformDict['aug'] = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32,padding=4), transforms.AugMix()])

    protolist = []
    proto_no_norm = []

    if kwargsUser['proto_aug']:
        if "crop" in kwargsUser['proto_aug']:
            protolist.append(transforms.RandomCrop(H, padding=4))
            proto_no_norm.append(transforms.RandomCrop(H, padding=4))
        if "flip" in kwargsUser['proto_aug']:
            protolist.append(transforms.RandomHorizontalFlip(p=0.5))
            proto_no_norm.append(transforms.RandomHorizontalFlip(p=0.5))
        if "invert" in kwargsUser['proto_aug']:
            protolist.append(transforms.RandomInvert(p=0.5))
            proto_no_norm.append(transforms.RandomInvert(p=0.5))

    protolist.append(transforms.Normalize(MEAN, STD))
    #print ("transform protos")
    transformDict['proto'] = transforms.Compose(protolist)
    transformDict['proto_no_norm'] = transforms.Compose(proto_no_norm)




    #data_separator = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=False, num_workers=4, pin_memory=True)
    #xdata = []
    #ydata = []

    print (len(trainset))




    #dataset=ConcatDataset([trainset,testset])
    #baseline set will have the transform in it
    #baseline_set = CustomDataSetWeighted(xdata, ydata, active_weights=1, transform=train_transform_tensor)
    #baseline_set = CustomDataSetWeightedDS(trainset, active_weights=1, transform= train_transform_tensor)
    #baseline_test_set = CustomDataSetWeighted(xdata, ydata, active_weights=0, transform=[])
    #baseline_test_set = CustomDataSetWeightedTest(trainset, transform = [])

    #full_test_set = CustomDataSetWeightedTest(testset)

    #skf = StratifiedKFold(n_splits=5, shuffle=True)

    #w_tensors = torch.ones(size=(len(dataset)), dtype=torch.float)
    
    splits = []

    # if args.dataset in ["STL10"]:
    #     all_inds = np.arange(len(trainset.labels))
    #     print (len(all_inds))
    # else:
    #     all_inds = np.arange(len(trainset.targets))

    all_inds = np.arange(len(targs_ds))

    split_labels = []

    for d in args.datasplits:

        cur_data_seed = 0

        for _ in range(args.modelinitials):

            if args.semi:
                cur_data_seed += 2
            else:
                cur_data_seed = args.seed
                
            if d < 1.0:
                # if args.dataset in ["STL10"]:
                #     inds_train1, inds_test1, y_train1, y_test1 = train_test_split(all_inds, trainset.labels, test_size=d, random_state=cur_data_seed, stratify=trainset.labels)
                # else:
                #     inds_train1, inds_test1, y_train1, y_test1 = train_test_split(all_inds, trainset.targets, test_size=d, random_state=cur_data_seed, stratify=trainset.targets)

                inds_train1, inds_test1, y_train1, y_test1 = train_test_split(all_inds, targs_ds, test_size=d, random_state=cur_data_seed, stratify=targs_ds)
                splits.append(inds_test1)
                split_labels.append(d)
            else:
                splits.append(all_inds)
                split_labels.append(1.0)

    #inds_train2, inds_test2, y_train2, y_test2 = train_test_split(all_inds, trainset.targets, test_size=0.4, random_state=args.seed, stratify=trainset.targets)

    #splits.append(inds_test2)

    #inds_train3, inds_test3, y_train3, y_test3 = train_test_split(all_inds, trainset.targets, test_size=0.6, random_state=args.seed, stratify=trainset.targets)

    #splits.append(inds_test3)

    #inds_train4, inds_test4, y_train4, y_test4 = train_test_split(all_inds, trainset.targets, test_size=0.7, random_state=args.seed, stratify=trainset.targets)

    #splits.append(inds_test4)


    #inds_train5, inds_test5, y_train5, y_test5 = train_test_split(all_inds, trainset.targets, test_size=0.8, random_state=args.seed, stratify=trainset.targets)

    #splits.append(inds_test5)

    #inds_train6, inds_test6, y_train6, y_test6 = train_test_split(all_inds, trainset.targets, test_size=0.9, random_state=args.seed, stratify=trainset.targets)

    #splits.append(inds_test6)


    #add 100% training
    #splits.append(all_inds)

    if (not args.continuous):
        lr_sched = [args.lr]*len(splits)
    else:
        lr_sched = [args.lr_fine*args.lr]*len(splits)
        lr_sched[0] = args.lr




    #splits = skf.split(baseline_set.Ydata,baseline_set.Ydata)    #list of tuples

    #for fold,(train_idx,test_idx) in enumerate(skf.split(baseline_set.Ydata,baseline_set.Ydata)):

    train_accs = []
    test_accs = []
    #cossim_means = []
    #cossim_sq_means = []
    #dsplits = []

    #loss_par_final_list = []

    # ave_ave_var_train_list = []
    # ave_max_var_train_list = []
    # ave_ave_var_test_list = []
    # ave_max_var_test_list = []
    # ave_ratio_train_list = []
    # ave_ratio_test_list = []
    # maxmean_ratio_train_list = []
    # maxmean_ratio_test_list = []

    # #these are for computing subset feature covariance matrices for each class, not the percentage of training dataset
    # ftr_pct = [0.35, 0.2, 0.1, 0.05, 0.02]

    # mean_class_cov_list = []
    # mean_class_cov_unit_list = []
    # mean_class_corr_list = []
    # mean_dissimsq_offdiag_list = []
    # chebyshev_list = []

    # for _ in range(len(ftr_pct)):
    #     mean_class_cov_list.append([])
    #     mean_class_cov_unit_list.append([])
    #     mean_class_corr_list.append([])

    # par_image_tensors = []

    # par_image_encodings = []
    # par_image_encodings.append(torch.arange(nclass, dtype=torch.long, device=device))
    # #targets = torch.arange(len(par_images), dtype=torch.long, device=device)
    # min_encoding = torch.zeros( [kwargsUser['num_classes'],kwargsUser['num_classes']], dtype=torch.float, device=device)
    # for u in range(nclass):
    #     for t in range(nclass):
    #         if u==t:
    #             min_encoding[u,t] = (1.5/nclass)
    #         else:
    #             min_encoding[u,t] = (1.0 - (1.5/nclass)) / (nclass-1)

    # par_image_encodings.append(min_encoding.clone())

    for j in range(len(args.datasplits)):
        for m in range(args.modelinitials):

            #once I start doing multiple initializations per datasplit, this will need to be modified
            #and will need to enter [0.5]*12 for the datasplits, and 3 for model initializations
            cur_split = j*args.modelinitials + m

            with open('{}/train_hist.txt'.format(model_dir), 'a') as f:
                f.write("\n")
                f.write("Training: split{},initial{} ".format(cur_split,m))
                f.write("\n")
            f.close()

            with open('{}/assess_proto_hist_{}.txt'.format(full_dir_plot,args.k), 'a') as f:
                f.write("\n")
                f.write("Training: split{},initial{} ".format(cur_split,m))
                f.write("\n")
            f.close()

            #correspondence issues between folds
            subtrain = torch.utils.data.Subset(trainset, splits[cur_split])
            #subtest = torch.utils.data.Subset(test, test_idx)
            print (len(subtrain))

            if args.tweak:
                #CustomDataSetWNoise(Dataset)

                cur_ds = CustomDataSetWNoise(subtrain)
            else:
                cur_ds = subtrain


            print('------------training no---------{}----------------------'.format(m))

            eval_train_loader = torch.utils.data.DataLoader(subtrain, batch_size=args.batch_size, shuffle=False,**kwargs)
            cur_loader = torch.utils.data.DataLoader(cur_ds, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
            

            #set beta=0.0 if you dont want to train with online prototypes
            #make par images
            if (j==0 or (not args.continuous)):
                with torch.no_grad():

                    #par_image_list = []

                    if kwargsUser['latent_proto']:

                        if args.proto_layer==4:
                            par_images_glob = torch.rand( [kwargsUser['num_classes'],args.ftrcnt], dtype=torch.float, device=device)
                        else:
                            par_images_glob = torch.rand( [kwargsUser['num_classes'],args.ftrcnt//2], dtype=torch.float, device=device)

                        if kwargsUser['proto_norm']:
                            par_images_glob = F.normalize(par_images_glob)

                    else:
                        par_images_glob = torch.rand([kwargsUser['num_classes'],nchannels,H,W],dtype=torch.float, device=device)

                        par_images_glob.clamp_(0.0,1.0)
            
                # init model, ResNet18() can be also used here for training
                if (args.model == "ResNet18"):
                    if args.dataset in ["CIFAR10","CIFAR100", "FASHION","STL10","PETS", "TINYIN"]:
                        if args.largemarginCE:
                            model = ResNet18Large(nclass = nclass, scale=args.model_scale , channels=nchannels, device=device, **kwargsUser).to(device)
                        else:
                            model = ResNet18(nclass = nclass, scale=args.model_scale , channels=nchannels, **kwargsUser).to(device)
                        #image_model = ResNet18(nclass = nclass,**kwargsUser).to(device)
                    elif args.dataset in ["IN100"]:
                        model = ResNet18IN(nclass=nclass, scale=args.model_scale , **kwargsUser).to(device)
                        #image_model = ResNet18IN(nclass = nclass,**kwargsUser).to(device)
                    else:
                        print ("Error matching model to dataset")
                        
                elif (args.model == "ResNet34"):
                    if args.dataset in ["CIFAR10","CIFAR100", "FASHION","STL10","PETS","TINYIN"]:
                        if args.largemarginCE:
                            model = ResNet34Large(nclass = nclass, scale=args.model_scale , channels=nchannels, device=device, **kwargsUser).to(device)
                        else:
                            model = ResNet34(nclass = nclass, scale=args.model_scale , channels=nchannels, **kwargsUser).to(device)
                        #image_model = ResNet18(nclass = nclass,**kwargsUser).to(device)                                                                                                                            
                    elif args.dataset in ["IN100"]:
                        model = ResNet34(nclass=nclass, scale=args.model_scale , **kwargsUser).to(device)
                        #image_model = ResNet18IN(nclass = nclass,**kwargsUser).to(device)                                                                                                                                            
                    else:
                        print ("Error matching model to dataset")
                        
                elif (args.model == "SmallNet"):
                    model = SmallNet(nclass = nclass, scale=args.model_scale, channels = nchannels, drop=0, **kwargsUser).to(device)
                elif (args.model == "LogNetBaseline"):
                    model = LogNetBaseline(nclass=nclass, scale=args.model_scale, channels= nchannels, **kwargsUser).to(device)
                elif (args.model == "PreActResNet18"):
                    model = PreActResNet18(nclass = nclass,**kwargsUser).to(device)
                    #image_model = PreActResNet18(nclass = nclass,**kwargsUser).to(device)
                elif (args.model == "WRN16_2"):
                    model = WRN16_2(nclass=nclass,**kwargsUser).to(device)
                    #image_model = WRN16_2(nclass = nclass,**kwargsUser).to(device)
                elif (args.model == "WRN16_4"):
                    model = WRN16_4(nclass = nclass,**kwargsUser).to(device)
                    #image_model = WRN16_4(nclass = nclass,**kwargsUser).to(device)
                elif (args.model == "DenseNet"):
                    model = DenseNetCifar(nclass = nclass, scale=args.model_scale , channels=nchannels, **kwargsUser).to(device)
                    #image_model = densenet(num_classes = kwargsUser['num_classes']).to(device)
                elif (args.model == "ResNeXt"):
                    model = resnext29(num_classes = kwargsUser['num_classes']).to(device)
                    #image_model = resnext29(num_classes = kwargsUser['num_classes']).to(device)
                elif (args.model == "AllConv"):
                    model = AllConvNet(num_classes = kwargsUser['num_classes']).to(device)
                    #image_model = AllConvNet(num_classes = kwargsUser['num_classes']).to(device)
                else:
                    print ("Invalid model architecture")

                #summary(model, (3, 32, 32))

            #model.multi_out=0
            #model.eval()
            #ini_loss_train, ini_acc_train = eval_train(args, model, device, cur_loader, transformDict)
            #ini_loss_test, ini_acc_test = eval_test(args, model, device, test_loader, transformDict)

            #with open('{}/train_hist.txt'.format(model_dir), 'a') as f:
            #    f.write("=====INITIAL======\n")
            #    f.write("{0:4.3f}\t{1:4.3f}\t{2:4.0f}\t{3:4.3f}\t{4:6.5f}\n".format(ini_acc_train,ini_acc_test,len(cur_loader.dataset),ini_loss_train,ini_loss_test))
            #    f.write("==================\n")
            #f.close()


            model.train()
            model.multi_out = 1
            for p in model.parameters():
                p.requires_grad = True


            if args.anneal in ["stairstep", "cosine"]:
                lr_i = lr_sched[cur_split]
            elif args.anneal in ["cyclic"]:
                lr_i = 0.2
            else:
                print ("Error setting learning rate")

            print (lr_i)
            #if (j==0 or (not args.continuous)):
            optimizer= 0.0

            if args.fc_freeze:
                opt_conv = optim.SGD(model.params_conv.parameters(), lr=lr_i, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
                opt_fc = optim.SGD(model.params_fc.parameters(), lr=lr_i, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
                optimizer = [opt_conv, opt_fc]
            else:
                optimizer = [optim.SGD(model.parameters(), lr=lr_i, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)]

            scheduler = 0.0
            steps_per_epoch = int(np.ceil(len(cur_loader.dataset) / args.batch_size))

            print ("len(cur_loader.dataset)", len(cur_loader.dataset))
            print ("len(cur_loader)", len(cur_loader))

            if args.anneal == "stairstep":
                pass
            elif args.anneal == "cosine":
                if args.fc_freeze:
                    sched_conv = optim.lr_scheduler.CosineAnnealingLR(optimizer[0], T_max=args.epochs*len(cur_loader), eta_min=0.0000001, last_epoch=-1, verbose=False)
                    sched_fc = optim.lr_scheduler.CosineAnnealingLR(optimizer[1], T_max=args.epochs*len(cur_loader), eta_min=0.0000001, last_epoch=-1, verbose=False)
                    scheduler = [sched_conv,sched_fc]
                else:
                    scheduler = [optim.lr_scheduler.CosineAnnealingLR(optimizer[0], T_max=args.epochs*len(cur_loader), eta_min=0.0000001, last_epoch=-1, verbose=False)]
            elif args.anneal == "cyclic":
                pct_start = 0.25
                #steps_per_epoch = 391   #50k / 128
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr_i, epochs = args.epochs, steps_per_epoch = steps_per_epoch, pct_start = pct_start)
            else:
                print ("ERROR making scheduler") 

            with torch.no_grad():
                targets_onehot = torch.arange(nclass, dtype=torch.long, device=device)  

            for epoch in range(1, args.epochs + 1):
                if args.largemarginCE:
                    model.linear.training = True
                # adjust learning rate for SGD
                if args.anneal == "stairstep":
                    adjust_learning_rate(optimizer, epoch)

                model.train()
                model.multi_out = 1
                par_images_glob_ref = par_images_glob.clone().detach()
                #with open('{}/lr_hist.txt'.format(model_dir), 'a') as f:
                #    f.write("{0:3.8f}".format(scheduler.get_last_lr()[0]))
                #    f.write("\n")
                #f.close()

                #print (scheduler.get_last_lr()[0])
                # proximity training
                if args.train_cheb:
                    train_cheb(args, model, device, cur_loader, optimizer, epoch, par_images_glob, proto_targ=targets_onehot, scheduler=scheduler, max_steps = steps_per_epoch, transformDict=transformDict, **kwargsUser)
                else:
                    train(args, model, device, cur_loader, optimizer, epoch, par_images_glob, scheduler=scheduler, max_steps = steps_per_epoch, transformDict=transformDict, **kwargsUser)

                #print (torch.mean(model.linear.weight))


                if args.fc_freeze:
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if 'linear' in name and 'weight' in name:
                                print (name)           
                                #print (param.shape) #[nclass,512]                                                                                                                                                               
                                #print (param[0].shape) #[512]  512 weights from pattern to C0                                                                                                                                  
                                #print (torch.min(param[0]))                                                                                                                                                                          
                                #print (torch.max(param[0]))
                                print (param.shape)
                                fc_weights = param.clone().cpu()  #[nclass, 512]
                                #fc_weights_t = param.clone().t()  #[512,nclass]

                        fc_weights_unit = F.normalize(fc_weights, dim=1)
                        #all_pairs_weight_dot = fc_weights @ fc_weights.t()
                        all_pairs_weight_CS = fc_weights_unit @ fc_weights_unit.t()
                        weight_count = len(all_pairs_weight_CS)

                        mean_cs = torch.mean(all_pairs_weight_CS)
                        #mean_cs = torch.mean(all_pairs_weight_CS.masked_select(~torch.eye(weight_count, dtype=bool)).view(weight_count,weight_count-1))
                        print ("Mean FC Weight CS is", mean_cs.item())

                        #stop optimizing the fully connected layer
                        if mean_cs < args.fc_limit:
                            optimizer = [opt_conv]
                            scheduler = [sched_conv]



                

                model.multi_out = 0
                if args.largemarginCE:
                    model.linear.training = False
                #par_images_glob.data = par_update.data
                
                with torch.no_grad():
                    par_change = torch.mean(torch.linalg.norm((par_images_glob - par_images_glob_ref).view(par_images_glob.shape[0],-1),2, dim=1))

                # evaluation on natural examples

                
                #print('================================================================')
                if (epoch % 20) == 0:
                    print ('===============================================================')
                    loss_train, acc_train = eval_train(args, model, device, eval_train_loader, transformDict)
                    loss_test, acc_test = eval_test(args, model, device, test_loader, transformDict)
                    print ("parametric images mean {0:4.3f}".format(torch.mean(par_images_glob).item()))
                    print ("parametric images change {0:4.8f}".format(par_change.item()))
                    print('================================================================')

                
                    with open('{}/train_hist.txt'.format(model_dir), 'a') as f:
                        f.write("{0:4.3f}\t{1:4.3f}\t{2:4.0f}\t{3:4.3f}\t{4:6.5f}\t{5:6.5f}\n".format(acc_train,acc_test,len(cur_loader.dataset),par_change.item(),loss_train,loss_test))
                    f.close()

                #or loss_train < 0.01 USED FOR EXPERIMENT 1 ACROSS DATA

                if (epoch == args.epochs):
                    torch.save(model.state_dict(),os.path.join(model_dir, 'model-{}-split{}_init{}.pt'.format(network_string,split_labels[cur_split],m)))
                    #torch.save(par_images_glob, os.path.join(model_dir,'prototypes_online_lyr_{}_pool_{}_epoch{}_training{}{}.pt'.format(args.proto_layer,args.proto_pool,epoch,m,m)))
                    break

                #the following commented out code was used for assessment thats already taken care of in cheb_assess.py and full_assess.py


    #         model.multi_out = 1
    #         par_images_random = torch.rand([kwargsUser['num_classes'],nchannels,H,W],dtype=torch.float, device=device)
    #         loss_derived, par_images_derived = train_image_data(args, model, device, par_images_random, cur_loader, iterations=1, mask=0, transformDict=transformDict,targets=-1,**kwargsUser)

    #         model.multi_out = 0
    #         with torch.no_grad():
    #             par_images_final = par_images_derived.clone()
    #             par_images_final_norm = transformDict['norm'](par_images_final)
    #             logits_par_final = model(par_images_final_norm)
    #             loss_par_final = F.cross_entropy(logits_par_final, targets_onehot)
    #             loss_par_final_list.append(loss_par_final)


    #         model.multi_out = 1

    #         train_accs.append(acc_train)
    #         test_accs.append(acc_test)
    #         dsplits.append(split_labels[cur_split])

    #         ave_ave_var_train = 0.0
    #         ave_max_var_train = 0.0
    #         ave_ave_var_test = 0.0
    #         ave_max_var_test = 0.0

    #         ave_ratio_train = 0.0
    #         ave_ratio_test = 0.0
    #         maxmean_ratio_train = 0.0
    #         maxmean_ratio_test = 0.0
            
    #         ave_ave_var_train, ave_max_var_train, ave_ratio_train, maxmean_ratio_train = compute_variance(device, model, cur_loader, nclass, transformDict, norm=args.proto_norm)
    #         ave_ave_var_test, ave_max_var_test, ave_ratio_test, maxmean_ratio_test = compute_variance(device, model, test_loader, nclass, transformDict, norm=args.proto_norm)

    #         ave_ave_var_train_list.append(ave_ave_var_train.clone())
    #         ave_max_var_train_list.append(ave_max_var_train.clone())
    #         ave_ave_var_test_list.append(ave_ave_var_test.clone())
    #         ave_max_var_test_list.append(ave_max_var_test.clone())

    #         ave_ratio_train_list.append(ave_ratio_train.clone())
    #         ave_ratio_test_list.append(ave_ratio_test.clone())
    #         maxmean_ratio_train_list.append(maxmean_ratio_train.clone())
    #         maxmean_ratio_test_list.append(maxmean_ratio_test.clone())
            
    #         pnt_CS_mats = []
    #         pnt_l2_mats = []

    #         for run in range(15):
    #             print ("computing prototypes")

    #             par_images_post = torch.rand([nclass,3,H,W],dtype=torch.float, device=device)
    #             last_loss = train_image_nodata(args, model, device, par_images_post, iterations=2, transformDict=transformDict, targets=targets_onehot, **kwargsUser)
    #             par_images_post = par_images_post.cpu()
    #             #print (last_loss)

    #             with torch.no_grad():
    #                 _par_images_final = par_images_post.clone().detach().requires_grad_(False).to(device)
    #                 _par_images_final_norm = transformDict['norm'](_par_images_final)
    #                 L2_img, logits_img = model(_par_images_final_norm)
    #                 #pnt_prototype_latents.append(L2_img.clone())
    #                 #pred = logits_img.max(1, keepdim=True)[1]
    #                 #probs = F.softmax(logits_img)
    #                 #print (torch.max(logits_img,dim=1)[1])

    #             l2_mat_latent_temp = all_pairs_L2(L2_img.clone())
    #             cos_mat_latent_temp = torch.zeros(nclass,nclass, dtype=torch.float)
    #             cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
    #             #latent_proto_all = torch.zeros([nclass,num_ftrs], dtype=torch.float)
    #             #print (nclass)
    #             for i in range(nclass):
    #                 for q in range(nclass):
    #                     if i != q:
    #                         #print (L2_img[i].shape)
    #                         cos_mat_latent_temp[i,q] = cos_sim(L2_img[i].view(-1), L2_img[q].view(-1)).clone().cpu()

    #             pnt_CS_mats.append(cos_mat_latent_temp.clone())
    #             pnt_l2_mats.append(l2_mat_latent_temp.clone())


    #         #compute mean and stddev of metrics over all derived prototypes
    #         pnt_CS_mats = torch.stack(pnt_CS_mats,dim=0)

    #         #mean latent cosine similarity matrix over 15 runs
    #         cos_mat_latent = torch.mean(pnt_CS_mats,dim=0)

    #         pnt_l2_mats = torch.stack(pnt_l2_mats, dim=0)
    #         l2_mat_latent = torch.mean(pnt_l2_mats, dim=0)

    #         #ftr_pct = [0.35, 0.2, 0.1, 0.05, 0.02]

    #         datalist = estimate_chebyshev(args, 
    #                                         model, 
    #                                         device, 
    #                                         ftr_pct, 
    #                                         par_images_post, 
    #                                         cos_mat_latent,
    #                                         l2_mat_latent, 
    #                                         transformDict, 
    #                                         nclass,
    #                                         HW=H)

    #         # return 
    #         # 0[class_covs, 
    #         # 1class_corrs, 
    #         # 2class_covs_unit, 
    #         # 3uni_Mg_cov_list, 
    #         # 4simp_Mg_cov_list, 
    #         # 5uni_Mg_corr_list, 
    #         # 6simp_Mg_corr_list, 
    #         # 7outside_class_covs, 
    #         # 8outside_class_corrs, 
    #         # 9dissim_off,
    #         # 10disL2_off, 
    #         # 11risk_cov_l2_list,
    #         # 12risk_l2_cov_list,
    #         # 13risk_covunit_cs_list,
    #         # 14risk_cs_covunit_list,
    #         # 15risk_covabs_l2_list]

    #         #class_cov[f] is single means for each class, need to be stacked and averaged for each f if desired
    #         for f in range(len(ftr_pct)):
    #             mean_class_cov_list[f].append(torch.mean(torch.stack(datalist[0][f],dim=0)))
    #             mean_class_cov_unit_list[f].append(torch.mean(torch.stack(datalist[2][f],dim=0)))
    #             mean_class_corr_list[f].append(torch.mean(torch.stack(datalist[1][f],dim=0)))
                
    #         mean_dissimsq_offdiag = datalist[10]
    #         if args.proto_norm:
    #             chebyshev = datalist[13][1]
    #         else:
    #             chebyshev = datalist[15][1]

    #         #mean_class_cov_list.append(mean_class_cov.clone())
    #         #mean_class_cov_unit_list.append(mean_class_cov_unit.clone())
    #         #mean_class_corr_list.append(mean_class_corr.clone())
    #         mean_dissimsq_offdiag_list.append(mean_dissimsq_offdiag.clone())
    #         chebyshev_list.append(chebyshev.clone())





    # with open('{}/assess_proto_summary_{}.txt'.format(full_dir_plot,args.k), 'a') as f:
    #     f.write("Test Acc \t AveAveVarTrain \t AveMaxVarTrain \t AveAveVarTest \t AveMaxVarTest \t MeanClassFtrCovSum \t MeanClassUnitFtrCovSum \t MeanClassFtrCorr \t MeanDissimSqOffDiag \t SumChebyshevOffDiag \n")
    #     for i in range(len(test_accs)):
    #         f.write("{0:4.4f} \t".format(test_accs[i]))
    #         f.write("{0:4.8f} \t".format(ave_ave_var_train_list[i]))
    #         f.write("{0:4.8f} \t".format(ave_max_var_train_list[i]))
    #         f.write("{0:4.8f} \t".format(ave_ratio_train_list[i]))
    #         f.write("{0:4.8f} \t".format(ave_ave_var_test_list[i]))
    #         f.write("{0:4.8f} \t".format(ave_max_var_test_list[i]))
    #         f.write("{0:4.8f} \t".format(ave_ratio_test_list[i]))
            
    #         f.write("{0:4.8f} \t".format(loss_par_final_list[i]))

    #         #f.write("{0:4.4f} \t".format(mean_class_cov_list[i]))
    #         #f.write("{0:4.8f} \t".format(mean_class_cov_unit_list[i]))
    #         #f.write("{0:4.8f} \t".format(mean_class_corr_list[i]))
    #         f.write("{0:4.8f} \t".format(mean_dissimsq_offdiag_list[i]))
    #         f.write("{0:4.8f} \n".format(chebyshev_list[i]))

    # f.close()


    # if not os.path.exists('feature_control_global_2.txt'):
    #     with open('feature_control_global_2.txt', 'a') as f:
    #         f.write("Mod \t Scale \t Set \t WD \t Aug \t Beta \t ZetaCov \t ZetaCS \t ProtoLayer \t ProtoNorm \t LayerNorm \t k \t SemiDesign \t Split \t TrainAcc \t TestAcc \t ProtoProxLoss \t AveMaxRatioMaxMeanRatio \t Cov \t CovUnit \t Corr \t DisL2 \t Chebyshev \n")
    #     f.close()


        
    # with open('feature_control_global_2.txt', 'a') as f:
    #     for i in range(len(test_accs)):
    #         f.write("{}\t".format(args.model))
    #         f.write("{}\t".format(args.model_scale))
    #         f.write("{}\t".format(args.dataset))
    #         f.write("{}\t".format(args.weight_decay))
    #         f.write("{}\t".format(args.flipcrop))
    #         f.write("{}\t".format(args.beta))
    #         f.write("{}\t".format(args.zetaCov))
    #         f.write("{}\t".format(args.zetaCS))
    #         f.write("{}\t".format(args.proto_layer))
    #         f.write("{}\t".format(args.proto_norm))
    #         f.write("{}\t".format(args.layer_norm))
    #         f.write("{}\t".format(args.k))
    #         f.write("{}\t".format(args.semi))
    #         f.write("{}\t".format(dsplits[i]))
    #         f.write("{0:4.4f} \t".format(train_accs[i]))
    #         f.write("{0:4.4f} \t".format(test_accs[i]))
    #         f.write("{0:4.11f} \t".format(loss_par_final_list[i]))
    #         f.write("{0:4.11f} \t".format(ave_ave_var_train_list[i]))
    #         f.write("{0:4.11f} \t".format(ave_max_var_train_list[i]))
    #         f.write("{0:4.11f} \t".format(ave_ratio_train_list[i]))
    #         f.write("{0:4.11f} \t".format(maxmean_ratio_train_list[i]))
    #         f.write("{0:4.11f} \t".format(ave_ave_var_test_list[i]))
    #         f.write("{0:4.11f} \t".format(ave_max_var_test_list[i]))
    #         f.write("{0:4.11f} \t".format(ave_ratio_test_list[i]))
    #         f.write("{0:4.11f} \t".format(maxmean_ratio_test_list[i]))

    #         #f.write("{0:4.8f} \t".format(mean_class_cov_list[i]))
    #         #f.write("{0:4.8f} \t".format(mean_class_cov_unit_list[i]))
    #         #f.write("{0:4.8f} \t".format(mean_class_corr_list[i]))
    #         for ft in range(len(ftr_pct)):
    #             f.write("{0:4.11f} \t".format(mean_class_cov_list[ft][i]))                                   
    #             f.write("{0:4.11f} \t".format(mean_class_cov_unit_list[ft][i]))                                                                                                                                                       
    #             f.write("{0:4.11f} \t".format(mean_class_corr_list[ft][i])) 
    #         f.write("{0:4.11f} \t".format(mean_dissimsq_offdiag_list[i]))
    #         f.write("{0:4.11f} \n".format(chebyshev_list[i]))
            
    # f.close()






if __name__ == '__main__':
    main()

