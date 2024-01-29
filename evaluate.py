from __future__ import print_function
import os
import sys
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
import copy
from scipy import stats
import random

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torchvision import datasets, transforms, models
from torch.hub import load_state_dict_from_url
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights, convnext_tiny, ConvNeXt_Tiny_Weights, densenet121, DenseNet121_Weights, swin_t, Swin_T_Weights
#from torch.utils.tensorboard import SummaryWriter

#from models.wideresnet import *
from models.resnet import *
from models.densenet import *
#from models.resnext import *
from models.simple import *
#from models.allconv import *
#from models.wideresnet import *
from loss_utils import *
from utils_proto import *
from utils import *
from plot_methods import *

from sklearn.model_selection import train_test_split

from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, L2CarliniWagnerAttack, LinfBasicIterativeAttack, L2DeepFoolAttack, L2ProjectedGradientDescentAttack, LinfProjectedGradientDescentAttack

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
parser.add_argument('--beta', default=0.1, type=float,
                    help='loss weight for proximity')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='../ProtoRuns/model-cifar10-',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--model',default="ResNet18",
                    help='network to use')
# parser.add_argument('--restart',default=0, type=int,
#                     help='restart training, make sure to specify directory')
parser.add_argument('--restart-epoch',default=100, type=int,
                    help='epoch to restart from')
parser.add_argument('--norm-type', default='batch',
                    help='batch, layer, or instance')
parser.add_argument('--par-servant', default=0, type=int,
                    help='whether normalization is learnable')
# parser.add_argument('--par-sparse', default=0, type=int,
#                     help='force L1 sparsity on prototype images')
# parser.add_argument('--par-zeta', default=0.01, type=float,
#                     help='L1 sparsity multiplier on prototype images')
# parser.add_argument('--expand-data-epoch', default=0, type=int,
#                     help='start mixing data with misclassified combinations with parametric images')
# parser.add_argument('--expand-interval', default=5, type=int,
#                     help='number of epochs to wait before re-expanding')
# parser.add_argument('--kldiv', default=0, type=int,
#                     help='enforce kldiv match between prototype and examples')
# parser.add_argument('--gamma', default=0.0, type=float,
#                     help='mult loss for kldiv match')
# parser.add_argument('--mixup', default=0, type=int,
#                     help='augment data with mixup par class x to examples not x')
# parser.add_argument('--alpha-mix', default=0.7, type=float,
#                     help='alpha for mixup')
# parser.add_argument('--mix-interval',default=3, type=int,
#                     help='how often to intra class mix')
parser.add_argument('--par-grad-mult', default=10.0, type=float,
                    help='boost image gradients if desired')
parser.add_argument('--par-grad-clip', default=0.01, type=float,
                    help='max magnitude per update for proto image updates')
# parser.add_argument('--class-centers', default=1, type=int,
#                     help='number of parametric centers per class')
parser.add_argument('--dataset', default="CIFAR10",
                    help='which dataset to use, CIFAR10, CIFAR100, IN100')
# parser.add_argument('--image-train', default=0, type=int,
#                     help='train parametric images on frozen model')
# parser.add_argument('--norm-data', default=0, type=int,
#                     help='normalize data')
# parser.add_argument('--anneal', default="stairstep", 
#                     help='type of LR schedule stairstep, cosine, or cyclic')
# parser.add_argument('--inter-mix', default=0, type=int,
#                     help='fill in holes within same class')
#parser.add_argument('--augmix', default=0, type=int,
#                     help='use augmix data augmentation')
#parser.add_argument('--prime', default=0, type=int,
#                     help='use PRIME data augmentation')
#parser.add_argument('--confusionmix', default=0, type=int,
#                     help='use confusionmix data augmentation')
# parser.add_argument('--js-loss', default=1, type=int,
#                     help='use jensen shannon divergence for augmix')
# parser.add_argument('--pipeline', nargs='+',default=[],
#                     help='augmentation pipeline')
parser.add_argument('--grad-clip', default = 0, type=int,
                     help='clip model weight gradients by 0.5')
# parser.add_argument('--confusion-mode', default = 2, type=int,
#                     help='0 = (mode0,mode0), 1 = (mode1,mode1), 2= (mode0,mode1) 3= (random,random)')
# parser.add_argument('--mode0rand', default = 0, type=int,
#                     help='randomly switch between window crop size 3 and 5 in mode 0')
parser.add_argument('--channel-norm', default = 0, type=int,
                     help='normalize each channel by training set mean and std')
# parser.add_argument('--channel-swap', default = 0, type=int,
#                     help='randomly permute channels augmentation')
# parser.add_argument('--window', nargs='+', default=[], type=int,
#                     help='possible windows for cutouts')
# parser.add_argument('--counts', nargs='+', default=[], type=int,
#                     help='possible counts for windows')
parser.add_argument('--model-scale', default=64, type=float,
                    help='model scale for network')
parser.add_argument('--proto-layer', default = 5, type=int,
                    help='after which block to compute prototype loss')
parser.add_argument('--proto-pool', default ="ave",
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
#parser.add_argument('--renorm-prox', default=0, type=int,
#                    help='set to 1 if proto-norm =0')
parser.add_argument('--psi', default=0.0, type=float,
                    help='weight for proxcos contribution, multiplied by beta')
parser.add_argument('--latent-proto', default=0, type=int,
                    help='whether prototypes should be held in latent space as opposed to image space')
parser.add_argument('--kprox', default=5, type=int,
                    help='topk of each row to consider in proto cosine sim loss')
parser.add_argument('--maxmean', default=1, type=int,
                    help='if 1, will use topk maxes from each row, if 0, topk means from cossim matrix')
parser.add_argument('--proxpwr', default=1.0, type=float,
                    help='power of the L2 dist on data to prototype')
parser.add_argument('--topkprox', default=5, type=int,
                    help='if not 0, will select only topk maxes from kprox selection ie top10 of top5 maxes')
parser.add_argument('--hsphere', default=0, type=int,
                    help='shrink variance on magnitudes to speed convergence')
parser.add_argument('--nsample', default=250, type=int,
                    help='number of pert samples')
#parser.add_argument('--trainpert', default=0, type=int,
#                    help='whether to use training data as the "perturbations" ')
parser.add_argument('--onlineproto', default=0, type=int,
                    help='whether to use online trained prototypes or post-hoc prototypes')
parser.add_argument('--proto-traject', default=0, type=int,
                    help='whether to compute and save prototype trajectories')

parser.add_argument('--assessments', nargs='+', default=[],
                    help='list of strings showing which assessments to make')
parser.add_argument('--pertsource', default='random',
                    help='how to generate perturbations in prototype images')
parser.add_argument('--intrafeatures', default=100, type=int,
                    help='how many features to compute intra covariance')
parser.add_argument('--violin', default=0, type=int,
                    help='whether to output violin plots for some number of prototypes')
parser.add_argument('--plot-protos', nargs='+', default=[0,1,2,3,4], type=int,
                    help='list of prototypes to consider for targetted plotting such as violin')
parser.add_argument('--curve-rate', default = 0.008, type=float,
                    help='curve rate for samples')
parser.add_argument('--encoded', default=0, type=int,
                    help='whether to use encoded prototypes')
parser.add_argument('--encoding', default=0, type=int,
                    help='what encoding to use for prototypes')
parser.add_argument('--schedule', nargs='+', type=float, default=[],
                    help='training points to consider')



parser.add_argument('--targeted', default=1, type=int,
                    help='whether to use targeted attacks for retraining')
parser.add_argument('--gamma', default=0.5, type=float,
                    help='weighting for adv loss during retrain')

parser.add_argument('--limType', default='inf',
                    help='limit type for adv attack')
parser.add_argument('--limit', default=1.0/255.0, type=float,
                    help='limit to place on perturbation field')
parser.add_argument('--iterations', default=10, type=int,
                    help='number of iterations for adv attack')
parser.add_argument('--step-size', default=0.25/255.0, type=float,
                    help='step size for adversarial update')
parser.add_argument('--weightpenalty', default=0.0, type=float,
                    help='weight penalty for retraining')
parser.add_argument('--frozen', default=1, type=int,
                    help='whether to perform frozen boundary points with local mst mixup')
parser.add_argument('--runs', default=10, type=int,
                    help='for applicable assessments, number of random runs')
parser.add_argument('--image-step', default=0.0, type=float,
                    help='for image training, number of decimals to round to')
parser.add_argument('--protoHW', default=32, type=int,
                    help='height and width of prototype image, rest will be masked off')
parser.add_argument('--proto-epochs', nargs='+', default=[100], type=int,
                     help='possible windows for cutouts')
parser.add_argument('--proto-pref', default='onehot',
                    help='height and width of prototype image, rest will be masked off')
parser.add_argument('--proto-pair', nargs='+', type=int, default=[],
                    help='whether to consider two classes specifically')
parser.add_argument('--proto-initial', default="random",
                    help='how to initialize the prototypes, random, zeros, or uniform')
parser.add_argument('--transfer', default=1, type=int,
                    help='whether this is analyzing a transfer learning run with validation split')
parser.add_argument('--source-set', default="IMAGENET1K",
                    help="source dataset")
parser.add_argument('--fcdepth', type=int, default=1,
                    help='number of fc layers')
parser.add_argument('--fcratio', type=float, default=1.0,
                    help='fc neuron dropoff per layer')
parser.add_argument('--fliers', type=int, default=0,
                    help='include outliers in box whiskers')
parser.add_argument('--data', type=int, default=0,
                    help='try to load data and previous training points')

parser.add_argument('--alldir', type=int, default=0,
                    help='whether to run assessments over all directories in targ')
parser.add_argument('--compute-Madv', type=int, default=1,
                    help='whether to compute Madv')
parser.add_argument('--dates', nargs='+', default=[],
                    help='dates to analyze')
parser.add_argument('--arch', nargs='+', default=[],
                    help='architectures to consider')
parser.add_argument('--layer-norm', default=0, type=int,
                    help='whether to use layer norm')
parser.add_argument('--proto-method', default='onehot',
                    help='what prototype method to use')
parser.add_argument('--semi', default=0,
                    help='semi random design')


args = parser.parse_args()

kwargsUser = {}
kwargsUser['norm_type'] = args.norm_type
#kwargsUser['augmix'] = args.augmix
#kwargsUser['prime'] = args.prime
# kwargsUser['js_loss'] = args.js_loss
# kwargsUser['proto_aug'] = args.proto_aug
# kwargsUser['pipeline'] = args.pipeline
# kwargsUser['augmix'] = "augmix" in args.pipeline
# kwargsUser['prime'] = "prime" in args.pipeline
# kwargsUser['confusion'] = "confusion" in args.pipeline
# kwargsUser['pipelength'] = len(args.pipeline)
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
kwargsUser['layer_norm'] = args.layer_norm

assert (args.proto_pool in ['none','max','ave'])



if (args.model == "ResNet18"):
    network_string = "ResNet18"
elif (args.model == "ResNet34"):
    network_string = "ResNet34"
elif (args.model == "SmallNet"):
    network_string = "SmallNet"
elif (args.model == "ResNet50"):
    network_string = "ResNet50"
elif (args.model == "Convnext"):
    network_string = "Convnext"
elif (args.model == "Shuffle"):
    network_string = "ShuffleX2"
elif (args.model == "DenseNet"):
    network_string = "Densenet121"
elif (args.model == "SwinT"):
    network_string = "SwinT"
else:
    print ("Invalid model architecture")
    
def get_datetime():
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H_%M_%S")
    return dt_string


# if (not args.restart and not args.image_train):
#     model_dir = ("{}_{}_beta_{}_k_{}_pool_{}_norm_{}_{}".format("../ProtoRuns/model-{}".format(args.dataset),network_string,args.beta,args.k,args.proto_pool,
#         args.proto_norm,get_datetime()))

#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)

#targ = "/home/lab/nxd551/Desktop/TransferRuns"
targ = "/home/lab/nxd551/Desktop/ProtoRuns"
plottarg = "/home/lab/nxd551/Desktop/PrototypeImage/metric_plots"
summarytarg_ = "/home/lab/nxd551/Desktop/PrototypeImage/metric_summary"
dir_suffix = args.model_dir   #i can put in anything here
#need local global results file                                                                                                                                                                             
#local_results_name = "../ProtoRuns/quicklook_{}_{}.txt".format(args.dataset,get_datetime())                                                                                                                        

#full_dir = os.path.join(targ, model_dir)
model_dir = os.path.join(targ, dir_suffix)
full_dir_plot = os.path.join(plottarg,dir_suffix)

summarytarg = os.path.join(summarytarg_, "{}_{}".format(dir_suffix, get_datetime()))

dir_names = []
dir_names.append(dir_suffix)

if args.alldir:
    for name in os.listdir(targ):
        #if os.path.exists(os.path.join(targ,name,"train_hist.txt")):
        #    dir_names.append(name)
        #    print (name)

        for fname in os.listdir(os.path.join(targ,name)):
            #if ("dcs" in fname) and ("truth" not in fname):
            if ("model" in fname):
                dir_names.append(name)
                break




#print (full_dir)

if args.data:
    if not os.path.exists(model_dir):
        sys.exit("ERROR run model directory does not exist")


if not os.path.exists(full_dir_plot):
    os.makedirs(full_dir_plot)

if not os.path.exists(summarytarg):
    os.makedirs(summarytarg)


use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
torch.cuda.empty_cache()

print ("cuda: ", use_cuda)

def generic_attack(model, X, Ytarg, ini_pert, iterations, step_size, lim=1.0/255, limType='inf', targeted=1, frozen=0, transformDict={}):
    
    A, B, C, D = X.size()
    source_labels = torch.arange(A, dtype=torch.long, device=device)
    model.eval()
    model.multi_out = 0
    for p in model.parameters():
        p.requires_grad = False


    #last_mat_tracker = torch.zeros_like(input_img)
    #last_projmat_tracker = torch.ones_like(input_img)
    #baseStep_matrix = baseStep*torch.ones_like(input_img)

    X_orig = X.clone().detach()

    global_noise_data = ini_pert*torch.rand([A,B,C,D],dtype=torch.float, device=device)
    image_cache_adv = X_orig.clone().detach()
    image_cache_src = X_orig.clone().detach()

    mask_frozen = torch.zeros(A, dtype=torch.bool, device=device)

    for i in range(iterations):

        noise_batch = global_noise_data[0:A].clone().detach().requires_grad_(True).to(device)

        _inputs = X_orig + noise_batch
        _inputs.clamp_(0.0, 1.0)

        _inputs_norm = transformDict['norm'](_inputs)
        
        #no output for stylization "network"
        Z = model(_inputs_norm)

        if frozen:
            with torch.no_grad():
                Yp_adv = Z.data.max(dim=1)[1]
                mask_freeze = (Yp_adv == Ytarg)

                #cache perturbed inputs that have reached target, but have not been frozen yet
                image_cache_adv[mask_freeze & ~mask_frozen] = _inputs[mask_freeze & ~mask_frozen].clone()

                #still update inputs (instead of X_orig), that are not frozen yet and have not reached target (allows algo to somewhat work even if target never reached)
                #comment following line out if, in event that boundary is not crossed will include X_orig in MST, image_cache_src already includes latest image before boundary
                #image_cache_adv[~mask_freeze & ~mask_frozen] = _inputs[~mask_freeze & ~mask_frozen].clone()

                #freeze indices that have just reached target and have not been frozen yet
                mask_frozen[mask_freeze & ~mask_frozen] = True

                #will automatically stop updating once target reached 
                image_cache_src[Yp_adv == source_labels] = _inputs[Yp_adv == source_labels].clone()


        loss = F.cross_entropy(Z, Ytarg)

        loss.backward()

        with torch.no_grad():

            pert_vector = torch.sign(noise_batch.grad)
            #grad_mag = gradients_unscaled.view(gradients_unscaled.shape[0], -1).norm(2, dim=-1)
            #gradients = baseStep*gradients_unscaled  / grad_mag.view(-1, 1, 1, 1)

            #update perturbation field
            if targeted:
                global_noise_data[0:A] -= step_size*pert_vector.data
            else:
                global_noise_data[0:A] += step_size*pert_vector.data

            #project noise if necessary
            if limType =='inf':
                global_noise_data[0:A].clamp_(-lim,lim)
            elif limType == 'l2':
                mags = torch.linalg.norm(global_noise_data[0:A].view(A,-1), dim=1)
                mask = mags > lim      #should be shape [A]

                #print ("mask shape", mask.shape)

                scaling_factors = mags[mask] / lim   #should be shape [A]

                #print ("scaling factor shape", scaling_factors.shape)
                #print ("global noise[mask] shape ", global_noise_data[mask].shape)

                global_noise_data[mask] /= scaling_factors.view(-1,1,1,1)

            noise_batch.grad.zero_()
            #model.grad.zero_()


            #global_noise_data[0:insize] = project_onto_l1_ball(global_noise_data[0:insize], lim_1)


    adv_imgs = (X_orig + global_noise_data[0:A].clone().detach()).clone().detach()
    adv_imgs.clamp_(0.0,1.0)

    #run diagnostics
    #with torch.no_grad():

    #    if frozen:
    #        _pre_norm = transformDict['norm'](image_cache_src.clone())
    #        _adv_norm = transformDict['norm'](image_cache_adv.clone())
    #    else:
    #        _adv_norm = transformDict['norm'](adv_imgs.clone())

    #    Z_pre = model(_pre_norm)
    #    Z = model(_adv_norm)
    #    Yp_pre = Z_pre.data.max(dim=1)[1]
    #    Yp_adv = Z.data.max(dim=1)[1]

    #    sm = F.softmax(Z,dim=1)
    #    prob_adv = sm.data.max(dim=1)[0]

    if frozen:
        #return prob_adv, Yp_adv, Yp_pre, adv_imgs, torch.cat((image_cache_src,image_cache_adv),dim=0)
        return adv_imgs.clone().detach(), torch.cat((image_cache_src, image_cache_adv),dim=0).clone().detach()
    else:    
        return prob_adv, Yp_adv, adv_imgs

#plot_box_whisker(name, feature_list, lbl, L2_par_all[lbl], var_mean, acc=train_acc, layer=kwargsUser['proto_layer'], data="Train")
def plot_box_whisker(args, plotdir, features, label, proto_vals, var_mean, acc, step, data="Train",layer=-1, extraData1=[], extraData2=[], extraData3=[], extraData4=[], **kwargs):
    #max_ind = len(means)
    #print (max_ind)
    #par_sorted, par_sorted_indices = torch.sort(L2_par_all, dim=1)
    sorted_vals, sorted_indices = proto_vals.clone().sort(dim=0)
    sorted_vals = sorted_vals.cpu()
    sorted_indices = sorted_indices.cpu()
    inds = list(range(len(sorted_vals)))
    
    # feature_stds, feature_means = torch.std_mean(features,unbiased=False,dim=0)   #[512]
    # means_sorted, means_sorted_indices = torch.sort(feature_means, dim=0)
    # #sorted_vals, sorted_indices = features.clone().sort(dim=0)
    # sorted_indices = means_sorted_indices.cpu()
    # inds = list(range(len(sorted_indices)))
    # #print (means_sorted.shape)
    # #print (means_sorted_indices.shape)

    features_sorted_cols = features[:,sorted_indices].clone().numpy()    #[num_examples, num_neurons]

    
    plt.rcParams["figure.figsize"] = [12, 8]
    plt.rcParams["figure.autolayout"] = True
    #plt.rcParams["text.usetex"] = False
    plt.rcParams.update({'font.family':'serif'})

    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors = ["blue", "red", "orange", "cyan", "olive", "purple", "pink", "brown", "blue", "blue", "blue", "blue"]

    #bp = ax.boxplot(features_sorted_cols, showbox=False, showcaps=False, flierprops={'markersize': 4})
    bp = ax.boxplot(features_sorted_cols, showbox=False, showcaps=False, showfliers=args.fliers)
    scat = ax.scatter(inds, sorted_vals, zorder=1, s=10, color='green')


    #extraData comes in as tuples of (label, vec)
    if extraData1:
        for ftr in extraData1:
            ax.scatter(inds, ftr[sorted_indices].clone().cpu().numpy(), zorder=1, s=10, color="green",marker="o")

    if extraData2:
        for ftr in extraData2:
            ax.scatter(inds, ftr[sorted_indices].clone().cpu().numpy(), zorder=1, s=10, color="red",marker="^")

    if extraData3:
        for ftr in extraData3:
            ax.scatter(inds, ftr[sorted_indices].clone().cpu().numpy(), zorder=1, s=10, color="blue",marker="s")

    if extraData4:
        for ftr in extraData4:
            ax.scatter(inds, ftr[sorted_indices].clone().cpu().numpy(), zorder=1, s=10, color="orange",marker="d")


            #features_sorted_cols = features[:,sorted_indices].clone().numpy()
            #ax.scatter(inds,tup[1][sorted_indices].clone().cpu().numpy(), zorder=1, s=10, color=colors[v], label="Misclassified_{}".format(tup[0]))

    #ax7.set(xlabel=None)
    ax.set(xticklabels=[])
    ax.tick_params(bottom=False)
    plt.xticks(np.arange(0, features.shape[1], 10))

    plt.plot([], [], ' ', label="Acc: {0:3.3f}".format(float(acc)))
    #print (var_mean**0.5)
    #plt.plot([], [], ' ', label="Mean Std: {0:6.5f}".format(float(var_mean**0.5)))

    # axes and labels
    #ax.set_xlim(-width,len(ind)+width)
    # if layer in [4,5]:
    #     ax.set_ylim(0,5)
    # else:
    #     ax.set_ylim(0,0.5)
        
    ax.set_xlabel('Sorted Activation Index', fontsize=18)
    ax.set_ylabel('Activation Magnitude', fontsize=18)
    #ax.set_title("Activation Boxplots at Prototype Layer {},{}, {} Data".format(layer,args.dataset,data), fontsize=18, wrap=True)
    #xTickMarks = ['Group'+str(i) for i in range(1,6)]
    #ax.set_xticks(ind+width)
    #xtickNames = ax.set_xticklabels(xTickMarks)
    #plt.setp(xtickNames, rotation=45, fontsize=10)

    ## add a legend
    #ax.legend( (rects, scat1), ('Mean Activation', '-1\u03C3'))
    ax.legend()
    #plt.text(0.5, 0.5, "Hello World!")
    plt.savefig("{}/ActivationBoxes_Layer_{}_Class_{}_{}_{}_{}_fliers{}.png".format(plotdir, layer, label, data, step,args.proto_initial,args.fliers))
    plt.close()
    #plot.show()


def prototype_trajectory(proto_im, latent_proto, cs_mat, nclass, model, transformDict):

    k_pair = min(nclass, 12)

    topk_vals, topk_inds = torch.topk(cs_mat, k=k_pair, dim=1, sorted=False)    #[num_class, 10 pct]

    max_vals, max_inds = torch.max(cs_mat, dim=1)

    max_max_val, max_max_inds = torch.topk(max_vals, k=k_pair, dim=0, sorted=False)

    data_tuples = []

    for p_ind in max_max_inds:

        for targ_ind in topk_inds[p_ind]:

            mixed = (0.9*proto_im[p_ind] + 0.1*proto_im[targ_ind])

            mixed_norm = transformDict['norm'](mixed.clone().unsqueeze(0))

            mixed_latent, mixed_Z = model(mixed_norm)

            cs_orig, cs_dest = F.cosine_similarity(mixed_latent, latent_proto[p_ind]), F.cosine_similarity(mixed_latent, latent_proto[targ_ind])

            data_tuples.append((p_ind.item(), targ_ind.item(), cs_mat[p_ind, targ_ind].item(), cs_orig.item(), cs_dest.item()))
                    
    return data_tuples


def plot_cosineconf_matrix(matrix_cos, matrix_conf, step, titleAdd='', plotdir='', protoplotlist=[], datapct='',cmap=plt.cm.Blues):
    
    #plt.rcParams["figure.figsize"] = [12, 12]
    #plt.rcParams["figure.autolayout"] = True
    #plt.rcParams["text.usetex"] = False
    #plt.rcParams.update({'font.family':'serif'})

    cl = np.arange(matrix_cos.shape[0])
    cl_list = [str(x) for x in cl]
    tick_marks = np.arange(len(cl_list))

    #cosine similarity
    if torch.is_tensor(matrix_cos):
        plt.figure(figsize=(12, 12))

        #plt.figure(figsize=(8, 8))
        plt.imshow(matrix_cos.clone().detach(), interpolation='nearest', cmap=cmap)
    
        plt.colorbar()
    
        #plt.xticks(tick_marks, cl_list, rotation=90, fontsize=8)
        #plt.yticks(tick_marks, cl_list, fontsize=8)
        plt.xticks([])
        plt.yticks([])
    
        plt.tight_layout()
        #plt.ylabel('Label', fontsize=14)
        #plt.xlabel('Label', fontsize=14)
        #plt.title("Cosine Similarity " + titleAdd, fontsize=14)

        plt.savefig("{}/CosineSim_{}_{}_{}_.png".format(plotdir,titleAdd, datapct, step))
        plt.close()

    #confusion
    if torch.is_tensor(matrix_conf):
        plt.figure(figsize=(12, 12))

        #plt.figure(figsize=(8, 8))
        plt.imshow(matrix_conf.clone().detach(), interpolation='nearest', cmap=cmap)
    
        plt.colorbar()
    
        #plt.xticks(tick_marks, cl_list, rotation=90, fontsize=8)
        #plt.yticks(tick_marks, cl_list, fontsize=8)
        plt.xticks([])
        plt.yticks([])
    
        plt.tight_layout()
        plt.ylabel('Label', fontsize=14)
        plt.xlabel('Label', fontsize=14)
        plt.title("Confusion Matrix " + titleAdd, fontsize=14)

        plt.savefig("{}/Confusion_{}.png".format(plotdir, datapct))
        plt.close()

    #ratio confusion / cosine sim

    if (torch.is_tensor(matrix_cos) and torch.is_tensor(matrix_conf)):
        ratio_matrix = torch.nan_to_num(matrix_conf / matrix_cos)

        plt.figure(figsize=(12, 12))

        #plt.figure(figsize=(8, 8))
        plt.imshow(ratio_matrix.clone().detach(), interpolation='nearest', cmap=cmap)
    
        plt.colorbar()
    
        plt.xticks(tick_marks, cl_list, rotation=90, fontsize=8)
        plt.yticks(tick_marks, cl_list, fontsize=8)
    
        plt.tight_layout()
        plt.ylabel('Label', fontsize=14)
        plt.xlabel('Label', fontsize=14)
        plt.title("Confusion Matrix " + titleAdd, fontsize=14)

        plt.savefig("{}/Ratio_{}.png".format(plotdir, datapct))
        plt.close()


def bin_search(model, x_source, x_target, source_label,its, transformDict):
    #try to do it with batches

    direction_mask = torch.ones(x_source.size(0), dtype=torch.float, device=device)
    alpha=torch.ones(x_source.size(0), dtype=torch.float, device=device)
    mag_cur=0.5

    src_Y = source_label*torch.ones(x_source.size(0), dtype=torch.long, device=device)
    
    x_source_opt = x_source.clone().detach()
    x_target_opt = x_target.clone().detach()

    
    with torch.no_grad():
        for j in range(its):

            alpha.sub_(mag_cur*direction_mask).clamp_(0.0,1.0)

            #print (alpha.shape)
            #print (alpha)

            scal_mat = alpha.clone().unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(x_source_opt)

            #print (scal_mat.shape)
            #print (scal_mat)
            
            x_cur = ((1.0-scal_mat)*x_source_opt + scal_mat*x_target_opt).clamp_(0.0,1.0)

            x_cur_norm = transformDict['norm'](x_cur)

            x_Z = model(x_cur_norm)

            Yp = x_Z.data.max(dim=1)[1]

            if j < (its-2):
                direction_mask[Yp != src_Y] = 1.0
                direction_mask[Yp == src_Y] = -1.0
                mag_cur*= 0.5
            else:
                direction_mask[Yp != src_Y] = -1.0
                direction_mask[Yp == src_Y] = -1.0

            #if correct, make direction_mask negative so that alpha grows and becomes more like x_target

    return Yp.clone(), alpha.clone(), x_cur.clone()


def plot_prototype_history(x_data, L2_im, L2_lat, Ang_im, Ang_lat, titleAdd='', plotdir=''):
    



    plt.rcParams["figure.figsize"] = [12, 8]
    plt.rcParams["figure.autolayout"] = True
    #plt.rcParams["text.usetex"] = False
    plt.rcParams.update({'font.family':'serif'})

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for y in L2_im:
        ax.plot(x_data[1:], y)  # Plot some data on the axes.
    ax.set_xlabel('Training Data Pct')  # Add an x-label to the axes.
    ax.set_ylabel('L2(xp[t] - xp[t-1])')  # Add a y-label to the axes.
    ax.set_title("{} Delta_L2 Image Space OneHot Prototypes".format(titleAdd))  # Add a title to the axes.
        #ax.legend()  # Add a legend.

    plt.savefig("{}/ProtoHist_L2Im_{}.png".format(plotdir,titleAdd))
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for y in L2_lat:
        ax.plot(x_data[1:], y)  # Plot some data on the axes.
    ax.set_xlabel('Training Data Pct')  # Add an x-label to the axes.
    ax.set_ylabel('L2(xp[t] - xp[t-1])')  # Add a y-label to the axes.
    ax.set_title("{} Delta_L2 Latent Space OneHot Prototypes".format(titleAdd))  # Add a title to the axes.
        #ax.legend()  # Add a legend.

    plt.savefig("{}/ProtoHist_L2Latent_{}.png".format(plotdir,titleAdd))
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for y in Ang_im:
        ax.plot(x_data[1:], y)  # Plot some data on the axes.
    ax.set_xlabel('Training Data Pct')  # Add an x-label to the axes.
    ax.set_ylabel('Theta(xp[t] - xp[t-1])')  # Add a y-label to the axes.
    ax.set_title("{} Delta_Angle Image Space OneHot Prototypes".format(titleAdd))  # Add a title to the axes.
        #ax.legend()  # Add a legend.

    plt.savefig("{}/ProtoHist_AngIm_{}.png".format(plotdir,titleAdd))
    plt.close()


    fig = plt.figure()
    ax = fig.add_subplot(111)

    for y in Ang_lat:
        ax.plot(x_data[1:], y)  # Plot some data on the axes.
    ax.set_xlabel('Training Data Pct')  # Add an x-label to the axes.
    ax.set_ylabel('Theta(xp[t] - xp[t-1])')  # Add a y-label to the axes.
    ax.set_title("{} Delta_Angle Latent Space OneHot Prototypes".format(titleAdd))  # Add a title to the axes.
        #ax.legend()  # Add a legend.

    plt.savefig("{}/ProtoHist_AngLatent_{}.png".format(plotdir,titleAdd))
    plt.close()

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
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy



def record_latent_history(features=[], timestamp='', targ_dir=''):
    #features should be a list of tensors, with the first tensor being the "initial" by which everything will be sorted
    # if kwargs['proto_sort']:
    # sorted_vals, sorted_indices = proto_vals.clone().sort(dim=0)
    # sorted_vals = sorted_vals.cpu()
    # sorted_indices = sorted_indices.cpu()
    # inds = list(range(len(sorted_vals)))
    # if kwargs['proto_sort']:
    #     scat = ax.scatter(inds, sorted_vals, zorder=1, s=10, color='green', label = "Prototype Activation")


    # #extraData comes in as tuples of (label, vec)
    # if extraData:
    #     for t, tup in enumerate(extraData):
    #         #features_sorted_cols = features[:,sorted_indices].clone().numpy()
    #         ax.scatter(inds,tup[1][sorted_indices].clone().cpu().numpy(), zorder=1, s=10, color=colors[v], label="Misclassified_{}".format(tup[0]))

    sorted_vals, sorted_indices = features[0].clone().sort(dim=0)

    with open('{}/latent_history_{}.txt'.format(targ_dir, timestamp), 'a') as f:
        for j in range(len(sorted_indices)):
            f.write("\n")
            f.write("{}\t".format(j))

            for i in range(len(features)):
                f.write("{}\t".format(features[i][sorted_indices][j]))          
    f.close()








def main():
    # setup data loader
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    MEAN = [0.5]*3
    STD = [0.5]*3

    #kwargsUser = {}

    gen_transform_train = transforms.Compose([transforms.ToTensor()])

    if (args.source_set == "CIFAR10"):
        MEAN = [0.5]*3
        STD = [0.5]*3
        if args.channel_norm:
            MEAN = [0.4914, 0.4822, 0.4465]
            STD = [0.2471, 0.2435, 0.2616] 

        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])

        train_transform_tensor = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])
        gen_transform_test = transforms.Compose(
            [transforms.ToTensor()])
    elif(args.source_set == "CIFAR100"):
        #MEAN = [0.5071, 0.4865, 0.4409]
        #STD = [0.2673, 0.2564, 0.2762]
        MEAN = [0.5]*3
        STD = [0.5]*3
        if args.channel_norm:
            MEAN = [0.5071, 0.4865, 0.4409]
            STD = [0.2673, 0.2564, 0.2762]

        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])

        train_transform_tensor = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])
        gen_transform_test = transforms.Compose(
            [transforms.ToTensor()])
    elif(args.source_set == "STL10"):
        #MEAN = [0.5071, 0.4865, 0.4409]                                                                                                                                                                                               
        #STD = [0.2673, 0.2564, 0.2762]                                                                                                                                                                                                
        MEAN = [0.5]*3
        STD = [0.5]*3
        if args.channel_norm:
            MEAN = [0.5071, 0.4865, 0.4409]
            STD = [0.2673, 0.2564, 0.2762]

        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96, padding=4)])

        train_transform_tensor = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96, padding=4)])
        gen_transform_test = transforms.Compose(
            [transforms.ToTensor()])

    elif  (args.source_set == "IN100"):
        #ImageNetFolder = "./Data_ImageNet/"
        #WordsFolder = "./Data_ImageNet/words/"
        MEAN = [0.5]*3
        STD = [0.5]*3
        if args.channel_norm:
            MEAN = [0.485, 0.456, 0.406]
            STD  = [0.229, 0.224, 0.225]

    elif (args.source_set == "FASHION"):
        MEAN = [0.5]
        STD = [0.5]
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

    elif (args.source_set == "IMAGENET1K"):

        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        Hsrc = 224


        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop((Hsrc,Hsrc), antialias=True),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()])

        gen_transform_test = transforms.Compose(
            [transforms.Resize((Hsrc,Hsrc), antialias=True),
             transforms.ToTensor()])

    elif (args.source_set == "TINYIN"):
        #this is situation when CIFAR10, CIFAR100 are analyzed but using a network pretrained on tiny imagenet
        Hsrc = 64

        train_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.RandomResizedCrop(Hsrc,interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.RandomHorizontalFlip()])

        gen_transform_test = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize(Hsrc, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)])

    else:
        print ("ERROR dataset not found")



    # gen_transform_train = transforms.Compose([transforms.ToTensor()])
    # #gen_transform_test = transforms.Compose([transforms.ToTensor()])

    # #first augmentation in pipeline gets [Tensor, Flip, Crop] by default
    # if args.dataset in ["CIFAR10","CIFAR100"]:
    #     train_transform = transforms.Compose(
    #         [transforms.ToTensor(),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, padding=4)])

    #     train_transform_tensor = transforms.Compose(
    #         [transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, padding=4)])
    #     gen_transform_test = transforms.Compose(
    #         [transforms.ToTensor()])

    # elif args.dataset in ["FASHION"]:

    #     train_transform = transforms.Compose(
    #         [transforms.ToTensor(),
    #         transforms.Pad(2),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, padding=4)])

    #     train_transform_tensor = transforms.Compose(
    #         [transforms.Pad(2),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, padding=4)])

    #     gen_transform_test = transforms.Compose(
    #         [transforms.ToTensor(),
    #         transforms.Pad(2)])
        
    # elif args.dataset in ["TINYIN"]:
    #     train_transform = transforms.Compose(
    #         [transforms.ToTensor(),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(64, padding=4)])

    #     train_transform_tensor = transforms.Compose(
    #         [transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(64, padding=4)])
    #     gen_transform_test = transforms.Compose(
    #         [transforms.ToTensor()])

    # elif args.dataset in ["IN100"]:
    #     train_transform = transforms.Compose(
    #         [transforms.ToTensor(),
    #         transforms.RandomResizedCrop(224),
    #          transforms.RandomHorizontalFlip()])

    #     train_transform_tensor = transforms.Compose(
    #         [transforms.RandomResizedCrop(224),
    #          transforms.RandomHorizontalFlip()])
    #     gen_transform_test = transforms.Compose(
    #         [transforms.ToTensor(),
    #          transforms.Resize(256),
    #          transforms.CenterCrop(224)])


    # else:
    #     print ("ERROR setting transforms")





    #comp_list_test = [transforms.ToTensor()]
    
    if (args.dataset == "CIFAR10"):

        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=gen_transform_test)
        train_loader_basic = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

        #both augmix and PRIME want [crop, flip] before their augmentations
        #trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
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

        #data_schedule = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
        #original paper [0.25, 0.4, 0.6, 0.7, 0.8, 0.9]
        data_schedule = args.schedule    
    elif (args.dataset == "CIFAR100"):

        trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=gen_transform_test)
        train_loader_basic = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)


        #trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
        #trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        #cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=gen_transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        #num_classes = 100
        kwargsUser['num_classes'] = 100
        nclass=100
        nchannels = 3
        H, W = 32, 32
        targs_ds = trainset.targets

        #data_schedule = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        #original paper [0.25, 0.4, 0.6, 0.7, 0.8, 0.9]
        data_schedule = args.schedule

    elif (args.dataset == "STL10"):

        trainset = torchvision.datasets.STL10(root='../data', split='train', download=True, transform=gen_transform_test)
        train_loader_basic = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)


        #trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)                                                                                                                
        #trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)                                                                                                                
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)                                                                                                                      
        #cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)                                                                                                                        
        testset = torchvision.datasets.STL10(root='../data', split='test', download=True, transform=gen_transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        #num_classes = 100                                                                                                                                                                                                             
        kwargsUser['num_classes'] = 10
        nclass=10
        nchannels = 3
        H, W = 96, 96
        targs_ds = trainset.labels

        #data_schedule = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]                                                                                                                                               
        #original paper [0.25, 0.4, 0.6, 0.7, 0.8, 0.9]                                                                                                                                                                                
        data_schedule = args.schedule

    elif (args.dataset == "FASHION"):

        trainset_basic = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=gen_transform_train)
        train_loader_basic = torch.utils.data.DataLoader(trainset_basic, batch_size=args.batch_size, shuffle=True, **kwargs)


        #trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
        trainset = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=train_transform)
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        #cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=gen_transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        #num_classes = 100
        kwargsUser['num_classes'] = 10
        nclass=10
        nchannels = 1
        H, W = 32, 32
        targs_ds = trainset.targets

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
        #data_schedule = trainset.targets
        
    elif (args.dataset == "TINYIN"):
        train_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.RandomCrop(Hsrc, padding=4),
                 transforms.RandomHorizontalFlip()])

        gen_transform_test = transforms.Compose(
                [transforms.ToTensor()])
        
        #trainset = datasets.ImageFolder(
        #    './Data_ImageNet/train_100',
        #    transform=gen_transform_train)
        #train_loader_basic = torch.utils.data.DataLoader(trainset_basic, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)


        trainset = datasets.ImageFolder(
            './tiny-imagenet-200/train')
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        #cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        testset = datasets.ImageFolder(
            './tiny-imagenet-200/val/images')
        #test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        kwargsUser['num_classes'] = 200
        nclass = 200
        nchannels = 3
        H, W = 64, 64
        targs_ds = trainset.targets

        #data_schedule = [0.25, 0.4, 0.6, 0.7, 0.8, 0.9] 
        data_schedule = args.schedule
    elif (args.dataset == "OXFORD"):
        trainset = torchvision.datasets.OxfordIIITPet(root='../data', split='trainval', download=True)
        testset = torchvision.datasets.OxfordIIITPet(root='../data', split='test', download=True)
        nclass=37
        nchannels=3
        H,W = 224, 224
        targs_ds = trainset._labels

        #data_schedule = [0.11,0.16,0.21,0.31, 0.41,0.51, 0.61,0.71, 0.81,0.91]
        data_schedule=args.schedule
    elif (args.dataset == "FLOWERS"):
        trainset = torchvision.datasets.Flowers102(root='../data', split='train', download=True)
        valset = torchvision.datasets.Flowers102(root='../data', split='val', download=True)
        testset = torchvision.datasets.Flowers102(root='../data', split='test', download=True)
        
        trainset._labels.extend(valset._labels)
        trainset._image_files.extend(valset._image_files)
        nclass = 102
        nchannels=3
        H,W = 224, 224
        targs_ds = trainset._labels

        #data_schedule = [0.25, 0.5, 0.75]
        data_scedule = args.schedule
    elif (args.dataset == "IMAGENET1K"):
        nclass=1000
        nchannels=3
        H,W = 224, 224
        data_schedule=args.schedule
        
        
    else:
          
        print ("Error getting dataset")



    transformDict = {}

    transformDict['basic'] = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(H, padding=4),transforms.Normalize(MEAN, STD)])
    transformDict['flipcrop'] = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(H, padding=4)])
    transformDict['norm'] = transforms.Compose([transforms.Normalize(MEAN, STD)])
    transformDict['mean'] = MEAN
    transformDict['std'] = STD



    # splits = []

    # #all_inds = np.arange(len(trainset.targets))

    # #need to save training data used in future in order to use this
    # if args.data:
    #     all_inds = np.arange(len(targs_ds))
        
    #     for d in data_schedule:
    #         inds_tr, inds_te, y_tr, y_te = train_test_split(all_inds, targs_ds, test_size=d, random_state=args.seed, stratify=targs_ds)
    #         #inds_tr, inds_te, y_tr, y_te = train_test_split(all_inds, trainset.targets, test_size=d, random_state=args.seed, stratify=trainset.targets)
    #         splits.append(inds_te.copy())
    
    #         #add 100% training
    #     splits.append(all_inds)
    # #data_schedule.append(1.0)


    p_magnitude = 0.03



    getdt = get_datetime()

    with open('{}/metric_trends_general_{}_proto{}_{}.txt'.format(summarytarg, args.dataset, args.proto_layer, getdt), 'a') as f:
        #f.write("topk per row: {}\n".format(args.kprox))                                                                                                                                                                     
        f.write("Model \t Scale \t Wd \t FlipCrop \t TrainAcc \t TestAcc \t TrainLoss \t TestLoss \t Gap \t MgMean \t MgStd \t MadvMean \t MadvStd \t rhoCSMC \t NCM_train \t NCM_test \n")

    f.close()

    with open('{}/metric_trends_detailed_{}_proto{}_{}.txt'.format(summarytarg, args.dataset, args.proto_layer, getdt), 'a') as f:
        #f.write("topk per row: {}\n".format(args.kprox))                                                                                                                                                                     
        f.write("Model \t Scale \t Wd \t FlipCrop \t TrainAcc \t TestAcc \t TrainLoss \t TestLoss \t Gap \t MgMean \t MgStd \t MadvMean \t MadvStd \t rhoCSMC \t NCM_train \t NCM_test \n")

    f.close()

    Mg_comp = []
    Madv_comp = []
    gap_comp = []
    data_pct_comp = []
    test_acc_comp = []

    Mg_cov = []
    Mg_corr = []

    Mg_comp_lists = []
    Madv_comp_lists = []
    gap_comp_lists = []
    data_pct_comp_lists = []
    test_acc_comp_lists = []

    Weight_dot_comp = []
    Weight_CS_comp = []

    Weight_dot_comp_lists = []
    Weight_CS_comp_lists = []

    hyper_labels = []

    NCM_train_acc = []
    NCM_test_acc = []
    NCM_train_acc_actual = []
    NCM_test_acc_actual = []

    row_img_mean_comp = []
    row_latent_mean_comp = []
    col_img_mean_comp = []
    col_latent_mean_comp = []

    row_img_std_comp = []
    row_latent_std_comp = []
    col_img_std_comp = []
    col_latent_std_comp = []

    row_img_mean_comp_lists = []
    row_latent_mean_comp_lists = []
    col_img_mean_comp_lists = []
    col_latent_mean_comp_lists = []

    row_img_std_comp_lists = []
    row_latent_std_comp_lists = []
    col_img_std_comp_lists = []
    col_latent_std_comp_lists = []

    ftr_pct = [1.0, 0.35, 0.2, 0.1, 0.05, 0.02]

    cov_mean_within_comp = []
    corr_mean_within_comp = []
    cov_mean_within_comp_lists = []
    corr_mean_within_comp_lists = []

    unified_Mg_covcov_comp = []
    unified_Mg_corrcorr_comp = []
    unified_Mg_covcov_comp_lists = []
    unified_Mg_corrcorr_comp_lists = []

    simple_Mg_covcov_comp = []
    simple_Mg_corrcorr_comp = []
    simple_Mg_covcov_comp_lists = []
    simple_Mg_corrcorr_comp_lists = []

    risk_cov_l2_comp = []
    risk_cov_l2_comp_lists = []
    risk_covabs_l2_comp = []
    risk_covabs_l2_comp_lists = []
    risk_l2_cov_comp = []
    risk_l2_cov_comp_lists = []

    risk_covunit_cs_comp = []
    risk_covunit_cs_comp_lists = []
    risk_cs_covunit_comp = []
    risk_cs_covunit_comp_lists = []

    risk_covunitabs_cs_comp = []
    risk_covunitabs_cs_comp_lists = []



    for fp in ftr_pct:
        cov_mean_within_comp.append([])
        corr_mean_within_comp.append([])
        cov_mean_within_comp_lists.append([])
        corr_mean_within_comp_lists.append([])
        unified_Mg_covcov_comp.append([])
        unified_Mg_corrcorr_comp.append([])
        unified_Mg_covcov_comp_lists.append([])
        unified_Mg_corrcorr_comp_lists.append([])
        simple_Mg_covcov_comp.append([])
        simple_Mg_corrcorr_comp.append([])
        simple_Mg_covcov_comp_lists.append([])
        simple_Mg_corrcorr_comp_lists.append([])

        risk_cov_l2_comp.append([])
        risk_cov_l2_comp_lists.append([])
        risk_l2_cov_comp.append([])
        risk_l2_cov_comp_lists.append([])

        risk_covunit_cs_comp.append([])
        risk_covunit_cs_comp_lists.append([])
        risk_cs_covunit_comp.append([])
        risk_cs_covunit_comp_lists.append([])

        risk_covabs_l2_comp.append([])
        risk_covabs_l2_comp_lists.append([])

        risk_covunitabs_cs_comp.append([])
        risk_covunitabs_cs_comp_lists.append([])

    cov_mean_outside_comp = []
    corr_mean_outside_comp = []
    cov_mean_outside_comp_lists = []
    corr_mean_outside_comp_lists = []

    data_pct = []
    data_accs = []
    data_Mg = []



    #each directory represents a different hyperparameter setting
    for dr in dir_names:

        if not any(date in dr for date in args.dates):
            continue
        #if not (args.dataset in name):
        #    continue
        # if not any(b in dr for b in args.beta):
        #      continue
        if not any(a in dr for a in args.arch):
            continue
        if not any(c in dr for c in args.dataset):
            continue
        

        #kwargsUser={}

        proto_folder = os.path.join(plottarg,dr,"prototypes")
        cur_folder = os.path.join(targ,dr)

        if not os.path.exists(proto_folder):
            os.makedirs(proto_folder)

        if os.path.exists('{}/commandline_args.txt'.format(cur_folder)):
            with open('{}/commandline_args.txt'.format(cur_folder), "r") as read_file:                                                                                                                                        
                command_dict = json.load(read_file)                                                                                                                                                                     
                read_file.close()
        else:
            continue

        wd = float(command_dict['weight_decay'])
        mscale = int(command_dict['model_scale'])
        mod = command_dict['model']
        fcrop = command_dict['flipcrop']
        ds = command_dict['dataset']

        if args.dataset=="CIFAR100" and ds == "CIFAR10":
            continue
        if args.dataset=="CIFAR100" and ds == "STL10":
            continue
        if args.dataset=="CIFAR10" and ds == "CIFAR100":
            continue
        if args.dataset=="CIFAR10" and ds =="STL10":
            continue
        if args.dataset=="STL10" and ds == "CIFAR10":
            continue
        if args.dataset=="STL10" and ds == "CIFAR100":
            continue

        if fcrop:
            continue

        if mod == "SmallNet":
            continue

        #if mscale not in [20]:
        #    continue

        #if wd not in [5e-4]:
        #    continue

        if len(command_dict['datasplits']) == 1:
            continue

        hasModel=False

        for u in os.listdir(cur_folder):
            if "model" in u:
                hasModel = True

        if not hasModel:
            continue


        print ("weight decay ",wd)
        print ("scale ", mscale)
        print ("model ",mod)
        print ("aug ",fcrop)
        print ("dataset ", ds)

        #hasModel=False
        
        #for u in os.listdir(cur_folder):
        #    if "model" in u:
        #        hasModel = True
                
        #if not hasModel:
        #    continue

        kwargsUser['proto_layer'] = args.proto_layer

        #full_dir_plot = os.path.join(plottarg,dir_suffix)
        #torch.save(model.state_dict(),os.path.join(model_dir, 'model-{}-split{}_init{}.pt'.format(network_string,j,m)))
        #summarytarg = "/home/lab/nxd551/Desktop/PrototypeImage/metric_summary"
        if mod == "ResNet18":
            #continue
            model = ResNet18(nclass = nclass, scale=mscale, channels=3, **kwargsUser).to(device)
            hyper_labels.append(("Res18", str(mscale),str(wd)))
        elif mod == "ResNet34":
            model = ResNet34(nclass = nclass, scale=mscale, channels=3, **kwargsUser).to(device)
            hyper_labels.append(("Res34", str(mscale),str(wd)))
        elif mod == "SmallNet":
            continue
            model = SmallNet(nclass = nclass, scale=mscale, channels=3, drop=0, **kwargsUser).to(device)
            hyper_labels.append(("VGG", str(mscale),str(wd),str(fcrop)))
        elif mod == "DenseNet":
            model = DenseNetCifar(nclass = nclass, scale=mscale , channels=3, **kwargsUser).to(device)
            hyper_labels.append(("Dense", str(mscale),str(wd)))
        else:
            print ("Error getting model")

        Mg = []
        Madv = []
        Mg_std = []
        Madv_std = []
        train_accs = []
        test_accs = []
        train_losses = []
        test_losses = []

        #data_pct = []
        data_opts = [0.25, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
        #data_accs = []
        #data_Mg = []
        #NCM_train_acc = []
        #NCM_test_acc = []
        #NCM_train_acc_actual = []
        #NCM_test_acc_actual = []
        
        row_img_mean_hyper_list = []
        row_latent_mean_hyper_list = []
        col_img_mean_hyper_list = []
        col_latent_mean_hyper_list = []

        row_img_std_hyper_list = []
        row_latent_std_hyper_list = []
        col_img_std_hyper_list = []
        col_latent_std_hyper_list = []

        cov_mean_within_hyper_list = []
        corr_mean_within_hyper_list = []

        for fp in ftr_pct:
            cov_mean_within_hyper_list.append([])
            corr_mean_within_hyper_list.append([])

        cov_mean_outside_hyper_list = []
        corr_mean_outside_hyper_list = []

        unified_Mg_covcov_hyper_list = []
        unified_Mg_corrcorr_hyper_list = []

        for fp in ftr_pct:
            unified_Mg_covcov_hyper_list.append([])
            unified_Mg_corrcorr_hyper_list.append([])

        simple_Mg_covcov_hyper_list = []
        simple_Mg_corrcorr_hyper_list = []

        for fp in ftr_pct:
            simple_Mg_covcov_hyper_list.append([])
            simple_Mg_corrcorr_hyper_list.append([])

        risk_cov_l2_hyper_list = []
        risk_l2_cov_hyper_list = []
        risk_covunit_cs_hyper_list = []
        risk_cs_covunit_hyper_list = []

        risk_covabs_l2_hyper_list = []
        risk_covunitabs_cs_hyper_list = []


        for fp in ftr_pct:
            risk_cov_l2_hyper_list.append([])
            risk_l2_cov_hyper_list.append([])
            risk_covunit_cs_hyper_list.append([])
            risk_cs_covunit_hyper_list.append([])
            risk_covabs_l2_hyper_list.append([])
            risk_covunitabs_cs_hyper_list.append([])


        weight_dot_hyper_list = []
        weight_CS_hyper_list = []

        # cov_mean_within_0_hyper_list = []
        # corr_mean_within_0_hyper_list = []
        # cov_mean_within_1_hyper_list = []
        # corr_mean_within_1_hyper_list = []
        # cov_mean_within_2_hyper_list = []
        # corr_mean_within_2_hyper_list = []
        # cov_mean_within_3_hyper_list = []
        # corr_mean_within_3_hyper_list = []
        # cov_mean_within_4_hyper_list = []
        # corr_mean_within_4_hyper_list = []



        with torch.no_grad():
            targets_onehot = torch.arange(nclass, dtype=torch.long, device=device)   #cpu
            total_runs = args.runs

        #hasModel=False

        #for u in os.listdir(cur_folder):
        #    if "model" in u:
        #        hasModel = True

        #if not hasModel:
        #    continue

        cur_init = -1
        #for each model in the directory (for each initialization of the hyperparameter setting)
        for n, name in enumerate(os.listdir(cur_folder)):
            

            if "model" in name:
                #cur_split = "0"
                #cur_init = n
                cur_init += 1
                print (cur_init)



                split_num = []
                for char in name[-8:]:
                    if char.isdigit():
                        split_num.append(char)

                cur_num = int("".join(split_num))
                cur_split = 2*cur_num + 2     #0->2, 1->4   2->6

                data_num = []
                for char in name[-18:-8]:
                    if char.isdigit() or char==".":
                        data_num.append(char)
                cur_data = float("".join(data_num))

                print (name)
                print (cur_data)
                print (cur_split)

                cur_seed = 42

                if args.semi:
                    cur_seed = cur_split

                all_inds = np.arange(len(targs_ds))
        
                if cur_data < 1.0:
                    inds_tr, inds_te, y_tr, y_te = train_test_split(all_inds, targs_ds, test_size=cur_data, random_state=cur_seed, stratify=targs_ds)

                    #correspondence issues between folds
                    subtrain = torch.utils.data.Subset(trainset, inds_te.copy())
                else:
                    subtrain = torch.utils.data.Subset(trainset, all_inds)
                #subtest = torch.utils.data.Subset(test, test_idx)
                #print (len(subtrain))


                eval_train_loader = torch.utils.data.DataLoader(subtrain, batch_size=args.batch_size, shuffle=False,**kwargs)
                cur_loader = torch.utils.data.DataLoader(subtrain, batch_size=args.batch_size, shuffle=True, **kwargs)
                test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
                

                #if cur_init not in [0, 1,11,21]:
                #    continue

                model_pnt = torch.load(os.path.join(cur_folder,name))
                print ("loading model")
                #model.load_state_dict(model_pnt, strict=False)
                #if mod == "SmallNet" and ds == "CIFAR10":
                #    model.transfer_param()

                model.load_state_dict(model_pnt)
                model.eval()
                model.multi_out =0
                for p in model.parameters(): 
                    p.requires_grad = False



                for name, param in model.named_parameters():
                    if 'linear' in name and 'weight' in name:
                        #print (name)                                                                                                                                                                                                 
                        #print (param.shape) #[nclass,512]                                                                                                                                                                            
                        #print (param[0].shape) #[512]  512 weights from pattern to C0                                                                                                                                                
                        #print (torch.min(param[0]))                                                                                                                                                                                  
                        #print (torch.max(param[0]))
                        #print (param.shape)
                        fc_weights = param.clone().cpu()  #[nclass, 512]
                        #fc_weights_t = param.clone().t()  #[512,nclass]

                fc_weights_unit = F.normalize(fc_weights, dim=1)
                all_pairs_weight_dot = fc_weights @ fc_weights.t()
                all_pairs_weight_CS = fc_weights_unit @ fc_weights_unit.t()
                #weights_dots.append(torch.mean(all_pairs_weight_dot))
                #weights_CS.append(torch.mean(all_pairs_weight_CS))

                #reconfirm accuracies and losses
                loss_train, acc_train = eval_train(args, model, device, eval_train_loader, transformDict)
                loss_test, acc_test = eval_test(args, model, device, test_loader, transformDict)

                #if "data_dep" in args.assessments:
                #    data_pct.append(data_opts[cur_init // 5])
                #    data_accs.append(acc_test)

                if "data_dep" in args.assessments:
                    data_pct.append(cur_data)
                    data_accs.append(acc_test)

            else:
                continue

            #do the following for a particular model "n" in the current folder
            pnt_prototype_batches = []
            pnt_prototype_latents = []
            #delta_profiles = []
            #last_losses = []
            pnt_Mg = []
            pnt_Madv = []
            pnt_CS_mats = []
            pnt_L2_mats = []

            model.eval()

            last_losses = []

            run=0
            counter=0
            model.multi_out=1

            if args.proto_method == 'onehot':
            
                for run in range(args.runs):
                    print ("computing prototypes")

                    par_images_glob = torch.rand([nclass,3,H,W],dtype=torch.float, device=device)
                    last_loss = train_image_nodata(args, model, device, par_images_glob, iterations=200, transformDict=transformDict, targets=targets_onehot, **kwargsUser)
                    par_images_glob = par_images_glob.cpu()
                    #print (last_loss)

                    with torch.no_grad():
                        _par_images_final = par_images_glob.clone().detach().requires_grad_(False).to(device)
                        _par_images_final_norm = transformDict['norm'](_par_images_final)
                        L2_img, logits_img = model(_par_images_final_norm)
                        pnt_prototype_latents.append(L2_img.clone())
                        pred = logits_img.max(1, keepdim=True)[1]
                        probs = F.softmax(logits_img)
                        #print (torch.max(logits_img,dim=1)[1])

                    pnt_prototype_batches.append(par_images_glob.clone())
                    torch.save(par_images_glob, os.path.join(proto_folder,'onehot_post_{}_init{}_step{}_{}_run{}.pt'.format(args.dataset, cur_init, args.image_step, args.proto_initial, run)))

                    l2_mat_latent_temp = all_pairs_L2(L2_img.clone())
                    #compute Mg
                    cos_mat_latent_temp = torch.zeros(nclass,nclass, dtype=torch.float)
                    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
                    #latent_proto_all = torch.zeros([nclass,num_ftrs], dtype=torch.float)
                    #print (nclass)
                    for i in range(nclass):
                        for q in range(nclass):
                            if i != q:
                                #print (L2_img[i].shape)
                                cos_mat_latent_temp[i,q] = cos_sim(L2_img[i].view(-1), L2_img[q].view(-1)).clone().cpu()

                    pnt_CS_mats.append(cos_mat_latent_temp.clone())
                    pnt_L2_mats.append(l2_mat_latent_temp.clone())

                    pnt_Mg.append((1.0-torch.mean(cos_mat_latent_temp)).clone())

                    #compute Madv
                    if args.compute_Madv:
                        model.eval()
                        model.multi_out=0
                        attack = L2DeepFoolAttack(overshoot=0.02,candidates=5)
                        preprocessing = dict(mean=MEAN, std=STD, axis=-3)
                        fmodel = PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing)
                        CS_df_latents = torch.zeros(nclass, dtype=torch.float)

                        cur_proto = par_images_glob.clone().to(device)

                        raw, X_new_torch, is_adv = attack(fmodel, cur_proto, targets_onehot, epsilons=100.0)

                        model.multi_out=1
                        with torch.no_grad():
                            X_new_torch_norm = transformDict['norm'](X_new_torch)
                            latent_p_adv, logits_p_adv = model(X_new_torch_norm)

                            #L2_df_image = torch.linalg.norm((X_new_torch - proto).view(nclass,-1), dim=1)
                            #L2_df_latent = torch.linalg.norm((latent_p_adv - latent_p).view(nclass,-1), dim=1)
                            #CS_df_image = F.cosine_similarity(X_new_torch.view(nclass,-1), proto.view(nclass,-1))
                            CS_df_latents = F.cosine_similarity(latent_p_adv.view(bs,-1), L2_img.view(bs,-1)).cpu()
                            #CS_df_latents = CS_df_latent.clone()

                        pnt_Madv.append(1.0-torch.mean(CS_df_latents))
                    else:
                        pnt_Madv.append(torch.tensor(0.0))
            
            elif args.proto_method == 'data':

                prototypes_data, cs_mat = calc_data_proto_CS_covs(args, model, device, nclass, ftr_length=int(8*mscale), loader=eval_train_loader, ftr_schedule=ftr_pct, transformDict=transformDict)
                


            if "weight_corr" in args.assessments:

                if mscale in [32]:
                    plt.rcParams["figure.figsize"] = [12, 8]
                    plt.rcParams["figure.autolayout"] = True
                    #plt.rcParams["text.usetex"] = False
                    plt.rcParams.update({'font.family':'serif'})

                    sorted_onehots, sorted_onehot_indices = torch.sort(L2_img, dim=1)
                    sorted_onehots, sorted_onehot_indices = sorted_onehots.cpu(), sorted_onehot_indices.cpu()

                    for k in range(10):

                        fig = plt.figure()
                        ax = fig.add_subplot()
                        #ax2 = ax.twinx()

                        xlabel = "Sorted Prototype Activations" 

                        ax.plot(sorted_onehots[k], fc_weights[k][sorted_onehot_indices[k]], 'o', color="black")
                        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

                        ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
                        ax.set_ylabel('Class Weight', fontsize=24, labelpad=18)
                        #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
                        ax.set_title("Weights To Class{} vs. Prototype Activations, {} ".format(k, args.dataset), fontsize=24, pad=18)

                        #ax.legend()
                        plt.savefig("{}/WeightActivations_Class{}_{}_{}_scale{}_wd{}_{}.png".format(summarytarg, k, args.dataset, args.proto_layer, mscale, wd, getdt))
                        plt.close()


            if args.proto_method=='onehot':
                #compute mean and stddev of metrics over all derived prototypes
                pnt_CS_mats = torch.stack(pnt_CS_mats,dim=0)

                #mean latent cosine similarity matrix over 15 runs
                cos_mat_latent = torch.mean(pnt_CS_mats,dim=0)

                pnt_L2_mats = torch.stack(pnt_L2_mats, dim=0)
                L2_mat_latent = torch.mean(pnt_L2_mats, dim=0)
                
                pnt_prototype_latents = torch.stack(pnt_prototype_latents,dim=0)

                #mean latent prototype over 15 runs, use this for each class
                prototype_latent_mean =torch.mean(pnt_prototype_latents,dim=0)

                pnt_Mg = torch.stack(pnt_Mg, dim=0)
                pnt_Madv = torch.stack(pnt_Madv, dim=0)

                std_pnt_Mg, mean_pnt_Mg = torch.std_mean(pnt_Mg)
                std_pnt_Madv, mean_pnt_Madv = torch.std_mean(pnt_Madv)

                Mg.append(mean_pnt_Mg.item())
                Madv.append(mean_pnt_Madv.item())
                Mg_std.append(std_pnt_Mg.item())
                Madv_std.append(std_pnt_Madv.item())

            elif args.proto_method=='data':
                cos_mat_latent = cs_mat
                prototype_latent_mean = prototypes_data
                L2_mat_latent = all_pairs_L2(prototype_latent_mean.clone())
                Mg.append(1.0-torch.mean(cos_mat_latent))
                Madv.append(1.0)
                Mg_std.append(1.0)
                Madv_std.append(1.0)
                model.multi_out = 1

                par_images_random = torch.rand([kwargsUser['num_classes'],nchannels,H,W],dtype=torch.float, device=device)
                loss_derived, par_images_glob = train_image_data(args, model, device, par_images_random, eval_train_loader, iterations=10, mask=0, transformDict=transformDict,targets=-1,**kwargsUser)

                model.multi_out = 0
                with torch.no_grad():
                    par_images_final = par_images_glob.clone()
                    par_images_final_norm = transformDict['norm'](par_images_final)
                    logits_par_final = model(par_images_final_norm)
                    loss_par_final = F.cross_entropy(logits_par_final, targets_onehot)



            train_accs.append(acc_train)
            test_accs.append(acc_test)
            
            train_losses.append(loss_train)
            test_losses.append(loss_test)


            weight_dot_hyper_list.append(torch.mean(all_pairs_weight_dot))
            weight_CS_hyper_list.append(torch.mean(all_pairs_weight_CS))


            datalist = estimate_chebyshev(args, 
                                            model, 
                                            device, 
                                            ftr_pct, 
                                            par_images_glob, 
                                            cos_mat_latent,
                                            L2_mat_latent, 
                                            transformDict, 
                                            nclass,
                                            H)

            # return 
            # 0[class_covs, 
            # 1class_corrs, 
            # 2class_covs_unit, 
            # 3uni_Mg_cov_list, 
            # 4simp_Mg_cov_list, 
            # 5uni_Mg_corr_list, 
            # 6simp_Mg_corr_list, 
            # 7outside_class_covs, 
            # 8outside_class_corrs, 
            # 9dissim_off,
            # 10disL2_off, 
            # 11risk_cov_l2_list,
            # 12risk_l2_cov_list,
            # 13risk_covunit_cs_list,
            # 14risk_cs_covunit_list,
            # 15risk_covabs_l2_list,
            # 16risk_covunitabs_cs_list,
            # 17class_min_offdiag_unit,
            # 18class_mean_offdiag_unit,
            # 19class_max_offdiag_unit,
            # 20class_mean_var_unit,
            # 21class_max_var_unit]


            for f, fp in enumerate(ftr_pct):
                unified_Mg_covcov_hyper_list[f].append(datalist[3][f].item())
                simple_Mg_covcov_hyper_list[f].append(datalist[4][f].item())
                unified_Mg_corrcorr_hyper_list[f].append(datalist[5][f].item())
                simple_Mg_corrcorr_hyper_list[f].append(datalist[6][f].item())

                cov_mean_within_hyper_list[f].append(torch.mean(torch.stack(datalist[0][f],dim=0)).item())
                corr_mean_within_hyper_list[f].append(torch.mean(torch.stack(datalist[1][f],dim=0)).item())

                risk_cov_l2_hyper_list[f].append(datalist[11][f].item())
                risk_l2_cov_hyper_list[f].append(datalist[12][f].item())
                risk_covunit_cs_hyper_list[f].append(datalist[13][f].item())
                risk_cs_covunit_hyper_list[f].append(datalist[14][f].item())
                risk_covabs_l2_hyper_list[f].append(datalist[15][f].item())
                risk_covunitabs_cs_hyper_list[f].append(datalist[16][f].item())

        



            cov_mean_outside_hyper_list.append(torch.mean(torch.stack(datalist[7],dim=0)).item())
            corr_mean_outside_hyper_list.append(torch.mean(torch.stack(datalist[8],dim=0)).item())


            #################################################
            #now compute new prototypes, targeting each class individually, originating from the mean one-hots of other classes
            clusters_img = []
            clusters_latent = []
            q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])

            #for every initialization for every hyperparameter set
            row_img_data = []
            col_img_data = []
            row_latent_data = []
            col_latent_data = []

            for tclass in range(nclass):
                with torch.no_grad():
                    targ_class = tclass*torch.ones(nclass, dtype=torch.long, device=device)

                    #targ_class = torch.zeros([nclass,nclass], dtype=torch.float, device=device)
                    #for k in range(nclass):
                    #     targ_class[k][k] = 0.49
                    #     targ_class[k][tclass] = 0.51

                par_images_src = par_images_glob.clone().to(device)
                
                last_loss = train_image_nodata(args, model, device, par_images_src, iterations=200, transformDict=transformDict, targets=targ_class, **kwargsUser)
                
                with torch.no_grad():
                    cluster_norm = transformDict['norm'](par_images_src)
                    ftr_cluster, logit_cluster = model(cluster_norm)
                    #print (torch.max(logit_cluster,dim=1))     #check classifications

            
                    par_images_src = par_images_src.cpu()
                    clusters_img.append(par_images_src.clone())
                    ftr_cluster = ftr_cluster.cpu()
                    clusters_latent.append(ftr_cluster.clone())

            mom_img_row = []
            mos_img_row = []
            mom_latent_row = []
            mos_latent_row = []
            
            mom_img_col = []
            mos_img_col = []
            mom_latent_col = []
            mos_latent_col = []

            for rowclass in range(nclass):
                #mags_img_cur = torch.linalg.norm(clusters_img[rowclass].view(nclass,-1),dim=1)
                #mags_latent_cur = torch.linalg.norm(clusters_latent[rowclass].view(nclass,-1),dim=1)
                img_norm_cur = F.normalize(clusters_img[rowclass].view(nclass,-1),dim=1).clone()
                latent_norm_cur = F.normalize(clusters_latent[rowclass].view(nclass,-1),dim=1).clone()
                cs_img_cur = img_norm_cur.view(nclass,-1) @ img_norm_cur.view(nclass,-1).t()    #[nclass,nclass]
                cs_latent_cur = latent_norm_cur.view(nclass,-1) @ latent_norm_cur.view(nclass,-1).t()

                #par_tens_flat = clusters_img[rowclass].clone().view(nclass,-1)
                #distmatpow = torch.pow(par_tens_flat, 2).sum(dim=1, keepdim=True).expand(par_tens_flat.shape[0], par_tens_flat.shape[0]) + \
                #torch.pow(par_tens_flat, 2).sum(dim=1, keepdim=True).expand(par_tens_flat.shape[0], par_tens_flat.shape[0]).t()
                #image_distmat = torch.nan_to_num(torch.sqrt(distmatpow.addmm_(par_tens_flat, par_tens_flat.t(), beta=1, alpha=-2)))


                #a.masked_select(~torch.eye(n, dtype=bool)).view(n, n - 1)
                std_img_cur, mean_img_cur = torch.std_mean(cs_img_cur.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1))
                std_latent_cur, mean_latent_cur = torch.std_mean(cs_latent_cur.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1))

                img_q_cur = torch.quantile(cs_img_cur.view(-1), q, dim=0)
                latent_q_cur = torch.quantile(cs_latent_cur.view(-1),q, dim=0)
                
                row_img_data.append([mean_img_cur.clone(), std_img_cur.clone(), img_q_cur[0].clone(),img_q_cur[1].clone(), img_q_cur[2].clone(), img_q_cur[3].clone(), img_q_cur[4].clone()])
                row_latent_data.append([mean_latent_cur.clone(), std_latent_cur.clone(), latent_q_cur[0].clone(),latent_q_cur[1].clone(), latent_q_cur[2].clone(), latent_q_cur[3].clone(), latent_q_cur[4].clone()])


                mom_img_row.append(mean_img_cur.clone())
                mos_img_row.append(std_img_cur.clone())
                mom_latent_row.append(mean_latent_cur.clone())
                mos_latent_row.append(std_latent_cur.clone())

            #stack and transpose to get columns
            clusters_img_stack = torch.stack(clusters_img,dim=0)
            clusters_latent_stack = torch.stack(clusters_latent,dim=0)
            clusters_img_t = torch.transpose(clusters_img_stack,0,1).clone()
            clusters_latent_t = torch.transpose(clusters_latent_stack,0,1).clone()

            for colclass in range(nclass):
                #mags_img_cur = torch.linalg.norm(clusters_img[rowclass].view(nclass,-1),dim=1)
                #mags_latent_cur = torch.linalg.norm(clusters_latent[rowclass].view(nclass,-1),dim=1)
                img_norm_cur = F.normalize(clusters_img_t[colclass].view(nclass,-1),dim=1).clone()
                latent_norm_cur = F.normalize(clusters_latent_t[colclass].view(nclass,-1),dim=1).clone()
                cs_img_cur = img_norm_cur.view(nclass,-1) @ img_norm_cur.view(nclass,-1).t()    #[nclass,nclass]
                cs_latent_cur = latent_norm_cur.view(nclass,-1) @ latent_norm_cur.view(nclass,-1).t()

                std_img_cur, mean_img_cur = torch.std_mean(cs_img_cur.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1))
                std_latent_cur, mean_latent_cur = torch.std_mean(cs_latent_cur.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1))

                img_q_cur = torch.quantile(cs_img_cur.view(-1), q, dim=0)
                latent_q_cur = torch.quantile(cs_latent_cur.view(-1),q, dim=0)

                col_img_data.append([mean_img_cur.clone(), std_img_cur.clone(), img_q_cur[0].clone(),img_q_cur[1].clone(), img_q_cur[2].clone(), img_q_cur[3].clone(), img_q_cur[4].clone()])
                col_latent_data.append([mean_latent_cur.clone(), std_latent_cur.clone(), latent_q_cur[0].clone(),latent_q_cur[1].clone(), latent_q_cur[2].clone(), latent_q_cur[3].clone(), latent_q_cur[4].clone()])

                mom_img_col.append(mean_img_cur.clone())
                mos_img_col.append(std_img_cur.clone())
                mom_latent_col.append(mean_latent_cur.clone())
                mos_latent_col.append(std_latent_cur.clone())



            mom_img_row =torch.stack(mom_img_row,dim=0)
            mos_img_row =torch.stack(mos_img_row,dim=0)
            mom_latent_row = torch.stack(mom_latent_row, dim=0)
            mos_latent_row = torch.stack(mos_latent_row, dim=0)
            
            mom_img_col = torch.stack(mom_img_col,dim=0)
            mos_img_col = torch.stack(mos_img_col,dim=0)
            mom_latent_col = torch.stack(mom_latent_col,dim=0)
            mos_latent_col = torch.stack(mos_latent_col,dim=0)


            #mean of means and mean of std
            row_img_mean_hyper_list.append(torch.mean(mom_img_row).item())
            row_latent_mean_hyper_list.append(torch.mean(mom_latent_row).item())
            col_img_mean_hyper_list.append(torch.mean(mom_img_col).item())
            col_latent_mean_hyper_list.append(torch.mean(mom_latent_col).item())

            row_img_std_hyper_list.append(torch.mean(mos_img_row).item())
            row_latent_std_hyper_list.append(torch.mean(mos_latent_row).item())
            col_img_std_hyper_list.append(torch.mean(mos_img_col).item())
            col_latent_std_hyper_list.append(torch.mean(mos_latent_col).item())


            with open('{}/proto_cluster_row_img_stats_{}_proto{}_{}.txt'.format(summarytarg, args.dataset, args.proto_layer, getdt), 'a') as f:
                                                                                                                                                             
                #f.write("Model \t Scale \t Wd \t FlipCrop \t TrainAcc \t TestAcc \t TrainLoss \t TestLoss \t MgMean \t MgStd \t MadvMean \t MadvStd \t rhoCSMC \t NCM_train \t NCM_test \n")
                for r in range(nclass):
                    f.write("{0} \t {1:2d} \t {2:4.5f} \t {3} \t {4:4.6f} \t {5:4.3f} \t {6:4.3f} \t {7:4.3f} \t {8:4.3f} \t {9:4.3f} \t {10:4.3f} \t {11:4.3f} \n".format(
                        mod,
                        mscale,
                        wd,  
                        fcrop, 
                        acc_test,
                        row_img_data[r][0],
                        row_img_data[r][1],
                        row_img_data[r][2],
                        row_img_data[r][3],
                        row_img_data[r][4],
                        row_img_data[r][5],
                        row_img_data[r][6]))

            f.close()

            with open('{}/proto_cluster_row_latent_stats_{}_proto{}_{}.txt'.format(summarytarg, args.dataset, args.proto_layer, getdt), 'a') as f:
                                                                                                                                                             
                #f.write("Model \t Scale \t Wd \t FlipCrop \t TrainAcc \t TestAcc \t TrainLoss \t TestLoss \t MgMean \t MgStd \t MadvMean \t MadvStd \t rhoCSMC \t NCM_train \t NCM_test \n")
                for r in range(nclass):
                    f.write("{0} \t {1:2d} \t {2:4.5f} \t {3} \t {4:4.6f} \t {5:4.3f} \t {6:4.3f} \t {7:4.3f} \t {8:4.3f} \t {9:4.3f} \t {10:4.3f} \t {11:4.3f} \n".format(
                        mod,
                        mscale,
                        wd,  
                        fcrop, 
                        acc_test,
                        row_latent_data[r][0],
                        row_latent_data[r][1],
                        row_latent_data[r][2],
                        row_latent_data[r][3],
                        row_latent_data[r][4],
                        row_latent_data[r][5],
                        row_latent_data[r][6]))

            f.close()

            #quantile plots
    
            if cur_init==0:
                fig = plt.figure()
                ax = fig.add_subplot()
                #ax2 = ax.twinx()                                                                                                                                                                                                                  

                xlabel = "Quantile"

                color = cm.rainbow(np.linspace(0, 1, nclass))

                for n in range(nclass):
                    ax.plot(q, [rld for rld in row_latent_data[n][2:]], color=color[n])

                ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
                ax.set_ylabel('Within Class Cluster Latent CS', fontsize=24, labelpad=18)
                #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)                                                                                                                                                                
                ax.set_title("Quantiles of Within Class Prototype Cluster Latent CS by Target Class, {} ".format(args.dataset), fontsize=24, pad=18)
                ax.set_ylim(0.0,1.0)


                #ax.legend()
                plt.savefig("{}/Quantiles_WithinClass_{}_{}_mod{}_scale{}_wd{}_{}.png".format(summarytarg, args.dataset, args.proto_layer, mod, mscale, wd, getdt))
                plt.close()

            with open('{}/proto_cluster_col_img_stats_{}_proto{}_{}.txt'.format(summarytarg, args.dataset, args.proto_layer, getdt), 'a') as f:
                                                                                                                                                             
                #f.write("Model \t Scale \t Wd \t FlipCrop \t TrainAcc \t TestAcc \t TrainLoss \t TestLoss \t MgMean \t MgStd \t MadvMean \t MadvStd \t rhoCSMC \t NCM_train \t NCM_test \n")
                for r in range(nclass):
                    f.write("{0} \t {1:2d} \t {2:4.5f} \t {3} \t {4:4.6f} \t {5:4.3f} \t {6:4.3f} \t {7:4.3f} \t {8:4.3f} \t {9:4.3f} \t {10:4.3f} \t {11:4.3f} \n".format(
                        mod,
                        mscale,
                        wd,  
                        fcrop, 
                        acc_test,
                        col_img_data[r][0],
                        col_img_data[r][1],
                        col_img_data[r][2],
                        col_img_data[r][3],
                        col_img_data[r][4],
                        col_img_data[r][5],
                        col_img_data[r][6]))

            f.close()

            with open('{}/proto_cluster_col_latent_stats_{}_proto{}_{}.txt'.format(summarytarg, args.dataset, args.proto_layer, getdt), 'a') as f:
                                                                                                                                                             
                #f.write("Model \t Scale \t Wd \t FlipCrop \t TrainAcc \t TestAcc \t TrainLoss \t TestLoss \t MgMean \t MgStd \t MadvMean \t MadvStd \t rhoCSMC \t NCM_train \t NCM_test \n")
                for r in range(nclass):
                    f.write("{0} \t {1:2d} \t {2:4.5f} \t {3} \t {4:4.6f} \t {5:4.3f} \t {6:4.3f} \t {7:4.3f} \t {8:4.3f} \t {9:4.3f} \t {10:4.3f} \t {11:4.3f} \n".format(
                        mod,
                        mscale,
                        wd,  
                        fcrop, 
                        acc_test,
                        col_latent_data[r][0],
                        col_latent_data[r][1],
                        col_latent_data[r][2],
                        col_latent_data[r][3],
                        col_latent_data[r][4],
                        col_latent_data[r][5],
                        col_latent_data[r][6]))

            f.close()


            if cur_init==0:
                fig = plt.figure()
                ax = fig.add_subplot()
                #ax2 = ax.twinx()                                                                                                                                                                                                                  

                xlabel = "Quantile"

                color = cm.rainbow(np.linspace(0, 1, nclass))

                for n in range(nclass):
                    ax.plot(q, [rld for rld in col_latent_data[n][2:]], color=color[n])

                ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
                ax.set_ylabel('Between Class Cluster Latent CS', fontsize=24, labelpad=18)
                #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)                                                                                                                                                                
                ax.set_title("Quantiles of Between Class Prototype Cluster Latent CS by Source Class, {} ".format(args.dataset), fontsize=24, pad=18)
                ax.set_ylim(0.0,1.0)

                #ax.legend()
                plt.savefig("{}/Quantiles_BetweenClass_{}_{}_mod{}_scale{}_wd{}_{}.png".format(summarytarg, args.dataset, args.proto_layer, mod, mscale, wd, getdt))
                plt.close()






            #Mg = []
            #Madv = []
            #train_accs = []
            #test_accs = []
            #train_losses = []
            #test_losses = []

        #once all model initializations are done
        #need to write to two files for mean over model initials and broken down
        #Mg_all = torch.mean(torch.stack(Mg,dim=0))
        #Madv_all = torch.mean(torch.stack(Madv,dim=0))
        #Mg_std_all = torch.mean(torch.stack(Mg_std,dim=0))
        #Madv_std_all = torch.mean(torch.stack(Madv_std,dim=0))

        Mg_all = np.mean(Mg)
        Madv_all = np.mean(Madv)
        Mg_std_all = np.mean(Mg_std)
        Madv_std_all = np.mean(Madv_std)


        model.multi_out = 1

        #plot confusion and cosine similarity matrices, do NCM classifier
        train_acc_NCM = 0.0
        test_acc_NCM = 0.0

        with torch.no_grad():

            #forward last prototypes to get their unit vectors for an NCM classifier
            #prototype_images_norm = transformDict['norm'](par_images_glob.clone().to(device))
            #ftr_proto, logit_proto = model(prototype_images_norm)

            ftr_proto_unit = F.normalize(prototype_latent_mean).to(device)  #[nclass, ftrs]

            for batch_idx, data in enumerate(train_loader_basic):
                X, Y = data[0].to(device), data[1].to(device)
                X = transformDict['norm'](X)
                ftr_X, logit_X = model(X)

                ftr_X_unit = F.normalize(ftr_X)

                CS_X_proto = ftr_X_unit @ ftr_proto_unit.t()  # [batch,ftr] @ [ftr, nclass] = [batch, nclass]

                cs_val, cs_max_ind = torch.max(CS_X_proto, dim=1)   #[batch]
                train_acc_NCM += torch.sum(cs_max_ind == Y)

            train_acc_NCM /= len(train_loader_basic.dataset)
            print ("train acc NCM ", train_acc_NCM)
            NCM_train_acc.append(train_acc_NCM.item())
            NCM_train_acc_actual.append(acc_train)
            

            truth = []
            pred = []
            test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

            for batch_idx, data in enumerate(test_loader):

                #print (batch_idx)

                X, Y = data[0].to(device), data[1].to(device)
                bs = X.size(0)

                X = transformDict['norm'](X)

                L2, logits = model(X)


                L2_X_unit = F.normalize(L2)

                CS_L2_proto = L2_X_unit @ ftr_proto_unit.t()
                cs_val, cs_max_ind = torch.max(CS_L2_proto, dim=1)   #[batch]
                test_acc_NCM += torch.sum(cs_max_ind == Y)

                z, Yp = logits.data.max(dim=1)

                truth.append(Y.clone().cpu())
                pred.append(Yp.clone().cpu())

            test_acc_NCM /= len(test_loader.dataset)
            print ("test acc NCM ", test_acc_NCM)
            NCM_test_acc.append(test_acc_NCM.item())
            NCM_test_acc_actual.append(acc_test)

            truth = torch.cat(truth,dim=0)
            pred = torch.cat(pred,dim=0)
            comp = torch.stack((truth,pred),dim=1)

            conf_mat = torch.zeros(nclass,nclass, dtype=torch.int64)


            for p in comp:
                tl, pl = p.tolist()
                if (tl != pl):
                    conf_mat[tl, pl] = conf_mat[tl, pl] + 1
        #def plot_cosineconf_matrix(matrix_cos, matrix_conf, titleAdd='', plotdir='', protoplotlist=[], datapct='',cmap=plt.cm.Blues):
        plot_cosineconf_matrix(cos_mat_latent, conf_mat, step=args.image_step, titleAdd=args.dataset, plotdir=os.path.join(plottarg,dr), protoplotlist=args.plot_protos, datapct = "1.0")

        cs_variables = cos_mat_latent.reshape(-1)
        misclass_variables = conf_mat.reshape(-1)
        cs_misclass_trend = torch.stack((cs_variables,misclass_variables), dim=0)
        corr_mat = torch.corrcoef(cs_misclass_trend)
        rho_cs_misclass = corr_mat[0][1]


        with open('{}/metric_trends_general_{}_proto{}_{}.txt'.format(summarytarg, args.dataset, args.proto_layer, getdt), 'a') as f:
                                                                                                                                                             
            #f.write("Model \t Scale \t Wd \t FlipCrop \t TrainAcc \t TestAcc \t TrainLoss \t TestLoss \t MgMean \t MgStd \t MadvMean \t MadvStd \t rhoCSMC \t NCM_train \t NCM_test \n")
            f.write("{0} \t {1:2d} \t {2:4.5f} \t {3} \t {4:4.6f} \t {5:4.6f} \t {6:4.6f} \t {7:4.6f} \t {8:4.6f} \t {9:4.3f} \t {10:4.3f} \t {11:4.3f} \t {12:4.3f} \t {13:4.3f} \t {14:4.3f} \t {15:4.3f} \n".format(
                mod,
                mscale,
                wd,  
                fcrop,
                acc_train, 
                acc_test, 
                loss_train,
                loss_test,
                (1.0-acc_test) - (1.0-acc_train),
                Mg_all,
                Mg_std_all,
                Madv_all,
                Madv_std_all,
                rho_cs_misclass,
                train_acc_NCM,
                test_acc_NCM))

        f.close()

        with open('{}/metric_trends_detailed_{}_proto{}_{}.txt'.format(summarytarg, args.dataset, args.proto_layer, getdt), 'a') as f:
                                                                                                                                                              
            #f.write("Model \t Scale \t Wd \t FlipCrop \t TrainAcc \t TestAcc \t TrainLoss \t TestLoss \t MgMean \t MgStd \t MadvMean \t MadvStd \t rhoCSMC \t NCM_train \t NCM_test \n")
            for i in range(len(Mg)):
                f.write("{0} \t {1:2d} \t {2:4.5f} \t {3} \t {4:4.6f} \t {5:4.6f} \t {6:4.6f} \t {7:4.6f} \t {8:4.6f} \t {9:4.3f} \t {10:4.3f} \t {11:4.3f} \t {12:4.3f} \t {13:4.3f} \t {14:4.3f} \t {15:4.3f} \n".format(
                    mod,
                    mscale,
                    wd,  
                    fcrop,
                    train_accs[i], 
                    test_accs[i], 
                    train_losses[i],
                    test_losses[i],
                    (1.0- test_accs[i]) - (1.0-train_accs[i]),
                    Mg[i],
                    Mg_std[i],
                    Madv[i],
                    Madv_std[i],
                    rho_cs_misclass,
                    train_acc_NCM,
                    test_acc_NCM))

        f.close()

        gaps = []

        test_accs_plot = []
        data_pct_plot = []

        for g in range(len(Mg)):
            gaps.append((1.0- test_accs[g]) - (1.0-train_accs[g]))
            test_accs_plot.append(test_accs[g])
            data_pct_plot.append(data_opts[g//5])

        Mg_comp.extend(Mg)
        Madv_comp.extend(Madv)
        gap_comp.extend(gaps)
        data_pct_comp.extend(data_pct_plot)
        test_acc_comp.extend(test_accs_plot)
        

        #individual lists for initializations given one hyperparameter set
        Mg_comp_lists.append(Mg)
        Madv_comp_lists.append(Madv)
        gap_comp_lists.append(gaps)
        data_pct_comp_lists.append(data_pct_plot)
        test_acc_comp_lists.append(test_accs_plot)

        Weight_dot_comp.extend(weight_dot_hyper_list)
        Weight_CS_comp.extend(weight_CS_hyper_list)

        Weight_dot_comp_lists.append(weight_dot_hyper_list)
        Weight_CS_comp_lists.append(weight_CS_hyper_list)


        for f, fp in enumerate(ftr_pct):
            cov_mean_within_comp[f].extend(cov_mean_within_hyper_list[f])
            corr_mean_within_comp[f].extend(corr_mean_within_hyper_list[f])
            cov_mean_within_comp_lists[f].append(cov_mean_within_hyper_list[f])
            corr_mean_within_comp_lists[f].append(corr_mean_within_hyper_list[f])

            unified_Mg_covcov_comp[f].extend(unified_Mg_covcov_hyper_list[f])
            unified_Mg_corrcorr_comp[f].extend(unified_Mg_corrcorr_hyper_list[f])
            unified_Mg_covcov_comp_lists[f].append(unified_Mg_covcov_hyper_list[f])
            unified_Mg_corrcorr_comp_lists[f].append(unified_Mg_corrcorr_hyper_list[f])

            simple_Mg_covcov_comp[f].extend(simple_Mg_covcov_hyper_list[f])
            simple_Mg_corrcorr_comp[f].extend(simple_Mg_corrcorr_hyper_list[f])
            simple_Mg_covcov_comp_lists[f].append(simple_Mg_covcov_hyper_list[f])
            simple_Mg_corrcorr_comp_lists[f].append(simple_Mg_corrcorr_hyper_list[f])

            risk_cov_l2_comp[f].extend(risk_cov_l2_hyper_list[f])
            risk_cov_l2_comp_lists[f].append(risk_cov_l2_hyper_list[f])

            risk_covunit_cs_comp[f].extend(risk_covunit_cs_hyper_list[f])
            risk_covunit_cs_comp_lists[f].append(risk_covunit_cs_hyper_list[f])

            risk_l2_cov_comp[f].extend(risk_l2_cov_hyper_list[f])
            risk_l2_cov_comp_lists[f].append(risk_l2_cov_hyper_list[f])

            risk_cs_covunit_comp[f].extend(risk_cs_covunit_hyper_list[f])
            risk_cs_covunit_comp_lists[f].append(risk_cs_covunit_hyper_list[f])

            risk_covabs_l2_comp[f].extend(risk_covabs_l2_hyper_list[f])
            risk_covabs_l2_comp_lists[f].append(risk_covabs_l2_hyper_list[f])

            risk_covunitabs_cs_comp[f].extend(risk_covunitabs_cs_hyper_list[f])
            risk_covunitabs_cs_comp_lists[f].append(risk_covunitabs_cs_hyper_list[f])


        row_img_mean_comp.extend(row_img_mean_hyper_list)
        row_latent_mean_comp.extend(row_latent_mean_hyper_list)
        col_img_mean_comp.extend(col_img_mean_hyper_list)
        col_latent_mean_comp.extend(col_latent_mean_hyper_list)

        row_img_std_comp.extend(row_img_std_hyper_list)
        row_latent_std_comp.extend(row_latent_std_hyper_list)
        col_img_std_comp.extend(col_img_std_hyper_list)
        col_latent_std_comp.extend(col_latent_std_hyper_list)



        row_img_mean_comp_lists.append(row_img_mean_hyper_list)
        row_latent_mean_comp_lists.append(row_latent_mean_hyper_list)
        col_img_mean_comp_lists.append(col_img_mean_hyper_list)
        col_latent_mean_comp_lists.append(col_latent_mean_hyper_list)

        row_img_std_comp_lists.append(row_img_std_hyper_list)
        row_latent_std_comp_lists.append(row_latent_std_hyper_list)
        col_img_std_comp_lists.append(col_img_std_hyper_list)
        col_latent_std_comp_lists.append(col_latent_std_hyper_list)


        cov_mean_outside_comp.extend(cov_mean_outside_hyper_list)
        corr_mean_outside_comp.extend(corr_mean_outside_hyper_list)
        print (cov_mean_outside_hyper_list)
        cov_mean_outside_comp_lists.append(cov_mean_outside_hyper_list)
        corr_mean_outside_comp_lists.append(corr_mean_outside_hyper_list)
        print (cov_mean_outside_comp_lists)


  


    #after all directories
    #Mg_comp
    #Madv_comp
    #gap_comp
    #statistics
    #print (gap_comp)
    #print (Mg_comp)
    
    gap_rank = stats.rankdata(gap_comp)
    data_rank = stats.rankdata(data_pct_comp)
    test_acc_rank = stats.rankdata(test_acc_comp)
    Mg_rank = stats.rankdata(Mg_comp)
    Madv_rank = stats.rankdata(Madv_comp)

    if "data_dep" in args.assessments:
        cur_rank = test_acc_rank
        cur_comp = test_acc_comp
    else:
        cur_rank = gap_rank
        cur_comp = gap_comp

    weight_dot_rank = stats.rankdata(Weight_dot_comp)
    weight_CS_rank = stats.rankdata(Weight_CS_comp)
    
    row_img_mean_rank = stats.rankdata(row_img_mean_comp)
    row_latent_mean_rank = stats.rankdata(row_latent_mean_comp)
    col_img_mean_rank = stats.rankdata(col_img_mean_comp)
    col_latent_mean_rank = stats.rankdata(col_latent_mean_comp)

    row_img_std_rank = stats.rankdata(row_img_std_comp)
    row_latent_std_rank = stats.rankdata(row_latent_std_comp)
    col_img_std_rank = stats.rankdata(col_img_std_comp)
    col_latent_std_rank = stats.rankdata(col_latent_std_comp)

    #print (gap_rank)
    #print (Mg_rank)

    Mg_corr, Mg_p = stats.pearsonr(cur_comp, Mg_comp)
    Mg_tau, Mg_p_tau = stats.kendalltau(cur_rank, Mg_rank)
    Mg_tau_raw, p_tau_raw = stats.kendalltau(cur_comp,Mg_comp)
    Madv_corr, Madv_p = stats.pearsonr(cur_comp, Madv_comp)
    Madv_tau, Madv_p_tau = stats.kendalltau(stats.rankdata(cur_comp), stats.rankdata(Madv_comp))

    try:
        weight_dot_corr, weight_dot_p = stats.pearsonr(cur_comp, Weight_dot_comp)
        weight_CS_corr, weight_cs_p = stats.pearsonr(cur_comp, Weight_CS_comp)
    except:
        weight_dot_corr = 0
        weight_CS_corr = 0

    weight_dot_tau, weight_dot_p_tau = stats.kendalltau(cur_comp, weight_dot_rank)
    weight_CS_tau, weight_CS_p_tau = stats.kendalltau(cur_comp, weight_CS_rank)

    try:
        corr_row_img_mean, p_row_img_mean = stats.pearsonr(cur_comp, row_img_mean_comp)
        corr_row_latent_mean, p_row_latent_mean = stats.pearsonr(cur_comp, row_latent_mean_comp)
        corr_col_img_mean, p_col_img_mean = stats.pearsonr(cur_comp, col_img_mean_comp)
        corr_col_latent_mean, p_col_latent_mean = stats.pearsonr(cur_comp, col_latent_mean_comp)

        corr_row_img_std, p_row_img_std = stats.pearsonr(cur_comp, row_img_std_comp)
        corr_row_latent_std, p_row_latent_std = stats.pearsonr(cur_comp, row_latent_std_comp)
        corr_col_img_std, p_col_img_std = stats.pearsonr(cur_comp, col_img_std_comp)
        corr_col_latent_std, p_col_latent_std = stats.pearsonr(cur_comp, col_latent_std_comp)
    except:
        corr_row_img_mean = 0
        corr_row_latent_mean = 0
        corr_col_img_mean = 0
        corr_col_latent_mean = 0
        corr_row_img_std = 0
        corr_row_latent_std = 0
        corr_col_img_std = 0
        corr_col_latent_std = 0
        


    tau_row_img_mean, pt_row_img_mean = stats.kendalltau(cur_rank, row_img_mean_rank)
    tau_row_latent_mean, pt_row_latent_mean = stats.kendalltau(cur_rank, row_latent_mean_rank)
    tau_col_img_mean, pt_col_img_mean = stats.kendalltau(cur_rank, col_img_mean_rank)
    tau_col_latent_mean, pt_col_latent_mean = stats.kendalltau(cur_rank, col_latent_mean_rank)

    tau_row_img_std, pt_row_img_std = stats.kendalltau(cur_rank, row_img_std_rank)
    tau_row_latent_std, pt_row_latent_std = stats.kendalltau(cur_rank, row_latent_std_rank)
    tau_col_img_std, pt_col_img_std = stats.kendalltau(cur_rank, col_img_std_rank)
    tau_col_latent_std, pt_col_latent_std = stats.kendalltau(cur_rank, col_latent_std_rank)


    #NEED TO DO THIS BY LIST F
    cov_mean_within_ranks = []
    corr_mean_within_ranks = []
    cov_mean_outside_ranks = []
    corr_mean_outside_ranks = []

    unified_Mg_covcov_ranks = []
    unified_Mg_corrcorr_ranks = []
    simple_Mg_covcov_ranks = []
    simple_Mg_corrcorr_ranks = []

    corr_unified_Mg_covcov = []
    corr_unified_Mg_corrcorr = []
    corr_simple_Mg_covcov = []
    corr_simple_Mg_corrcorr = []

    tau_unified_Mg_covcov = []
    tau_unified_Mg_corrcorr = []
    tau_simple_Mg_covcov = []
    tau_simple_Mg_corrcorr = []

    corr_cov_mean_within = []
    corr_corr_mean_within = []
    corr_cov_mean_outside = []
    corr_corr_mean_outside = []

    tau_cov_mean_within = []
    tau_corr_mean_within = []
    tau_cov_mean_outside = []
    tau_corr_mean_outside = []

    for f, fp in enumerate(ftr_pct):
        cov_mean_within_ranks.append(stats.rankdata(cov_mean_within_comp[f]))
        corr_mean_within_ranks.append(stats.rankdata(corr_mean_within_comp[f]))
        unified_Mg_covcov_ranks.append(stats.rankdata(unified_Mg_covcov_comp[f]))
        unified_Mg_corrcorr_ranks.append(stats.rankdata(unified_Mg_corrcorr_comp[f]))
        simple_Mg_covcov_ranks.append(stats.rankdata(simple_Mg_covcov_comp[f]))
        simple_Mg_corrcorr_ranks.append(stats.rankdata(simple_Mg_corrcorr_comp[f]))

    cov_mean_outside_ranks = stats.rankdata(cov_mean_outside_comp)
    corr_mean_outside_ranks = stats.rankdata(corr_mean_outside_comp)

    for f, fp in enumerate(ftr_pct):
        #print (corr_mean_within_comp[f])
        try:
            corr_cov_mean_within.append(stats.pearsonr(cur_comp, cov_mean_within_comp[f])[0])
            corr_corr_mean_within.append(stats.pearsonr(cur_comp, corr_mean_within_comp[f])[0])

            corr_unified_Mg_covcov.append(stats.pearsonr(cur_comp, unified_Mg_covcov_comp[f])[0])
            corr_unified_Mg_corrcorr.append(stats.pearsonr(cur_comp, unified_Mg_corrcorr_comp[f])[0])

            corr_simple_Mg_covcov.append(stats.pearsonr(cur_comp, simple_Mg_covcov_comp[f])[0])
            corr_simple_Mg_corrcorr.append(stats.pearsonr(cur_comp, simple_Mg_corrcorr_comp[f])[0])
        except:
            corr_cov_mean_within.append(0)
            corr_corr_mean_within.append(0)

            corr_unified_Mg_covcov.append(0)
            corr_unified_Mg_corrcorr.append(0)

            corr_simple_Mg_covcov.append(0)
            corr_simple_Mg_corrcorr.append(0)
            
        tau_cov_mean_within.append(stats.kendalltau(cur_comp, cov_mean_within_ranks[f])[0])
        tau_corr_mean_within.append(stats.kendalltau(cur_comp, corr_mean_within_ranks[f])[0])

        tau_unified_Mg_covcov.append(stats.kendalltau(cur_comp, unified_Mg_covcov_ranks[f])[0])
        tau_unified_Mg_corrcorr.append(stats.kendalltau(cur_comp, unified_Mg_corrcorr_ranks[f])[0])
        tau_simple_Mg_covcov.append(stats.kendalltau(cur_comp, simple_Mg_covcov_ranks[f])[0])
        tau_simple_Mg_corrcorr.append(stats.kendalltau(cur_comp, simple_Mg_corrcorr_ranks[f])[0])

        #print ("gap comp ", gap_comp)
        #print ("cov mean ", cov_mean_outside_comp[f])
    #print (cov_mean_outside_comp)
    #print (corr_mean_outside_comp)
    try:
        corr_cov_mean_outside = stats.pearsonr(cur_comp, cov_mean_outside_comp)[0]
        corr_corr_mean_outside = stats.pearsonr(cur_comp, corr_mean_outside_comp)[0]
    except:
        corr_cov_mean_outside = 0
        corr_corr_mean_outside = 0
        
    tau_cov_mean_outside = stats.kendalltau(cur_comp, cov_mean_outside_ranks)[0]
    tau_corr_mean_outside = stats.kendalltau(cur_comp, corr_mean_outside_ranks)[0]
    #cov_mean_within_rank = stats.rankdata(cov_mean_within_comp)
    #corr_mean_within_rank = stats.rankdata(corr_mean_within_comp)
    #cov_mean_outside_rank = stats.rankdata(cov_mean_outside_comp)
    #corr_mean_outside_rank = stats.rankdata(corr_mean_outside_comp)

    # corr_cov_mean_within, p_cov_mean_within = stats.pearsonr(gap_comp, cov_mean_within_comp)
    # corr_corr_mean_within, p_corr_mean_within = stats.pearsonr(gap_comp, corr_mean_within_comp)
    # corr_cov_mean_outside, p_cov_mean_outside = stats.pearsonr(gap_comp, cov_mean_outside_comp)
    # corr_corr_mean_outside, p_corr_mean_outside = stats.pearsonr(gap_comp, corr_mean_outside_comp)

    # tau_cov_mean_within, pt_cov_mean_within = stats.kendalltau(gap_rank, cov_mean_within_rank)
    # tau_corr_mean_within, pt_corr_mean_within = stats.kendalltau(gap_rank, corr_mean_within_rank)
    # tau_cov_mean_outside, pt_cov_mean_outside = stats.kendalltau(gap_rank, cov_mean_outside_rank)
    # tau_corr_mean_outside, pt_corr_mean_outside = stats.kendalltau(gap_rank, corr_mean_outside_rank)


    plt.rcParams["figure.figsize"] = [12, 8]
    plt.rcParams["figure.autolayout"] = True
    #plt.rcParams["text.usetex"] = False
    plt.rcParams.update({'font.family':'serif'})

    if "data_dep" in args.assessments:
        xlabel = "Training Data (%)"
        comp_lists = data_pct_comp_lists
        xlab = "DataPct"
    else:
        xlabel = "Generalization Gap (0-1 Loss)"
        comp_lists = gap_comp_lists
        xlab = "GenGap"

    if "data_dep" in args.assessments:
        fig = plt.figure()
        ax = fig.add_subplot()
        #ax2 = ax.twinx()

        #xlabel = "Training Data (%)" 
        

        ax.plot(data_pct, data_accs, 'o', color="black", label="Test Accuracy")
        ax.plot(data_pct, Mg, 'x', color="black", label="Mg")


        ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
        ax.set_ylabel('Mg and Test Acc', fontsize=24, labelpad=18)
        #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
        ax.set_title("Mg and Test Accuracy, {} ".format(args.dataset), fontsize=24, pad=18)

        #ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(Mg_corr))
        #ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(Mg_tau))
        ax.set_ylim(0.0,1.0)

        ax.legend()
        plt.savefig("{}/Mg_Acc_DataPct_{}_{}_{}.png".format(summarytarg, args.dataset, args.proto_layer, getdt))
        plt.close()



    fig = plt.figure()
    ax = fig.add_subplot()
    #ax2 = ax.twinx()

    #xlabel = "Generalization Gap (0-1 Loss)" 
    
    color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
    list_indices =  [j for j in range(len(gap_comp_lists))]

    for idx, x, y, c in zip(list_indices, comp_lists, Mg_comp_lists, color):
        #plt.scatter(inds, y,s=10,  color=c)
        #plt.bar(inds,y,width=1, color=c)
        ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel('Mg', fontsize=24, labelpad=18)
    #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
    ax.set_title("Mg, {} ".format(args.dataset), fontsize=24, pad=18)

    ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(Mg_corr))
    ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(Mg_tau))
    ax.set_ylim(0.0,1.0)
    ax.legend()
    plt.savefig("{}/Mg_{}_{}_{}_{}.png".format(summarytarg, xlab,args.dataset, args.proto_layer, getdt))
    plt.close()


    #weight dot and cs plots
    
    fig = plt.figure()
    ax = fig.add_subplot()
    #ax2 = ax.twinx()                                                                                                                                                                                                                  

    #xlabel = "Generalization Gap (0-1 Loss)"

    color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
    list_indices =  [j for j in range(len(gap_comp_lists))]

    for idx, x, y, c in zip(list_indices, comp_lists, Weight_dot_comp_lists, color):
        #plt.scatter(inds, y,s=10,  color=c)                                                                                                                                                                                          
        #plt.bar(inds,y,width=1, color=c)                                                                                                                                                                          
        ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")
        
    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel('Mean Weight Dot Product', fontsize=24, labelpad=18)
    #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)                                                                                                                                                                
    ax.set_title("Mean Weight Dot Product, {} ".format(args.dataset), fontsize=24, pad=18)

    if weight_dot_corr:
        ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(weight_dot_corr))
    ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(weight_dot_tau))

    ax.legend()
    plt.savefig("{}/WeightDot_{}_{}_{}_{}.png".format(summarytarg, xlab,args.dataset, args.proto_layer, getdt))
    plt.close()



    fig = plt.figure()
    ax = fig.add_subplot()
    #ax2 = ax.twinx()
    
    #xlabel = "Generalization Gap (0-1 Loss)"

    color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
    list_indices =  [j for j in range(len(gap_comp_lists))]

    for idx, x, y, c in zip(list_indices, comp_lists, Weight_CS_comp_lists, color):
        #plt.scatter(inds, y,s=10,  color=c)                                                                                                                                                                                          
        #plt.bar(inds,y,width=1, color=c)                                                                                                                                                                                            
        ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")                                                                                                                                                                     

    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel('Mean Weight CS', fontsize=24, labelpad=18)
    #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)                                                                                                                                                              
    
    ax.set_title("Mean Weight CS, {} ".format(args.dataset), fontsize=24, pad=18)

    if weight_CS_corr:
        ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(weight_CS_corr))
    ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(weight_CS_tau))

    ax.legend()
    plt.savefig("{}/WeightCS_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, getdt))
    plt.close()



    #row_img_mean_comp_lists = []
    #row_latent_mean_comp_lists = []
    #col_img_mean_comp_lists = []
    #col_latent_mean_comp_lists = []

    #row_img_std_comp_lists = []
    #row_latent_std_comp_lists = []
    #col_img_std_comp_lists = []
    #col_latent_std_comp_lists = []
    ###############################

    fig = plt.figure()
    ax = fig.add_subplot()
    #ax2 = ax.twinx()

    #xlabel = "Generalization Gap (0-1 Loss)" 
    
    color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
    list_indices =  [j for j in range(len(gap_comp_lists))]

    for idx, x, y, c in zip(list_indices, comp_lists, row_img_mean_comp_lists, color):
        #plt.scatter(inds, y,s=10,  color=c)
        #plt.bar(inds,y,width=1, color=c)
        ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel('Mean Within Cluster CS, Image', fontsize=24, labelpad=18)
    #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
    ax.set_title("Mean Within Cluster CS, Image, {} ".format(args.dataset), fontsize=24, pad=18)

    if corr_row_img_mean:
        ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(corr_row_img_mean))
    ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(tau_row_img_mean))
    
    ax.set_ylim(0.0,1.0)
    ax.legend()
    plt.savefig("{}/MeanWithinClassClusterCS_Image_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, getdt))
    plt.close()




    fig = plt.figure()
    ax = fig.add_subplot()
    #ax2 = ax.twinx()

    #xlabel = "Generalization Gap (0-1 Loss)" 
    
    color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
    list_indices =  [j for j in range(len(gap_comp_lists))]

    for idx, x, y, c in zip(list_indices, comp_lists, row_latent_mean_comp_lists, color):
        #plt.scatter(inds, y,s=10,  color=c)
        #plt.bar(inds,y,width=1, color=c)
        ax.plot(x, y, 'o', color=c, label="Intra, {}".format(hyper_labels[idx]))
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

    for idx, x, y, c in zip(list_indices, comp_lists, col_latent_mean_comp_lists, color):
        #plt.scatter(inds, y,s=10,  color=c)                                                                                                                                                                                          
        #plt.bar(inds,y,width=1, color=c)                                                                                                                                                                                             
        ax.plot(x, y, 'x', color=c, label="Inter, {}".format(hyper_labels[idx]))
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--") 

    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel('Mean Cluster CS, Latent', fontsize=24, labelpad=18)
    #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
    ax.set_title("Marginal Prototype Cluster Intra & Inter CS, Latent, {} ".format(args.dataset), fontsize=24, pad=18)

    #if corr_row_latent_mean:
    #    ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(corr_row_latent_mean))
    #ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(tau_row_latent_mean))
    ax.set_ylim(0.0,1.0)
    ax.legend()
    plt.savefig("{}/MeanBothClassClusterCS_Latent_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, getdt))
    plt.close()



    fig = plt.figure()
    ax = fig.add_subplot()
    #ax2 = ax.twinx()

    #xlabel = "Generalization Gap (0-1 Loss)" 
    
    color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
    list_indices =  [j for j in range(len(gap_comp_lists))]

    for idx, x, y, c in zip(list_indices, comp_lists, row_img_std_comp_lists, color):
        #plt.scatter(inds, y,s=10,  color=c)
        #plt.bar(inds,y,width=1, color=c)
        ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel('Ave Std Within Cluster CS, Image', fontsize=24, labelpad=18)
    #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
    ax.set_title("Ave Std Within Cluster CS, Image vs. {}, {} ".format(xlab, args.dataset), fontsize=24, pad=18)

    if corr_row_img_std:
        ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(corr_row_img_std))
    ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(tau_row_img_std))

    ax.legend()
    plt.savefig("{}/StdWithinClassClusterCS_Image_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, getdt))
    plt.close()



    fig = plt.figure()
    ax = fig.add_subplot()
    #ax2 = ax.twinx()

    #xlabel = "Generalization Gap (0-1 Loss)" 
    
    color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
    list_indices =  [j for j in range(len(gap_comp_lists))]

    for idx, x, y, c in zip(list_indices, comp_lists, row_latent_std_comp_lists, color):
        #plt.scatter(inds, y,s=10,  color=c)
        #plt.bar(inds,y,width=1, color=c)
        ax.plot(x, y, 'o', color=c, label="Intra, {}".format(hyper_labels[idx]))
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

    for idx, x, y, c in zip(list_indices, comp_lists, col_latent_std_comp_lists, color):
        #plt.scatter(inds, y,s=10,  color=c)                                                                                                                                                                                          
        #plt.bar(inds,y,width=1, color=c)                                                                                                                                                                                             
        ax.plot(x, y, 'x', color=c, label="Inter, {}".format(hyper_labels[idx]))
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--") 

    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel('Ave Std Cluster CS, Latent', fontsize=24, labelpad=18)
    #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
    ax.set_title("Ave Std Cluster CS Intra & Inter, Latent, {} ".format(args.dataset), fontsize=24, pad=18)

    #if corr_row_latent_std:
    #    ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(corr_row_latent_std))
    #ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(tau_row_latent_std))

    ax.legend()
    plt.savefig("{}/StdBothClassClusterCS_Latent_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, getdt))
    plt.close()


    #############################



    fig = plt.figure()
    ax = fig.add_subplot()
    #ax2 = ax.twinx()

    #xlabel = "Generalization Gap (0-1 Loss)" 
    
    color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
    list_indices =  [j for j in range(len(gap_comp_lists))]

    for idx, x, y, c in zip(list_indices, comp_lists, col_img_mean_comp_lists, color):
        #plt.scatter(inds, y,s=10,  color=c)
        #plt.bar(inds,y,width=1, color=c)
        ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel('Mean Outside Cluster CS, Image', fontsize=24, labelpad=18)
    #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
    ax.set_title("Mean Outside Cluster CS, Image vs. {}, {} ".format(xlab, args.dataset), fontsize=24, pad=18)

    if corr_col_img_mean:
        ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(corr_col_img_mean))
    ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(tau_col_img_mean))

    ax.legend()
    plt.savefig("{}/MeanOutsideClassClusterCS_Image_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, getdt))
    plt.close()




    fig = plt.figure()
    ax = fig.add_subplot()
    #ax2 = ax.twinx()

    #xlabel = "Generalization Gap (0-1 Loss)" 
    
    color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
    list_indices =  [j for j in range(len(gap_comp_lists))]

    for idx, x, y, c in zip(list_indices, comp_lists, col_latent_mean_comp_lists, color):
        #plt.scatter(inds, y,s=10,  color=c)
        #plt.bar(inds,y,width=1, color=c)
        ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel('Mean Outside Cluster CS, Latent', fontsize=24, labelpad=18)
    #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
    ax.set_title("Mean Outside Cluster CS, Latent vs. {}, {} ".format(xlab, args.dataset), fontsize=24, pad=18)

    if corr_col_latent_mean:
        ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(corr_col_latent_mean))
    ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(tau_col_latent_mean))

    ax.legend()
    plt.savefig("{}/MeanOutsideClassClusterCS_Latent_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, getdt))
    plt.close()



    fig = plt.figure()
    ax = fig.add_subplot()
    #ax2 = ax.twinx()

    #xlabel = "Generalization Gap (0-1 Loss)" 
    
    color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
    list_indices =  [j for j in range(len(gap_comp_lists))]

    for idx, x, y, c in zip(list_indices, comp_lists, col_img_std_comp_lists, color):
        #plt.scatter(inds, y,s=10,  color=c)
        #plt.bar(inds,y,width=1, color=c)
        ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel('Ave Std Outside Cluster CS, Image', fontsize=24, labelpad=18)
    #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
    ax.set_title("Ave Std Outside Cluster CS, Image vs. {}, {} ".format(xlab, args.dataset), fontsize=24, pad=18)

    if corr_col_img_std:
        ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(corr_col_img_std))
    ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(tau_col_img_std))

    ax.legend()
    plt.savefig("{}/StdOutsideClassCS_Image_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, getdt))
    plt.close()



    fig = plt.figure()
    ax = fig.add_subplot()
    #ax2 = ax.twinx()

    #xlabel = "Generalization Gap (0-1 Loss)" 
    
    color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
    list_indices =  [j for j in range(len(gap_comp_lists))]

    for idx, x, y, c in zip(list_indices, comp_lists, col_latent_std_comp_lists, color):
        #plt.scatter(inds, y,s=10,  color=c)
        #plt.bar(inds,y,width=1, color=c)
        ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel('Ave Std Outside Cluster CS, Latent', fontsize=24, labelpad=18)
    #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
    ax.set_title("Ave Std Outside Cluster CS, Latent vs. {}, {} ".format(xlab, args.dataset), fontsize=24, pad=18)

    if corr_col_latent_std:
        ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(corr_col_latent_std))
    ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(tau_col_latent_std))

    ax.legend()
    plt.savefig("{}/StdOutsideClassCS_Latent_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, getdt))
    plt.close()


    #############################################

    #mean covariance within

    for f, fp in enumerate(ftr_pct):

        fig = plt.figure()
        ax = fig.add_subplot()
        #ax2 = ax.twinx()

        #xlabel = "Generalization Gap (0-1 Loss)" 
        
        color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
        list_indices =  [j for j in range(len(gap_comp_lists))]

        for idx, x, y, c in zip(list_indices, comp_lists, cov_mean_within_comp_lists[f], color):
            #plt.scatter(inds, y,s=10,  color=c)
            #plt.bar(inds,y,width=1, color=c)
            ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
            #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

        ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
        ax.set_ylabel('Mean Covariance', fontsize=24, labelpad=18)
        #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
        ax.set_title("Mean Self-Covariance Top {} Ftr vs. {}, {} ".format(fp, xlab, args.dataset), fontsize=24, pad=18)

        if corr_cov_mean_within[f]:
            ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(corr_cov_mean_within[f]))
        ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(tau_cov_mean_within[f]))

        ax.legend()
        plt.savefig("{}/MeanSelfCovFtr_{}_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, fp, getdt))
        plt.close()


        #mean correlation within
        
        fig = plt.figure()
        ax = fig.add_subplot()
        #ax2 = ax.twinx()

        #xlabel = "Generalization Gap (0-1 Loss)" 
        
        color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
        list_indices =  [j for j in range(len(gap_comp_lists))]

        for idx, x, y, c in zip(list_indices, comp_lists, corr_mean_within_comp_lists[f], color):
            #plt.scatter(inds, y,s=10,  color=c)
            #plt.bar(inds,y,width=1, color=c)
            ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
            #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

        ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
        ax.set_ylabel('Mean Correlation', fontsize=24, labelpad=18)
        #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
        ax.set_title("Mean Self-Correlation Top {} Ftr vs. {}, {} ".format(fp, xlab, args.dataset), fontsize=24, pad=18)

        if corr_corr_mean_within[f]:
            ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(corr_corr_mean_within[f]))
        ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(tau_corr_mean_within[f]))

        ax.legend()
        plt.savefig("{}/MeanSelfCorrFtr_{}_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, fp, getdt))
        plt.close()



    #mean cov outside
        
    fig = plt.figure()
    ax = fig.add_subplot()
    #ax2 = ax.twinx()

    #xlabel = "Generalization Gap (0-1 Loss)" 
        
    color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
    list_indices =  [j for j in range(len(gap_comp_lists))]

    for idx, x, y, c in zip(list_indices, comp_lists, cov_mean_outside_comp_lists, color):
        #plt.scatter(inds, y,s=10,  color=c)
        #plt.bar(inds,y,width=1, color=c)
        ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel('Mean Covariance', fontsize=24, labelpad=18)
    #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
    ax.set_title("Mean Covariance BtwnClass Top 5% Ftr vs. {}, {} ".format(xlab, args.dataset), fontsize=24, pad=18)

    if corr_cov_mean_outside:
        ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(corr_cov_mean_outside))
    ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(tau_cov_mean_outside))

    ax.legend()
    plt.savefig("{}/MeanCovBtwnClassFtr_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer,  getdt))
    plt.close()




    #mean corr outside
        
    fig = plt.figure()
    ax = fig.add_subplot()
    #ax2 = ax.twinx()

    #xlabel = "Generalization Gap (0-1 Loss)" 
        
    color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
    list_indices =  [j for j in range(len(gap_comp_lists))]

    for idx, x, y, c in zip(list_indices, comp_lists, corr_mean_outside_comp_lists, color):
        #plt.scatter(inds, y,s=10,  color=c)
        #plt.bar(inds,y,width=1, color=c)
        ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel('Mean Correlation', fontsize=24, labelpad=18)
    #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
    ax.set_title("Mean Correlation BtwnClass Top 5% Ftr vs. {}, {} ".format(xlab, args.dataset), fontsize=24, pad=18)

    if corr_corr_mean_outside:
        ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(corr_corr_mean_outside))
    ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(tau_corr_mean_outside))

    ax.legend()
    plt.savefig("{}/MeanCorrBtwnClassFtr_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, getdt))
    plt.close()


        #############################################

    #unified and simplified metrics

    for f, fp in enumerate(ftr_pct):

        fig = plt.figure()
        ax = fig.add_subplot()
        #ax2 = ax.twinx()

        #xlabel = "Generalization Gap (0-1 Loss)" 
        
        color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
        list_indices =  [j for j in range(len(gap_comp_lists))]

        for idx, x, y, c in zip(list_indices, comp_lists, unified_Mg_covcov_comp_lists[f], color):
            #plt.scatter(inds, y,s=10,  color=c)
            #plt.bar(inds,y,width=1, color=c)
            ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
            #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

        ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
        ax.set_ylabel('Risk', fontsize=24, labelpad=18)
        #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
        ax.set_title("Covariance Risk Top {} Ftr vs. {}, {} ".format(fp, xlab, args.dataset), fontsize=24, pad=18)

        if corr_cov_mean_within[f]:
            ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(corr_unified_Mg_covcov[f]))
        ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(tau_unified_Mg_covcov[f]))

        ax.legend()
        plt.savefig("{}/Unified_Mg_CovCov_{}_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, fp, getdt))
        plt.close()


        #mean correlation within
        
        fig = plt.figure()
        ax = fig.add_subplot()
        #ax2 = ax.twinx()

        #xlabel = "Generalization Gap (0-1 Loss)" 
        
        color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
        list_indices =  [j for j in range(len(gap_comp_lists))]

        for idx, x, y, c in zip(list_indices, comp_lists, unified_Mg_corrcorr_comp_lists[f], color):
            #plt.scatter(inds, y,s=10,  color=c)
            #plt.bar(inds,y,width=1, color=c)
            ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
            #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

        ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
        ax.set_ylabel('Risk', fontsize=24, labelpad=18)
        #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
        ax.set_title("Correlation Risk Top {} Ftr vs. {}, {} ".format(fp, xlab, args.dataset), fontsize=24, pad=18)

        if corr_corr_mean_within[f]:
            ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(corr_unified_Mg_corrcorr[f]))
        ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(tau_unified_Mg_corrcorr[f]))

        ax.legend()
        plt.savefig("{}/Unified_Mg_CorrCorr_{}_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, fp, getdt))
        plt.close()




        fig = plt.figure()
        ax = fig.add_subplot()
        #ax2 = ax.twinx()

        #xlabel = "Generalization Gap (0-1 Loss)" 
        
        color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
        list_indices =  [j for j in range(len(gap_comp_lists))]

        for idx, x, y, c in zip(list_indices, comp_lists, simple_Mg_covcov_comp_lists[f], color):
            #plt.scatter(inds, y,s=10,  color=c)
            #plt.bar(inds,y,width=1, color=c)
            ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
            #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

        ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
        ax.set_ylabel('Risk', fontsize=24, labelpad=18)
        #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
        ax.set_title("Covariance Risk Top {} Ftr vs. {}, {} ".format(fp, xlab, args.dataset), fontsize=24, pad=18)

        if corr_cov_mean_within[f]:
            ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(corr_simple_Mg_covcov[f]))
        ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(tau_simple_Mg_covcov[f]))

        ax.legend()
        plt.savefig("{}/Simple_Mg_CovCov_{}_{}_{}_{}_{}.png".format(summarytarg, xlab,args.dataset, args.proto_layer, fp, getdt))
        plt.close()


        #mean correlation within
        
        fig = plt.figure()
        ax = fig.add_subplot()
        #ax2 = ax.twinx()

        #xlabel = "Generalization Gap (0-1 Loss)" 
        
        color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
        list_indices =  [j for j in range(len(gap_comp_lists))]

        for idx, x, y, c in zip(list_indices, comp_lists, simple_Mg_corrcorr_comp_lists[f], color):
            #plt.scatter(inds, y,s=10,  color=c)
            #plt.bar(inds,y,width=1, color=c)
            ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
            #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

        ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
        ax.set_ylabel('Risk', fontsize=24, labelpad=18)
        #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
        ax.set_title("Correlation Risk Top {} Ftr vs. {}, {} ".format(fp, xlab, args.dataset), fontsize=24, pad=18)

        if corr_corr_mean_within[f]:
            ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(corr_corr_mean_within[f]))
        ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(tau_corr_mean_within[f]))

        ax.legend()
        plt.savefig("{}/Simple_Mg_CorrCorr_{}_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, fp, getdt))
        plt.close()



        fig = plt.figure()
        ax = fig.add_subplot()
        #ax2 = ax.twinx()

        #xlabel = "Generalization Gap (0-1 Loss)" 
        
        color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
        list_indices =  [j for j in range(len(gap_comp_lists))]

        for idx, x, y, c in zip(list_indices, comp_lists, risk_cov_l2_comp_lists[f], color):
            #plt.scatter(inds, y,s=10,  color=c)
            #plt.bar(inds,y,width=1, color=c)
            ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
            #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

        ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
        ax.set_ylabel('Risk', fontsize=24, labelpad=18)
        #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
        ax.set_title("Cov/L2 Risk Top {} Ftr vs. {}, {} ".format(fp, xlab, args.dataset), fontsize=24, pad=18)


        ax.legend()
        plt.savefig("{}/Risk_CovL2_{}_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, fp, getdt))
        plt.close()


        fig = plt.figure()
        ax = fig.add_subplot()
        #ax2 = ax.twinx()

        #xlabel = "Generalization Gap (0-1 Loss)" 
        
        color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
        list_indices =  [j for j in range(len(gap_comp_lists))]

        for idx, x, y, c in zip(list_indices, comp_lists, risk_covunit_cs_comp_lists[f], color):
            #plt.scatter(inds, y,s=10,  color=c)
            #plt.bar(inds,y,width=1, color=c)
            ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
            #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

        ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
        ax.set_ylabel('Risk', fontsize=24, labelpad=18)
        #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
        ax.set_title("CovUnit/CS Risk Top {} Ftr vs. {}, {} ".format(fp, xlab, args.dataset), fontsize=24, pad=18)


        ax.legend()
        plt.savefig("{}/Risk_CovUnitCS_{}_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, fp, getdt))
        plt.close()


        fig = plt.figure()
        ax = fig.add_subplot()
        #ax2 = ax.twinx()

        #xlabel = "Generalization Gap (0-1 Loss)" 
        
        color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
        list_indices =  [j for j in range(len(gap_comp_lists))]

        for idx, x, y, c in zip(list_indices, comp_lists, risk_l2_cov_comp_lists[f], color):
            #plt.scatter(inds, y,s=10,  color=c)
            #plt.bar(inds,y,width=1, color=c)
            ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
            #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

        ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
        ax.set_ylabel('Risk', fontsize=24, labelpad=18)
        #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
        ax.set_title("L2/Cov Risk Top {} Ftr vs. {}, {} ".format(fp, xlab, args.dataset), fontsize=24, pad=18)


        ax.legend()
        plt.savefig("{}/Risk_L2Cov_{}_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, fp, getdt))
        plt.close()


        fig = plt.figure()
        ax = fig.add_subplot()
        #ax2 = ax.twinx()

        #xlabel = "Generalization Gap (0-1 Loss)" 
        
        color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
        list_indices =  [j for j in range(len(gap_comp_lists))]

        for idx, x, y, c in zip(list_indices, comp_lists, risk_cs_covunit_comp_lists[f], color):
            #plt.scatter(inds, y,s=10,  color=c)
            #plt.bar(inds,y,width=1, color=c)
            ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
            #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

        ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
        ax.set_ylabel('Risk', fontsize=24, labelpad=18)
        #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
        ax.set_title("CS/CovUnit Risk Top {} Ftr vs. {}, {} ".format(fp, xlab, args.dataset), fontsize=24, pad=18)


        ax.legend()
        plt.savefig("{}/Risk_CSCovUnit_{}_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, fp, getdt))
        plt.close()



        fig = plt.figure()
        ax = fig.add_subplot()
        #ax2 = ax.twinx()

        #xlabel = "Generalization Gap (0-1 Loss)" 
        
        color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
        list_indices =  [j for j in range(len(gap_comp_lists))]

        for idx, x, y, c in zip(list_indices, comp_lists, risk_covabs_l2_comp_lists[f], color):
            #plt.scatter(inds, y,s=10,  color=c)
            #plt.bar(inds,y,width=1, color=c)
            ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
            #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

        ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
        ax.set_ylabel('Risk', fontsize=24, labelpad=18)
        #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
        ax.set_title("AbsCov/L2 Risk Top {} Ftr vs. {}, {} ".format(fp, xlab, args.dataset), fontsize=24, pad=18)


        ax.legend()
        plt.savefig("{}/Risk_AbsCovL2_{}_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, fp, getdt))
        plt.close()



        fig = plt.figure()
        ax = fig.add_subplot()
        #ax2 = ax.twinx()

        #xlabel = "Generalization Gap (0-1 Loss)" 
        
        color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
        list_indices =  [j for j in range(len(gap_comp_lists))]

        for idx, x, y, c in zip(list_indices, comp_lists, risk_covunitabs_cs_comp_lists[f], color):
            #plt.scatter(inds, y,s=10,  color=c)
            #plt.bar(inds,y,width=1, color=c)
            ax.plot(x, y, 'o', color=c, label=hyper_labels[idx])
            #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

        ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
        ax.set_ylabel('Risk', fontsize=24, labelpad=18)
        #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
        ax.set_title("AbsUnitCov/CS Risk Top {} Ftr vs. {}, {} ".format(fp, xlab, args.dataset), fontsize=24, pad=18)


        ax.legend()
        plt.savefig("{}/Risk_AbsCovUnitCS_{}_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, fp, getdt))
        plt.close()


    #############################################


    fig = plt.figure()
    ax = fig.add_subplot()
    #ax2 = ax.twinx()

    #xlabel = "Generalization Gap (0-1 Loss)" 
    
    color = cm.rainbow(np.linspace(0, 1, len(gap_comp_lists)))
    list_indices =  [j for j in range(len(gap_comp_lists))]

    for idx, x, y, c in zip(list_indices, comp_lists, Madv_comp_lists, color):
        #plt.scatter(inds, y,s=10,  color=c)
        #plt.bar(inds,y,width=1, color=c)
        ax.plot(x, y,'o', color=c, label=hyper_labels[idx])
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")

    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel('Madv', fontsize=24, labelpad=18)
    #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)
    ax.set_title("Madv vs. {} for Different Learning Algorithms, {} ".format(xlab, args.dataset), fontsize=24, pad=18)

    ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(Madv_corr))
    ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(Madv_tau))

    ax.legend()
    plt.savefig("{}/Madv_{}_{}_{}_{}.png".format(summarytarg, xlab, args.dataset, args.proto_layer, getdt))
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot()
    #ax2 = ax.twinx()                                          
    xlabel = "Test Accuracy Actual"

    color = cm.rainbow(np.linspace(0, 1, len(NCM_test_acc_actual)))
    list_indices =  [j for j in range(len(NCM_test_acc_actual))]

    for idx, x, y, c in zip(list_indices, NCM_test_acc_actual, NCM_test_acc, color):
        #plt.scatter(inds, y,s=10,  color=c)                                                                                                                                                      
    #plt.bar(inds,y,width=1, color=c)                                                                         
        ax.plot(x, y,'o', color=c, label=hyper_labels[idx])
        #ax2.plot(xdata, y2, color=c, label=lab, linestyle="--")                                                                                             

    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel('NCM Test Acc', fontsize=24, labelpad=18)
    #ax2.set_ylabel('Variance', fontsize=24, labelpad=36, rotation=270)                                             
    ax.set_title("NCM Test Acc vs. Actual Test Acc for Different Learning Algorithms, {} ".format(args.dataset), fontsize=24, pad=18)

    ax.plot([], [], ' ', label="\u03C1: {0:3.3f}".format(Madv_corr))
    ax.plot([], [], ' ', label="\u03C4: {0:3.3f}".format(Madv_tau))

    ax.legend()
    plt.savefig("{}/NCM_Test_{}_{}_{}.png".format(summarytarg, args.dataset, args.proto_layer, getdt))
    plt.close()


if __name__ == '__main__':
    main()

