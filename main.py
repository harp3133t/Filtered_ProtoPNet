import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
import model

import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

import random
import numpy as np
import copy


import time
from util.helpers import makedir
import push_trivial, push_support, model, train_and_test as tnt
from util import save
from util.log import create_logger
from util.preprocess import mean, std, preprocess_input_function
import settings_CUB_DOG

import wandb
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(41)
random.seed(41)
np.random.seed(41)


parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0,1,2,3') # python3 main.py -gpuid=0,1,2,3
parser.add_argument('-project', type=str, default='Adjust ratio Support Trivial')
parser.add_argument('-name')
parser.add_argument('-exp_num')
parser.add_argument('-trivial_num')
parser.add_argument('-support_num')
args = parser.parse_args()

wandb.init(
    # set the wandb project where this run will be logged
    project=args.project,#"ProtoPNet_full",
    name = args.name,
    # track hyperparameters and run metadata
)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
###############################################################################


#setting parameter
experiment_run = settings_CUB_DOG.experiment_run
base_architecture = 'vgg19'
dataset_name = 'bone'

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

model_dir ='saved_models/{}/'.format(datestr()) + base_architecture + '/' + experiment_run + '/'

if os.path.exists(model_dir) is True:
    shutil.rmtree(model_dir)
makedir(model_dir)


shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings_CUB_DOG.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'models', base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'push_trivial.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'push_support.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), './util/helpers.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'


#model param
num_classes = settings_CUB_DOG.num_classes
img_size = settings_CUB_DOG.img_size
add_on_layers_type = settings_CUB_DOG.add_on_layers_type
prototype_shape = (int(args.trivial_num) + int(args.support_num), 64, 1, 1)#settings_CUB_DOG.prototype_shape
prototype_activation_function = settings_CUB_DOG.prototype_activation_function

#datasets
train_batch_size = settings_CUB_DOG.train_batch_size
test_batch_size = settings_CUB_DOG.test_batch_size
train_push_batch_size = settings_CUB_DOG.train_push_batch_size
data_path = '../data/full/'
train_dir = data_path + 'train_augmented/'
test_dir = data_path + 'test/'
train_push_dir = data_path + 'train/'




# weighting of different training losses
coefs = settings_CUB_DOG.coefs
# number of training epochs, number of warm epochs, push start epoch, push epochs
num_train_epochs = settings_CUB_DOG.num_train_epochs
num_warm_epochs = settings_CUB_DOG.num_warm_epochs
push_start = settings_CUB_DOG.push_start
push_epochs = settings_CUB_DOG.push_epochs


log(train_dir)

normalize = transforms.Normalize(mean=mean, std=std)

# all datasets
# train set
num_workers = 2  # 20
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.RandomAffine(degrees=(-25, 25), shear=15),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=False)
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=False)
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

log("backbone architecture:{}".format(base_architecture))
log("prototype shape:{}".format(prototype_shape))
# construct the model
ppnet = model.construct_STProtoPNet(base_architecture=base_architecture,
                                    pretrained=True, img_size=img_size,
                                    prototype_shape=prototype_shape,
                                    num_classes=num_classes,
                                    prototype_activation_function=prototype_activation_function,
                                    add_on_layers_type=add_on_layers_type,
                                    threshold = 0.1,
                                    trivial = int(args.trivial_num),
                                    support = int(args.support_num)
                                    )
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)


class_specific = True


if dataset_name == "CUB":
    weight_decay_factor = 0.1  # CUB
else:
    weight_decay_factor = 0.5  # DOG


# define optimizer
from settings_CUB_DOG import joint_optimizer_lrs, joint_lr_step_size

if base_architecture == 'resnet50' or dataset_name == "DOG":
    joint_optimizer_lrs['features'] = 1e-5

joint_optimizer_specs = \
[
 {'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': weight_decay_factor*1e-3},  # bias are now also being regularized
 {'params': ppnet.add_on_layers_trivial.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': weight_decay_factor*1e-3},
 {'params': ppnet.add_on_layers_support.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': weight_decay_factor*1e-3},
 {'params': ppnet.prototype_vectors_trivial, 'lr': joint_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.prototype_vectors_support, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
if dataset_name == 'CUB':
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=int(2 * joint_lr_step_size), gamma=0.2)  # CUB
else:
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=int(1 * joint_lr_step_size), gamma=0.2)  # DOG


from settings_CUB_DOG import warm_optimizer_lrs
warm_optimizer_specs = \
[
 {'params': ppnet.add_on_layers_trivial.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': weight_decay_factor*1e-3},
 {'params': ppnet.add_on_layers_support.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': weight_decay_factor*1e-3},
 {'params': ppnet.prototype_vectors_trivial, 'lr': warm_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.prototype_vectors_support, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)


from settings_CUB_DOG import last_layer_optimizer_lr
last_layer_optimizer_specs = \
[
 {'params': ppnet.last_layer_trivial.parameters(), 'lr': last_layer_optimizer_lr},
 {'params': ppnet.last_layer_support.parameters(), 'lr': last_layer_optimizer_lr},
]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

#best acc
best_acc = 0
best_epoch = 0
best_time = 0

# train the model
log('start training')

for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    #stage 1: Training of CNN backbone and prototypes
    #train
    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _, train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log, wandb_logger = wandb)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        joint_lr_scheduler.step()
        _, train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log, wandb_logger = wandb)

    # test
    accu, test_results = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=log, wandb_logger = wandb)
    if accu > best_acc:
        best_acc = accu
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                    target_accu=0.60, log=log)


    #stage2: prototype projection
    if epoch >= push_start and epoch in push_epochs:
        push_trivial.push_prototypes(
            train_push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function,  # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir + 'trivial',  # if not None, prototypes will be saved here
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        push_support.push_prototypes(
            train_push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function,  # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir + 'support',  # if not None, prototypes will be saved here
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu, test_results = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=log, wandb_logger = wandb)
        if accu > best_acc:
            best_acc = accu
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                        target_accu=0.60, log=log)
    #stage3:  Training of FC layers
        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(10):   # 20
                log('iteration: \t{0}'.format(i))
                _, train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log, wandb_logger = wandb)

                accu, test_results = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log, wandb_logger = wandb)
                if accu > best_acc:
                    best_acc = accu
                    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                                target_accu=0.60, log=log)

####################################################################################
wandb.finish()
logclose()
