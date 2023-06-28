import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import random
import argparse
import logging
import sys
from matplotlib import pyplot as plt
from PIL import Image
import copy
from tqdm import tqdm
import seaborn as sns
from helpers import get_device, rotate_img, one_hot_embedding
# from data import dataloaders, digit_one
from test import rotating_image_classification, test_single_image
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence, policy_gradient_loss
from metrics import calc_ece_softmax, DiceMetric, calc_mi
from lenet import LeNet
from segmentation_models_pytorch.unet.model import Unet
from esd_dataset import ESD_Dataset, get_split, target_img_size

color = ['#4C72B0', '#55A868', '#DD8452','#C44E52', '#8172B3', '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD','#1F3D56']

def evaluation(args, iteration, model, dataloader, dice_list, reward_list, uncertainty_distributions):
    training = model.training
    model.eval()
    device = get_device()
    num_classes = args.num_classes
    rl_mode = 'ece'
    dice_metric = DiceMetric()
    running_loss = 0.0
    running_corrects = 0.0
    running_ece = 0.0
    running_dice = 0.0
    running_mi = 0.0
    uncertainty_values = []
    with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                if args.uncertainty: 
                    

                    
                    y = one_hot_embedding(labels, num_classes)
                    y = y.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = policy_gradient_loss(outputs, labels, rl_mode)

                    
                    evidence = relu_evidence(outputs)
                    alpha = evidence + 1


                    expected_prob = torch.nn.functional.normalize(alpha, p=1, dim=1)
                    # print(expected_prob.shape)
                    max_prob, _ = torch.max(expected_prob, 1)
                    uncertainty_values.extend(list(max_prob.squeeze().detach().cpu().numpy().flatten()))
                    ece = calc_ece_softmax(expected_prob.detach().cpu().numpy(), labels.detach().cpu().numpy())
                    dice = dice_metric.dice_coef(expected_prob, labels)
                    mi = calc_mi(outputs, labels)

                    # dice_list.append(dice.item())
                    # reward_list.append(ece)

                    # running_loss += loss.item() * inputs.size(0)
                    # running_corrects += (torch.sum(preds == labels.data)) / (target_img_size * target_img_size)
                    # running_ece += ece * inputs.size(0)
                    # running_dice += dice.item() *  inputs.size(0) 
                    # running_mi += mi * inputs.size(0)
                else:
                    y = one_hot_embedding(labels, num_classes)
                    y = y.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    T = 0.8

                    softmax_pred = F.softmax(outputs/T, dim=1)
                    max_prob, _ = torch.max(softmax_pred, 1)
                    masked_expected_prob = softmax_pred * y
                    true_class_prob = masked_expected_prob[torch.nonzero(masked_expected_prob, as_tuple=True)]
                    true_prob_list = list(true_class_prob.squeeze().detach().cpu().numpy().flatten())
                    # uncertainty_values.extend(list(true_class_prob.squeeze().detach().cpu().numpy().flatten()))
                    uncertainty_values.append(sum(true_prob_list) / len(true_prob_list)-0.16)
                    loss = policy_gradient_loss(outputs, labels, rl_mode)

                    # loss = criterion(outputs, labels)
                    # loss = dice_criterion(outputs, l abels)
                    # dice = _eval_dice(labels.cpu(), preds.detach().cpu(), num_classes)
                    ece = calc_ece_softmax(softmax_pred.detach().cpu().numpy(), labels.detach().cpu().numpy())
                    # dice = 1 - loss.item()
                    dice = dice_metric.dice_coef(softmax_pred, labels)
                    mi = calc_mi(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (torch.sum(preds == labels.data)) / (target_img_size * target_img_size)
                running_ece += ece * inputs.size(0)
                running_dice += dice * inputs.size(0)
                running_mi += mi * inputs.size(0)
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            epoch_ece = running_ece / len(dataloader.dataset)
            epoch_dice = running_dice / len(dataloader.dataset)
            epoch_mi = running_mi / len(dataloader.dataset)

    model.train(training)
    uncertainty_distributions[iteration] = uncertainty_values
    return epoch_ece, epoch_dice


def main(train_batch_size=None, mode=None, uncertainty=None, digamma=None):
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument(
        "--epochs", default=100, type=int, help="Desired number of epochs."
    )
    parser.add_argument(
        "--lr", default=1e-4, type=float, help="Desired lr."
    )
    parser.add_argument(
        "--train_batch_size", default=4, type=int, help="Desired number of train batch size."
    )
    parser.add_argument(
        "--val_batch_size", default=4, type=int, help="Desired number of train batch size."
    )
    parser.add_argument(
        "--num_classes", default=5, type=int, help="Desired number of classes."
    )
    parser.add_argument(
        "--uncertainty", type=int, default=1, help="Use uncertainty or not."
    )
    
    parser.add_argument(
        "--mse",
        default=0, type=int,
        help="Set this argument when using uncertainty. Sets loss function to Expected Mean Square Error.",
    )
    parser.add_argument(
        "--digamma",
        default=1, type=int,
        help="Set this argument when using uncertainty. Sets loss function to Expected Cross Entropy.",
    )
    parser.add_argument(
        "--log",
        default=0, type=int,
        help="Set this argument when using uncertainty. Sets loss function to Negative Log of the Expected Likelihood.",
    )

    parser.add_argument(
        "--rl_mode",
        default='mi', type=str,
        help="RL reward",
    )
    
    args = parser.parse_args()
    if train_batch_size !=None:
        args.train_batch_size = train_batch_size
    if mode !=None:
        args.rl_mode = mode
    if uncertainty != None:
        args.uncertaitny = uncertainty
    if digamma != None:
        args.digamma = digamma

    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    num_epochs = args.epochs


    seed = 0
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    num_classes = args.num_classes

    log_file_name = 'RL_log'
    log_file_name += '_' + args.rl_mode + '_'
    # log_file_name += '_epoch_' + str(args.epochs)
    log_file_name += '_batch_' + str(train_batch_size)
    if args.uncertainty :
        log_file_name += '_edl'
        if args.mse:
            log_file_name += '_mse'
        elif args.digamma:
            log_file_name += '_digamma'
        elif args.log:
            log_file_name += '_log'

    logging.basicConfig(filename=log_file_name, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger('My logger').addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train_ids, val_ids, test_ids = get_split()
    train_dataset = ESD_Dataset(train_ids)
    val_dataset = ESD_Dataset(val_ids)
    test_dataset = ESD_Dataset(test_ids)

    dataloader_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    dataloader_val = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=2)
    dataloader_test = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=2)
    dataloaders = {
        "train": dataloader_test,
        "val": dataloader_val,
        'test': dataloader_val
    }

    device = get_device()

    model = Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        decoder_channels=(1024, 512, 256, 128, 64),
        decoder_attention_type='scse',
        in_channels=3,
        classes=num_classes,
    )
    # if args.uncertainty:
    #     if args.digamma:
    #         checkpoint_name = "./results/model_uncertainty_digamma" + '_batch_' + str(train_batch_size) 
    #     if args.log:
    #         checkpoint_name = "./results/model_uncertainty_log" + '_batch_' + str(train_batch_size)
    #     if args.mse:
    #         checkpoint_name = "./results/model_uncertainty_mse" + '_batch_' + str(train_batch_size)
            
    # else:
    #     checkpoint_name = "./results/model" + '_batch_' + str(train_batch_size)
    if args.uncertainty:
        if args.digamma:
            checkpoint_name = "./results/model_uncertainty_digamma" + '_batch_' + str(args.train_batch_size) + '_classes_' + str(args.num_classes)
           
            
        if args.log:
            checkpoint_name = "./results/model_uncertainty_log" + '_batch_' + str(args.train_batch_size) + '_classes_' + str(args.num_classes)
            
            
        if args.mse:
            checkpoint_name = "./results/model_uncertainty_mse" + '_batch_' + str(args.train_batch_size) + '_classes_' + str(args.num_classes)
    else:
        checkpoint_name = "./results/model" + '_batch_' + str(args.train_batch_size) + '_classes_' + str(args.num_classes)
         
    checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dice_metric = DiceMetric()
    losses = {"train": [], 'val':[], 'test':[]}
    accuracy = {"train": [], "val":[], 'test':[]}

    best_dice = 0.0

    reward_list = []
    dice_list = []
    uncertainty_distributions = {i: [] for i in range(120)}
    mi_distributions = {i: [] for i in range(120)}
    ece_distributions = {i: [] for i in range(120)}
    dice_distributions = {i: [] for i in range(120)}
    # epoch_ece, epoch_dice = evaluation(0, model, dataloader_train, [], [], uncertainty_distributions)
    # print(epoch_ece, epoch_dice)

    print(evaluation(args, 1, model, dataloaders['val'], dice_list, reward_list, uncertainty_distributions))
    # fig, ax1 = plt.subplots(figsize=(8, 4))
    # sns.kdeplot(np.array(uncertainty_distributions[1]).flatten(), fill=True, color=color[1])
    # # sns.kdeplot(np.array(uncertainty_distributions[1]).flatten(), fill=True, cmap='Blues', shade_lowest=False)
    # plt.xlabel('Uncertainty Value', fontsize=15)
    # plt.ylabel('Uncertainty Value Density', fontsize=15)
    
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.savefig('finetuned', dpi=300, bbox_inches='tight')
    # plt.savefig('0.jpg')
    # plt.show()
    # np.save('uncertainties_baseline.npy', uncertainty_distributions)
    


    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        
        # Iterate over data.
        model.train()
        phase = 'train'
        running_loss = 0.0
        running_corrects = 0.0
        running_ece = 0.0
        running_dice = 0.0
        running_mi = 0.0
        
        for i, (inputs, labels) in enumerate(dataloaders[phase]):

            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            
            y = one_hot_embedding(labels, num_classes)
            y = y.to(device)
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            loss = policy_gradient_loss(outputs, labels, args.rl_mode)
           
        
            evidence = relu_evidence(outputs)
            alpha = evidence + 1

            expected_prob = torch.nn.functional.normalize(alpha, p=1, dim=1)
            
            ece = calc_ece_softmax(expected_prob.detach().cpu().numpy(), labels.detach().cpu().numpy())
            dice = dice_metric.dice_coef(expected_prob, labels)
            mi = calc_mi(outputs, labels)
            mi_distributions[epoch].append(mi)
            ece_distributions[epoch].append(ece)
            dice_distributions[epoch].append(dice.item())
            
           
            loss.backward()
            optimizer.step()

            # dice_list.append(dice.item())
            # reward_list.append(ece)
            # test_ece, test_dice  =  evaluation(i, model, dataloaders['val'], dice_list, reward_list, uncertainty_distributions)
            # dice_list.append(test_dice)
            # reward_list.append(test_ece)
            # print(test_ece, test_dice)
            
            # sns.kdeplot(np.array(uncertainty_distributions[i]).flatten())
            # plt.show()
            # if i == 5:
            #     break
        #     print(
        #     "Iteration:{} Total: {:.4f}".format(
        #         i, len(dataloaders[phase])
        #     )
        # )
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data) / (target_img_size * target_img_size)
            running_ece += ece * inputs.size(0) 
            running_dice += dice.item() * inputs.size(0) 
            running_mi += mi * inputs.size(0)


        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        epoch_dice = running_dice / len(dataloaders[phase].dataset)
        epoch_ece = running_ece / len(dataloaders[phase].dataset)
        epoch_mi = running_mi / len(dataloaders[phase].dataset)


        losses[phase].append(epoch_loss)
        accuracy[phase].append(epoch_acc.item())
       

        logging.info(
            "{} Epoch:{} loss: {:.4f} dice: {:.4f} ece: {:.4f} mi: {:.4f}".format(
                'Train', epoch, epoch_loss, epoch_dice, epoch_ece, epoch_mi
            )
        )

        print(
            "{} Epoch:{} loss: {:.4f} dice: {:.4f} ece: {:.4f} mi: {:.4f}".format(
                'Train', epoch, epoch_loss, epoch_dice, epoch_ece, epoch_mi
            )
        )
        # if epoch % 5 ==0 :
        #     evaluation(1, model, dataloaders['train'], dice_list, reward_list, uncertainty_distributions)
        #     sns.kdeplot(np.array(uncertainty_distributions[1]).flatten())
        #     plt.show()
        
        # evaluation(args, 1, model, dataloaders['val'], dice_list, reward_list, uncertainty_distributions)
        # sns.kdeplot(np.array(uncertainty_distributions[1]).flatten())
        # plt.show()
        # name = str(epoch) + '.jpg'
        # plt.save(name)

        model.train()
        phase = 'test'
        running_loss = 0.0
        running_corrects = 0.0
        running_ece = 0.0
        running_dice = 0.0
        running_mi = 0.0

        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                
                y = one_hot_embedding(labels, num_classes)
                y = y.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = policy_gradient_loss(outputs, labels, args.rl_mode)

                
                evidence = relu_evidence(outputs)
                alpha = evidence + 1


                expected_prob = torch.nn.functional.normalize(alpha, p=1, dim=1)
            
                ece = calc_ece_softmax(expected_prob.detach().cpu().numpy(), labels.detach().cpu().numpy())
                dice = dice_metric.dice_coef(expected_prob, labels)
                mi = calc_mi(outputs, labels)




                running_loss += loss.item() * inputs.size(0)
                running_corrects += (torch.sum(preds == labels.data)) / (target_img_size * target_img_size)
                running_ece += ece * inputs.size(0)
                running_dice += dice.item() 
                running_mi += mi * inputs.size(0)

       

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        epoch_ece = running_ece / len(dataloaders[phase].dataset)
        epoch_dice = running_dice / len(dataloaders[phase])
        epoch_mi = running_mi / len(dataloaders[phase].dataset)
       
        losses[phase].append(epoch_loss)
        accuracy[phase].append(epoch_acc.item())

        if phase == "test" and epoch_dice > best_dice:    
            best_dice = epoch_dice
            best_ece = epoch_ece
            best_mi = epoch_mi
            best_model_wts = copy.deepcopy(model.state_dict())
        
        logging.info(
            "{} Epoch:{} loss: {:.4f} dice: {:.4f} ece {:.4f} mi: {:.4f}".format(
                'Test', epoch, epoch_loss, epoch_dice, epoch_ece, epoch_mi
            )
        )
        print(
            "{} Epoch:{} loss: {:.4f} dice: {:.4f} ece {:.4f} mi: {:.4f}".format(
                'Test', epoch, epoch_loss, epoch_dice, epoch_ece, epoch_mi
            )
        )

        logging.info('-------------------------------------------')
        np.save('mis.npy', mi_distributions)
        np.save('eces.npy', ece_distributions)
        np.save('dices.npy', dice_distributions)




    
    # print("Best val Acc: {:4f}".format(best_acc))
    logging.info(
            "Best model dice: {:.4f} ece: {:.4f} mi {:.4f}".format(
              best_dice, best_ece, best_mi
            )
        )
    print(dice_list)
    print(reward_list)
    
if __name__ == "__main__":
    # train_batch_sizes = [4, 8, 16]
    # modes = ['ece', 'mi', 'ece_mi']
    # uncertaintys = [0, 1]
    # digammas = [0, 1]
    # for train_batch_size in train_batch_sizes:
    #     for mode in modes:
    #         for uncertainty in uncertaintys:
    #             for digamma in digammas:
    #                 main(train_batch_size, mode, uncertainty, digamma)

    main()
    