# !/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: gaichaoli
# FileName: main_dgl_dataset_node_classification.py
# Remark: 
# version: python 3.8
# Date: 2022/4/9 21:18
# Software: PyCharm 
# ========================================================
"""
    IMPORTING LIBS
"""
import numpy as np
import os
import time
import wandb
import random
import argparse, json
import math
import torch
import torch.nn.functional as F

import torch.optim as optim
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

epsilon = 1 - math.log(2)





"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from load_net import gnn_model
from dataset import LoadData
from dataset import DataLoaderX

# GPU setup
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # print("gpu_setup_gpu_id: ", gpu_id)
    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda:{}".format(str(gpu_id)))
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


# Viewing model config and params
def view_model_param(MODEL_NAME, net_params, data_params):
    model = gnn_model(MODEL_NAME, net_params, data_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


# Loss function
def custom_loss_function(x, labels):
    # labels = torch.tensor(labels)
    y = F.cross_entropy(x, labels, reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


# Evaluating function
@torch.no_grad()
def evaluate(model, feat, labels, prepro_bias, train_idx, valid_idx, test_idx, device):
    """
    evaluate the perfomance of the model
    """
    model.eval()
    attn_bias, spatial_pos, in_degree, out_degree, attn_edge_type = prepro_bias
    batched_data = {'x':feat.long(),'attn_bias':attn_bias,
                    'spatial_pos':spatial_pos,'in_degree':in_degree,
                    'out_degree':out_degree,'attn_edge_type':attn_edge_type,
                    'edge_input':attn_edge_type}
    kwagrs = {'perturb':None,'masked_tokens':None}
    
    pred = model(batched_data, **kwagrs)
    pred = pred[:,1:,:]
    train_loss = custom_loss_function(pred[train_idx], labels[train_idx])
    valid_loss = custom_loss_function(pred[valid_idx], labels[valid_idx])
    test_loss = custom_loss_function(pred[test_idx], labels[test_idx])

    pred_label = pred.argmax(dim=-1, keepdim=True)
    train_acc = accuracy_score(labels[train_idx].cpu().numpy(), pred_label[train_idx].cpu().numpy())
    valid_acc = accuracy_score(labels[valid_idx].cpu().numpy(), pred_label[valid_idx].cpu().numpy())
    test_acc = accuracy_score(labels[test_idx].cpu().numpy(), pred_label[test_idx].cpu().numpy())
    return (
        train_acc,
        valid_acc,
        test_acc,
        train_loss,
        valid_loss,
        test_loss,
        pred
    )


def train_epoch(model, feat, prepro_bias, optimizer, device):
    """"""
    model.train()
    optimizer.zero_grad()
    attn_bias, spatial_pos, in_degree, out_degree, attn_edge_type = prepro_bias
    batched_data = {'x':feat.long(),'attn_bias':attn_bias,
                    'spatial_pos':spatial_pos,'in_degree':in_degree,
                    'out_degree':out_degree,'attn_edge_type':attn_edge_type,
                    'edge_input':attn_edge_type}
    kwagrs = {'perturb':None,'masked_tokens':None}
    pred = model(batched_data, **kwagrs)

    return pred


# Training code
def train_val_pipeline(MODEL_NAME, dataset, params, net_params, data_params, dirs, device):
    start0 = time.time()
    per_epoch_time = []
    DATASET_NAME = dataset.dataset_name
    # root dirs to save logs, checkpoints and other files
    root_log_dir, root_checkpoint_dir, write_file_name, write_config_file = dirs
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    # Write network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format(
            DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
    writer = SummaryWriter(log_dir=log_dir)

    # instantiate a gnn model
    model = gnn_model(MODEL_NAME, net_params, data_params)
    # print("cuda_visible_devices: ", os.environ["CUDA_VISIBLE_DEVICES"])
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(2)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    total_time = 0
    best_val_acc, best_val_loss = 0, float("inf")
    best_test_acc, corresponding_valid_acc, best_test_loss = 0, 0, float("inf")
    final_pred = None

    accs, train_accs, valid_accs, test_accs = [], [], [], []
    losses, train_losses, valid_losses, test_losses = [], [], [], []
    dataloader = DataLoaderX(dataset=dataset, shuffle=False)
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        # with tqdm(range(params['epochs'])) as t:
        for epoch in range(params['epochs']):
            # for epoch in t:
            for batch_data in tqdm(dataloader):
                # t.set_description('Epoch %d' % epoch)
                batch_data_cuda = [item.to(device) for item in batch_data]
                feat, labels, train_mask, valid_mask, test_mask, \
                attn_bias, spatial_pos, in_degree, out_degree, attn_edge_type = batch_data_cuda
                start_time = time.time()
                prepro_bias = (attn_bias, spatial_pos, in_degree, out_degree, attn_edge_type)
                pred = train_epoch(model, feat, prepro_bias, optimizer, device)
                pred = pred[:,1:,:]
                loss = custom_loss_function(pred[train_mask], labels[train_mask])
                loss.backward()
                optimizer.step()

                pred_label = pred[train_mask].argmax(dim=-1, keepdim=True).flatten()
                label_train = labels[train_mask]
                acc = accuracy_score(label_train.cpu().numpy(), pred_label.cpu().numpy())

                train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, pred = evaluate(model, feat,
                                                                                                   labels,
                                                                                                   prepro_bias,
                                                                                                   train_mask,
                                                                                                   valid_mask,
                                                                                                   test_mask, device)

                end_time = time.time()
                per_epoch_time.append(end_time - start_time)
                total_time += end_time - start_time
                # t.set_postfix(time=end_time - start_time, lr=optimizer.param_groups[0]['lr'])

                wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc})
                print(
                    f"Average_epoch_time:{total_time / (epoch + 1):.2f} Loss:{loss:.4f} Acc:{acc:.4f} "
                    f"Train_Loss:{train_loss:.4f} Val_Loss:{valid_loss:.4f} Test_Loss:{test_loss:.4f} "
                    f"Train_Acc:{train_acc:.4f} Val_Acc:{valid_acc:.4f} Test_Acc:{test_acc:.4f}"
                )

                if valid_acc > best_val_acc:
                    best_val_acc = valid_acc
                    final_test_acc = test_acc
                    # final_pred = pred

                    # save checkpoint
                    checkpoint_dir = os.path.join(root_checkpoint_dir, "RUN_")
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    torch.save(model.state_dict(), "{}.pkl".format(checkpoint_dir + "/epoch_" + str(epoch)))

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    corresponding_valid_acc = valid_acc

                for record, item in zip(
                        [accs, train_accs, valid_accs, test_accs, losses, train_losses, valid_losses, test_losses],
                        [acc, train_acc, valid_acc, test_acc, loss, train_loss, valid_loss, test_loss]
                ):
                    record.append(item)

                writer.add_scalar('train_loss', train_loss, epoch)
                writer.add_scalar('valid_loss', valid_loss, epoch)
                writer.add_scalar('train_acc', train_acc, epoch)
                writer.add_scalar('valid_acc', valid_acc, epoch)
                writer.add_scalar('test_acc', test_acc, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                # files = glob.glob(checkpoint_dir + '/*.pkl')
                # for file in files:
                #     epoch_nb = file.split('_')[-1]
                #     epoch_nb = int(epoch_nb.split('.')[0])
                #     if epoch_nb < epoch - 1:
                #         os.remove(file)

                scheduler.step(valid_loss)  # adjust learning rate

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
                    break

                # Stop training after params['max_time'] hours
                # if time.time() - start0 > params['max_time'] * 3600:
                #     print('-' * 89)
                #     print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                #     break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    print("Best Validation Accuracy: {:.4f}".format(corresponding_valid_acc))
    print("Best Test Accuracy: {:.4f}".format(best_test_acc))
    print("Train Accuracy: {:.4f}".format(train_acc))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - start0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nBEST TEST ACCURACY: {:.4f}\nCORRESPONDING VALIDATION ACCURACY: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n""" \
                .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                        best_test_acc, corresponding_valid_acc, epoch, (time.time() - start0) / 3600,
                        np.mean(per_epoch_time)))
    wandb.log({"best_test_acc": best_test_acc, "best_validation_acc": best_val_acc,
               "total_time": (time.time() - start0) / 3600})
    # net_params["best_test_acc_list"].append(best_test_acc)
    # print("BEST TEST ACC LIST: %s" % str(net_params["best_test_acc_list"]))
    wandb.watch(model)


def main():
    """
        USER CONTROLS
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/cora_Graphormer.json",
                        help="Please give a config.json file with training/model/data/param details")
    args = parser.parse_args()
    
    print(os.getcwd())
    with open(args.config) as f:
        config = json.load(f)

    # device
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    MODEL_NAME = config['model']
    DATASET_NAME = config['dataset']
    
    # out_dir
    out_dir = config['out_dir'] + DATASET_NAME + "/"
    
    # parameters
    params = config['params']

    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']

    # data parameters
    data_params = config['data_params']
    data_params['dataset_name'] = DATASET_NAME
    
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_checkpoint_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_final_name = out_dir + "results/final_result"
    dirs = root_log_dir, root_checkpoint_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    # net_params["best_test_acc_list"] = []

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    # setting cuda seed
    device = net_params['device']
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    dataset = LoadData(DATASET_NAME, graph_file_num=1)
    # net_params["origin_dim"] = dataset.feat_dim
    # net_params["in_dim"] = net_params["hidden_dim"] lxl
    data_params["num_classes"] = dataset.num_classes
    data_params["spatial_pos_max"] = dataset.spatial_pos_max
    data_params["num_in_degree"] = dataset.num_in_degree
    data_params["num_out_degree"] = dataset.num_out_degree
    data_params['num_atoms'] = dataset.num_atoms  #节点数量
    data_params['num_edges'] = 512*3  #暂时不知道干嘛的
    data_params['max_nodes'] = 3000  #最多支持节点的数量
    
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params, data_params)

    # MODEL_NAME: GraphTransformer, dataset: arxiv, params: dict,
    print("init_lr = %s" % params["init_lr"])
    # print("layer L = %s" % net_params["L"]) #lxl
    wandb.init(project="graphormer", entity="ciqu0518", name=DATASET_NAME, config=args)

    # wandb.config(net_params)
    wandb.config = net_params
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, data_params, dirs, device)


# with open(write_final_name + '.txt', 'w') as f:
#     f.write("""BEST_TEST_ACC_LIST: {}\nMEAN_BEST_ACC: {}\n"""
#             .format(net_params["best_test_acc_list"],
#                     sum(net_params["best_test_acc_list"]) / len(net_params["best_test_acc_list"])))


if __name__ == '__main__':
    main()
    # from torch_geometric.datasets import Planetoid
    # dataset = Planetoid(root='data/dataset_de/cora', name='cora')

