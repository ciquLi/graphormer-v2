# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from pkgutil import get_data
import time
import torch
import numpy as np
import torch_geometric
import torch_geometric.datasets
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from torch.utils import data


def LoadData(DATASET_NAME, graph_file_num):
    """
        This function is called in the main_xx.py file 
        returns:
        ; dataset object
    """
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ogbn-arxiv':
        return ""
    else:
        return myDataset(DATASET_NAME, graph_file_num)
    
class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
                     torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


# def create_graph(dataset_list):
#     """
#     使用pyg来构图
#     :return:
#     """
#     adj, features, labels = dataset_list[0], dataset_list[1], dataset_list[2]
#     u, v = adj._indices()  # 获取稀疏矩阵的索引，即为图的边, 这里得到的图已经是双向图了

#     edge_index = torch.stack((u, v), dim=0)

#     train_idx, valid_idx, test_idx = dataset_list[3], dataset_list[4], dataset_list[5]
#     train_idx, valid_idx, test_idx = torch.tensor(train_idx), torch.tensor(valid_idx), torch.tensor(test_idx)

#     g = torch_geometric.data.Data(x=features, y=labels, edge_index=edge_index, edge_attr=None)
#     num_classes = len(labels.unique())
#     return g, features, labels, train_idx, valid_idx, test_idx, num_classes


def create_graph(dataset_list):
    """
    使用pyg来构图
    :return:
    """
    g = dataset_list
    features = g.x
    labels = g.y
    # edge_index = g.edge_index
    train_idx, valid_idx, test_idx = g.train_mask, g.val_mask, g.test_mask
    
    # adj, features, labels = dataset_list[0], dataset_list[1], dataset_list[2]
    # u, v = adj._indices()  # 获取稀疏矩阵的索引，即为图的边, 这里得到的图已经是双向图了

    # edge_index = torch.stack((u, v), dim=0)

    # train_idx, valid_idx, test_idx = dataset_list[3], dataset_list[4], dataset_list[5]
    # train_idx, valid_idx, test_idx = torch.tensor(train_idx), torch.tensor(valid_idx), torch.tensor(test_idx)

    # g = torch_geometric.data.Data(x=features, y=labels, edge_index=edge_index, edge_attr=None)
    num_classes = len(labels.unique())
    return g, features, labels, train_idx, valid_idx, test_idx, num_classes


class myDataset(data.Dataset):
    def __init__(self, dataset_name, graph_file_num=1):
        """
        Loading planetoid datasets
        """
        self.graph_file_num = graph_file_num
        self.dataset_name = dataset_name
        self.get_data_params(0)
        
    def __getitem__(self, index):
        start = time.time()
        print("[I] Loading dataset %s..." % (self.dataset_name + "_" + str(index)))
        # time.sleep(10)
        # print(os.getcwd())

        dataset_path = "data/dataset_de/" + str(self.dataset_name) + "/de_" + self.dataset_name + "_" + str(
            index) + ".pt"
        dataset_list = torch.load(dataset_path)[0]
   
        self.saved_tensor_path = "data/saved_tensors/" + str(self.dataset_name) + "/de_" + self.dataset_name + "_" + \
                                 str(index) + "/"

        graph, feat, labels, train_mask, valid_mask, test_mask, _ = create_graph(dataset_list)
        attn_bias, spatial_pos, in_degree, out_degree, attn_edge_type = self.preprocess_item(graph)
        
        return feat, labels, train_mask, valid_mask, test_mask, attn_bias, spatial_pos, in_degree, out_degree, attn_edge_type

    def __len__(self):
        return self.graph_file_num
    
    def get_data_params(self,index):
        start = time.time()
        print("[I] Loading dataset %s..." % (self.dataset_name + "_" + str(index)))
        # time.sleep(10)
        # print(os.getcwd())

        dataset_path = "data/dataset_de/" + str(self.dataset_name) + "/de_" + self.dataset_name + "_" + str(
            index) + ".pt"
        dataset_list = torch.load(dataset_path)[0]
   
        self.saved_tensor_path = "data/saved_tensors/" + str(self.dataset_name) + "/de_" + self.dataset_name + "_" + \
                                 str(index) + "/"

        graph, feat, labels, train_mask, valid_mask, test_mask, num_classes = create_graph(dataset_list)

        # print("input_dim: %s" % self.feat_dim)
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

        prepro_start = time.time()
        print("[II] Preprocessing spatial SPD and node centrality...")
        attn_bias, spatial_pos, in_degree, out_degree, attn_edge_type = self.preprocess_item(graph)
        print("[II] Finished preprocessing.")
        print("[II] Preprocessing time: {:.4f}s".format(time.time() - prepro_start))
        
        self.num_atoms = feat.size(0)
        self.feat_dim = feat.size(1)
        self.spatial_pos_max = torch.max(spatial_pos) + 1
        self.num_in_degree = torch.max(in_degree) + 1
        self.num_out_degree = torch.max(out_degree) + 1
        self.num_classes = num_classes
  
    def preprocess_item(self, graph):
        # paths
        attn_bias_path = self.saved_tensor_path + "attn_bias.pt"
        spatial_pos_path = self.saved_tensor_path + "spatial_bias.pt"
        in_degree_path = self.saved_tensor_path + "in_degree.pt"
        out_degree_path = self.saved_tensor_path + "out_degree.pt"
        attn_edge_type_path = self.saved_tensor_path + "attn_edge_type.pt"
        # edge_input_path = self.saved_tensor_path + "edge_input_path.pt"
        # print(os.path.getsize(self.saved_tensor_path))

        if os.path.exists(self.saved_tensor_path) and os.path.getsize(self.saved_tensor_path):
            # loading from saved tensors
            attn_bias = torch.load(attn_bias_path)
            spatial_pos = torch.load(spatial_pos_path)
            in_degree = torch.load(in_degree_path)
            out_degree = torch.load(out_degree_path)
            attn_edge_type = torch.load(attn_edge_type_path)
            # edge_input = torch.load(edge_input_path)
            return attn_bias, spatial_pos, in_degree, out_degree, attn_edge_type
        else:
            if not os.path.exists(self.saved_tensor_path):
                os.makedirs(self.saved_tensor_path)

            edge_attr, edge_index, x = graph.edge_attr, graph.edge_index, graph.x
            
            if edge_attr is None:
                edge_attr = torch.zeros((edge_index.shape[1]), dtype=torch.long)
        
            N = x.size(0)
            x = convert_to_single_emb(x)

            # node adj matrix [N, N] bool
            adj = torch.zeros([N, N], dtype=torch.bool)
            adj[edge_index[0, :], edge_index[1, :]] = True
            
            # edge feature here
            if len(edge_attr.size()) == 1:
                edge_attr = edge_attr[:, None]
            attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
            attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
                convert_to_single_emb(edge_attr) + 1
            )

            shortest_path_result, path = algos.floyd_warshall(adj.numpy())
            # max_dist = np.amax(shortest_path_result)
            # edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
            spatial_pos = torch.from_numpy((shortest_path_result)).long()
            print("spatial_pos size: ", spatial_pos.size())
            attn_bias = torch.zeros([N+1, N+1], dtype=torch.float)  # without graph token

            attn_bias = attn_bias
            spatial_pos = spatial_pos
            in_degree = adj.long().sum(dim=1).view(-1)
            out_degree = adj.long().sum(dim=0).view(-1)

            # save the tensors
            torch.save(attn_bias, attn_bias_path)
            torch.save(spatial_pos, spatial_pos_path)
            torch.save(in_degree, in_degree_path)
            torch.save(out_degree, out_degree_path)
            torch.save(attn_edge_type, attn_edge_type_path)
            # torch.save(edge_input, edge_input_path)
            return attn_bias, spatial_pos, in_degree, out_degree, attn_edge_type
    
    
    
class PlanDataset(object):
    def __init__(self, dataset_name, dataset_id=0):
        """
        Loading planetoid datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (dataset_name + "_" + str(dataset_id)))
        # time.sleep(10)
        # print(os.getcwd())
        self.name = dataset_name

        dataset_path = "data/dataset_de/" + str(dataset_name) + "/de_" + dataset_name + "_" + str(
            dataset_id) + ".pt"
        dataset_list = torch.load(dataset_path)[0]
        self.saved_tensor_path = "data/saved_tensors/" + str(dataset_name) + "/de_" + dataset_name + "_" + \
                                 str(dataset_id) + "/"

        self.graph, self.feat, self.labels, self.train_mask, self.valid_mask, self.test_mask, self.num_classes \
            = create_graph(dataset_list)

        self.node_num = self.feat.size(0)
        self.feat_dim = self.feat.size(1)

        # print("input_dim: %s" % self.feat_dim)
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

        prepro_start = time.time()
        print("[II] Preprocessing spatial SPD and node centrality...")
        self.preprocess_bias = self.preprocess_item(self.graph)
        print("[II] Finished preprocessing.")
        print("[II] Preprocessing time: {:.4f}s".format(time.time() - prepro_start))



    def preprocess_item(self, graph):
        # paths
        attn_bias_path = self.saved_tensor_path + "attn_bias.pt"
        spatial_pos_path = self.saved_tensor_path + "spatial_bias.pt"
        in_degree_path = self.saved_tensor_path + "in_degree.pt"
        out_degree_path = self.saved_tensor_path + "out_degree.pt"
        attn_edge_type_path = self.saved_tensor_path + "attn_edge_type.pt"
        # edge_input_path = self.saved_tensor_path + "edge_input_path.pt"
        # print(os.path.getsize(self.saved_tensor_path))

        if os.path.exists(self.saved_tensor_path) and os.path.getsize(self.saved_tensor_path):
            # loading from saved tensors
            attn_bias = torch.load(attn_bias_path)
            spatial_pos = torch.load(spatial_pos_path)
            in_degree = torch.load(in_degree_path)
            out_degree = torch.load(out_degree_path)
            attn_edge_type = torch.load(attn_edge_type_path)
            # edge_input = torch.load(edge_input_path)
            return (attn_bias, spatial_pos, in_degree, out_degree, attn_edge_type)
        else:
            if not os.path.exists(self.saved_tensor_path):
                os.makedirs(self.saved_tensor_path)

            edge_attr, edge_index, x = graph.edge_attr, graph.edge_index, graph.x
            
            if edge_attr is None:
                edge_attr = torch.zeros((edge_index.shape[1]), dtype=torch.long)
        
            N = x.size(0)
            x = convert_to_single_emb(x)

            # node adj matrix [N, N] bool
            adj = torch.zeros([N, N], dtype=torch.bool)
            adj[edge_index[0, :], edge_index[1, :]] = True
            
            # edge feature here
            if len(edge_attr.size()) == 1:
                edge_attr = edge_attr[:, None]
            attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
            attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
                convert_to_single_emb(edge_attr) + 1
            )

            shortest_path_result, path = algos.floyd_warshall(adj.numpy())
            # max_dist = np.amax(shortest_path_result)
            # edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
            spatial_pos = torch.from_numpy((shortest_path_result)).long()
            print("spatial_pos size: ", spatial_pos.size())
            attn_bias = torch.zeros([N+1, N+1], dtype=torch.float)  # without graph token

            attn_bias = attn_bias
            spatial_pos = spatial_pos
            in_degree = adj.long().sum(dim=1).view(-1)
            out_degree = adj.long().sum(dim=0).view(-1)

            # save the tensors
            torch.save(attn_bias, attn_bias_path)
            torch.save(spatial_pos, spatial_pos_path)
            torch.save(in_degree, in_degree_path)
            torch.save(out_degree, out_degree_path)
            torch.save(attn_edge_type, attn_edge_type_path)
            # torch.save(edge_input, edge_input_path)
            return (attn_bias, spatial_pos, in_degree, out_degree, attn_edge_type)


if __name__ == '__main__':
    name = "pubmed"
    PD = PlanDataset(dataset_name=name, dataset_id=0)

    dataloader = torch.utils.data.DataLoader(dataset=PD, shuffle=False, sampler=PD.train_mask)
    print("")
