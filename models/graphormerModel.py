# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils

from fairseq.modules import (
    LayerNorm,
)

from graphormer import  GraphormerGraphEncoder

logger = logging.getLogger(__name__)


class GraphormerModel(nn.Module):
    def __init__(self, net_params, data_params):
        super().__init__()
        self.args = net_params
        self.encoder_embed_dim = net_params['encoder_embed_dim']
        self.encoder = GraphormerEncoder(net_params,data_params)

    def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)


class GraphormerEncoder(nn.Module):
    def __init__(self, net_params, data_params):
        super().__init__()
        self.max_nodes = data_params['max_nodes']

        self.graph_encoder = GraphormerGraphEncoder(
            # < for graphormer
            num_atoms=data_params['num_atoms'],
            num_in_degree=data_params['num_in_degree'],
            num_out_degree=data_params['num_out_degree'],
            num_edges=data_params['num_edges'],
            num_spatial=data_params['num_spatial'],
            num_edge_dis=data_params['num_edge_dis'],
            edge_type=data_params['edge_type'],
            multi_hop_max_dist=data_params['multi_hop_max_dist'],
            # >
            num_encoder_layers=net_params['encoder_layers'],
            embedding_dim=net_params['encoder_embed_dim'],
            ffn_embedding_dim=net_params['encoder_ffn_embed_dim'],
            num_attention_heads=net_params['encoder_attention_heads'],
            dropout=net_params['dropout'],
            attention_dropout=net_params['attention_dropout'],
            activation_dropout=net_params['act_dropout'],
            encoder_normalize_before=net_params['encoder_normalize_before'],
            pre_layernorm=net_params['pre_layernorm'],
            apply_graphormer_init=net_params['apply_graphormer_init'],
            activation_fn=net_params['activation_fn'],
        )

        self.share_input_output_embed = net_params['share_encoder_input_output_embed']
        self.embed_out = None
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not net_params['remove_head']

        self.masked_lm_pooler = nn.Linear(
            net_params['encoder_embed_dim'], net_params['encoder_embed_dim']
        )

        self.lm_head_transform_weight = nn.Linear(
            net_params['encoder_embed_dim'], net_params['encoder_embed_dim']
        )
        self.activation_fn = utils.get_activation_fn(net_params['activation_fn'])
        self.layer_norm = LayerNorm(net_params['encoder_embed_dim'])

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    net_params['encoder_embed_dim'], data_params['num_classes'], bias=False
                )
            else:
                raise NotImplementedError


    def forward(self, batched_data, perturb=None, masked_tokens=None):
        inner_states, graph_rep = self.graph_encoder(
            batched_data,
            perturb=perturb,
        )

        x = inner_states[-1].transpose(0, 1)

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
            self.graph_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        return x








