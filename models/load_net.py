"""
    Utility file to select GraphNN model as
    selected by the user
"""

from graphormerModel import GraphormerModel


def Graphormer(net_params, data_params):
    return GraphormerModel(net_params, data_params)


def gnn_model(MODEL_NAME, net_params, data_params):
    models = {
        'Graphormer': Graphormer
    }

    return models[MODEL_NAME](net_params,data_params)
