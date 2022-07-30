import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

from utils import setup_seed, args_parse, load_data, accuracy


# https://github.com/H-Ambrose/GNNs_on_node-level_tasks
class SAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def graphsage_train(model, optimizer, data, labels, mask_train, params):
    model.train()
    min_loss = 9999
    patience = 0

    for epoch in range(params['epoch']):
        optimizer.zero_grad()
        out = model.forward(data)
        loss = F.nll_loss(out[mask_train], labels[mask_train])
        loss.backward()
        optimizer.step()

        if loss < min_loss:
            min_loss = loss
            patience = 0
        else:
            patience = patience + 1

        if patience > params['patience']:
            break


def main(dataset_p='', spt=''):
    print("run sage")
    mean_list = []

    params = args_parse()
    if spt != '':
        params['spt_num'] = spt
    if dataset_p != '':
        params['dataset'] = dataset_p
    loop = params['loop']

    for i in range(loop):
        setup_seed(i)

        data, labels, mask_train, mask_test = load_data(params)

        model = SAGE(params['in_dim'], params['hid_dim'], params['out_dim']).to(params['device'])
        optimizer = torch.optim.Adam([
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters()}],
            lr=params['l_rate'], weight_decay=params['w_decay'])

        graphsage_train(model, optimizer, data, labels, mask_train, params)

        acc = accuracy(model, data, labels, mask_test)
        # print('i: {},acc: {}'.format(i, acc))
        mean_list.append(acc)

    mean, var = round(np.mean(mean_list) * 100, 2), round(np.var(mean_list) * 100, 2)
    mean_var = str(mean) + '±' + str(var) + '[' + str(params['n_way']) + ']'
    print("sage {} n-way: {} k-spt: {}/{} loop: {}, mean/var: {}".format(params['dataset'], params['n_way'],
                                                                         params['out_dim'], params['spt_num'], loop,
                                                                         mean_var))
    return mean_var

# main()
