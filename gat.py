import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from utils import args_parse, setup_seed, load_data, accuracy


# https://github.com/H-Ambrose/GNNs_on_node-level_tasks/blob/master/GATmodel.ipynb
class GAT(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_dim, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_dim, dropout=0.6)

    def forward(self, data):
        x = F.dropout(data.x, p=0.4, training=self.training)
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


def gat_train(model, optimizer, data, labels, mask_train, params):
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
    print("run gat")
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

        model = GAT(params['in_dim'], params['hid_dim'], params['out_dim']).to(params['device'])
        optimizer = torch.optim.Adam([
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters()}],
            lr=params['l_rate'], weight_decay=params['w_decay'])

        gat_train(model, optimizer, data, labels, mask_train, params)

        acc = accuracy(model, data, labels, mask_test)
        # print('i: {}, acc: {}'.format(i, acc))
        mean_list.append(acc)

    mean, var = round(np.mean(mean_list) * 100, 2), round(np.var(mean_list) * 100, 2)
    mean_var = str(mean) + '±' + str(var)
    print(
        "gat {} n_way: {} spt: {} loop: {}, mean/var: {}".format(params['dataset'], params['n_way'], params['spt_num'],
                                                                 loop, mean_var))
    return mean_var

# main()
