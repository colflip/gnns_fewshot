import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from utils import args_parse, setup_seed, load_data, accuracy


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def gcn_train(model, optimizer, data, labels, mask_train, params):
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
    print("run gcn")
    mean_list = []
    loop = 100
    for i in range(loop):
        setup_seed(i)

        params = args_parse()
        if spt != '':
            params['spt_num'] = spt
        if dataset_p != '':
            params['dataset'] = dataset_p

        data, labels, mask_train, mask_test = load_data(params)

        model = GCN(params['in_dim'], params['hid_dim'], params['out_dim']).to(params['device'])
        optimizer = torch.optim.Adam([
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters()}],
            lr=params['lrate'], weight_decay=params['wdecay'])

        gcn_train(model, optimizer, data, labels, mask_train, params)

        acc = accuracy(model, data, labels, mask_test)
        # print('loop step: {},acc: {}'.format(i, acc))
        mean_list.append(acc)
    print(
        "gcn {} nway: {} spt: {} loop: {}, mean: {}".format(params['dataset'], params['nway'], params['spt_num'], loop,
                                                            np.mean(mean_list)))


main()
