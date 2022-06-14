import argparse
import random

import numpy as np
import torch
from torch_geometric.datasets import Coauthor, Amazon, Planetoid, CoraFull


def args_parse():
    parser = argparse.ArgumentParser(description='Graph arguments.')
    parser.add_argument('--cuda', type=int, default=1, help='cuda number')
    parser.add_argument('--dataset', type=str, default='Computers', help='Cora/CiteSeer/cs/Computers/Photo/CoraFull')
    parser.add_argument('--hid_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--mid_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wdecay', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--nway', type=int, default=2, help='test_classes')
    parser.add_argument('--spt_num', type=int, default=5, help='supprt set number')
    parser.add_argument('--qry_num', type=int, default=90, help='query set number')
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    params = {
        'device': device,
        'cuda': args.cuda,
        'dataset': args.dataset,
        'hid_dim': args.hid_dim,
        'mid_dim': args.mid_dim,
        'lrate': args.lrate,
        'wdecay': args.wdecay,
        'epoch': args.epoch,
        'patience': args.patience,
        'nway': args.nway,
        'spt_num': args.spt_num,
        'qry_num': args.qry_num
    }
    return params


def load_data(params):
    dname = params['dataset']
    if dname == "Cora":
        dataset = Planetoid(name='Cora', root='./data')
    elif dname == 'CiteSeer':
        dataset = Planetoid(name='Citeseer', root='./data')
    elif dname == 'cs':
        dataset = Coauthor(name='cs', root='./data')
    elif dname == 'Computers':
        dataset = Amazon(name='Computers', root='./data')
    elif dname == 'Photo':
        dataset = Amazon(name='Photo', root='./data')
    elif dname == 'CoraFull':
        dataset = CoraFull(root='./data')
    else:
        return None
    data = dataset[0].to(params['device'])
    labels = (data.y).clone().detach()
    num_classes = dataset.num_classes
    node_num = len(data.y)

    params['nway'] = num_classes * 2 // 5
    all_classes = [i for i in range(num_classes)]
    test_classes = random.sample(all_classes, params['nway'])
    train_classes = [i for i in all_classes if i not in test_classes]

    params['in_dim'] = dataset.num_features
    params['out_dim'] = num_classes

    masks = [[] for i in range(num_classes)]
    for i in range(node_num):
        cls = labels[i]
        masks[cls].append(i)

    # meta测试集的mask
    mask_train = []
    mask_test = []

    for cls in test_classes:
        spt = random.sample(masks[cls], params['spt_num'])
        mask_train.extend(spt)
        # for subitem in spt:
        #     masks[cls].remove(subitem)

        if len(masks[cls]) > params['qry_num']:
            qry = random.sample(masks[cls], params['qry_num'])
        else:
            qry = masks[cls].copy()
        mask_test.extend(qry)

    return data, labels, mask_train, mask_test


def accuracy(model, data, labels, mask_test):
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[mask_test].eq(labels[mask_test]).sum().item())
    acc = correct / len(mask_test)
    return acc


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
