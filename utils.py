import argparse
import random

import numpy as np
import torch
from torch_geometric.datasets import Coauthor, Amazon, Planetoid, CoraFull


def args_parse():
    parser = argparse.ArgumentParser(description='Graph arguments.')
    parser.add_argument('--loop', type=int, default=500, help='epoch')

    parser.add_argument('--cuda', type=int, default=1, help='cuda number')
    parser.add_argument('--dataset', type=str, default='Computers', help='Cora/CiteSeer/Photo/cs/Computers/CoraFull')
    parser.add_argument('--hid_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--mid_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--l_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--w_decay', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--n_way', type=int, default=2, help='test_classes')
    parser.add_argument('--spt_num', type=int, default=1, help='supprt set number')
    parser.add_argument('--qry_num', type=int, default=12, help='query set number')
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    params = {
        'loop': args.loop,
        'device': device,
        'cuda': args.cuda,
        'dataset': args.dataset,
        'hid_dim': args.hid_dim,
        'mid_dim': args.mid_dim,
        'l_rate': args.l_rate,
        'w_decay': args.w_decay,
        'epoch': args.epoch,
        'patience': args.patience,
        'n_way': args.n_way,
        'spt_num': args.spt_num,
        'qry_num': args.qry_num
    }
    return params


def load_data(params):
    d_name = params['dataset']
    if d_name == "Cora":
        dataset = Planetoid(name='Cora', root='./data')
    elif d_name == 'CiteSeer':
        dataset = Planetoid(name='Citeseer', root='./data')
    elif d_name == 'cs':
        dataset = Coauthor(name='cs', root='./data')
    elif d_name == 'Computers':
        dataset = Amazon(name='Computers', root='./data')
    elif d_name == 'Photo':
        dataset = Amazon(name='Photo', root='./data')
    elif d_name == 'CoraFull':
        dataset = CoraFull(root='./data/CoraFull')
    else:
        return None
    data = dataset[0].to(params['device'])
    labels = data.y.clone().detach()
    num_classes = dataset.num_classes
    node_num = len(data.y)

    params['n_way'] = num_classes * 2 // 5
    all_classes = [i for i in range(num_classes)]
    test_classes = random.sample(all_classes, params['n_way'])

    params['in_dim'] = dataset.num_features
    params['out_dim'] = num_classes

    masks = [[] for i in range(num_classes)]
    for i in range(node_num):
        cls = labels[i]
        masks[cls].append(i)

    mask_train = []
    mask_test = []

    for cls in test_classes:
        spt = random.sample(masks[cls], params['spt_num'])
        mask_train.extend(spt)
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
