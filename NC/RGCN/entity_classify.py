"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""

import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import RelGraphConv
from functools import partial
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset

from model import BaseRGCN, GCN, GAT, EarlyStopping


class EntityClassify(BaseRGCN):
    def create_features(self):
        features = torch.arange(self.num_nodes)
        if self.use_cuda:
            features = features.cuda()
        return features

    def build_input_layer(self):
        return RelGraphConv(self.num_nodes, self.h_dim, self.num_rels, "basis",
                            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                            dropout=self.dropout)

    def build_hidden_layer(self, idx):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                            dropout=self.dropout)

    def build_output_layer(self):
        return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis",
                            self.num_bases, activation=None,
                            self_loop=self.use_self_loop)


def main(args):
    # load graph data
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'mutag':
        dataset = MUTAGDataset()
    elif args.dataset == 'bgs':
        dataset = BGSDataset()
    elif args.dataset == 'am':
        dataset = AMDataset()
    else:
        raise ValueError()

    # Load from hetero-graph
    hg = dataset[0]

    num_rels = len(hg.canonical_etypes)
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = hg.nodes[category].data.pop('train_mask')
    test_mask = hg.nodes[category].data.pop('test_mask')
    train_idx = torch.nonzero(train_mask).squeeze()
    test_idx = torch.nonzero(test_mask).squeeze()
    labels = hg.nodes[category].data.pop('labels')

    # split dataset into train, validate, test
    val_idx = train_idx[:len(train_idx) // 5]
    train_idx = train_idx[len(train_idx) // 5:]

    # calculate norm for each edge type and store in edge
    for canonical_etype in hg.canonical_etypes:
        u, v, eid = hg.all_edges(form='all', etype=canonical_etype)
        _, inverse_index, count = torch.unique(
            v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = torch.ones(eid.shape[0]).float() / degrees.float()
        norm = norm.unsqueeze(1)
        hg.edges[canonical_etype].data['norm'] = norm

    # get target category id
    category_id = len(hg.ntypes)
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i

    g = dgl.to_homogeneous(hg, edata=['norm'])
    num_nodes = g.number_of_nodes()
    node_ids = torch.arange(num_nodes)
    edge_norm = g.edata['norm']
    edge_type = g.edata[dgl.ETYPE].long()

    # find out the target node ids in g
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]

    # since the nodes are featureless, the input feature is then the node id.
    if not args.model == "rgcn":
        # feats = g.ndata[dgl.NTYPE]
        # feats = F.one_hot(feats, len(hg.ntypes))
        i = torch.LongTensor([[i for i in range(num_nodes)], [
                             i for i in range(num_nodes)]])
        v = torch.FloatTensor([1 for i in range(num_nodes)])
        feats = torch.sparse.FloatTensor(
            i, v, torch.Size([num_nodes, num_nodes]))
    else:
        feats = torch.arange(num_nodes)

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        feats = feats.cuda()
        edge_type = edge_type.cuda()
        edge_norm = edge_norm.cuda()
        labels = labels.cuda()

    # create model
    if args.model == "gcn":
        model = GCN(num_nodes,
                    args.n_hidden,
                    num_classes,
                    args.n_layers,
                    F.relu,
                    args.dropout,
                    True)
    elif args.model == "gat":
        heads = [4]*args.n_layers + [1]
        slope = 0.1
        model = GAT(
            in_dim=num_nodes,
            num_hidden=args.n_hidden,
            num_classes=num_classes,
            num_layers=args.n_layers,
            activation=F.elu,
            feat_drop=args.dropout,
            attn_drop=args.dropout,
            heads=heads,
            negative_slope=slope,
            residual=False,
            sparse_input=True)
    else:
        model = EntityClassify(num_nodes,
                               args.n_hidden,
                               num_classes,
                               num_rels,
                               num_bases=args.n_bases,
                               num_hidden_layers=args.n_layers - 2,
                               dropout=args.dropout,
                               use_self_loop=args.use_self_loop,
                               use_cuda=use_cuda)

    if use_cuda:
        model.cuda()
        g = g.to('cuda:%d' % args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, delta=1e-6,
                                   save_path='checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.n_layers))
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        if args.model == "gcn" or args.model == "gat":
            logits = model(g, feats)
        else:
            logits = model(g, feats, edge_type, edge_norm)
        logits = logits[target_idx]
        logits = F.softmax(logits)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        t1 = time.time()
        loss.backward()
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
              format(epoch, forward_time[-1], backward_time[-1]))
        train_acc = torch.sum(logits[train_idx].argmax(
            dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = torch.sum(logits[val_idx].argmax(
            dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
              format(train_acc, loss.item(), val_acc, val_loss.item()))
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('Early stopping!')
            break
    print()

    model.load_state_dict(torch.load(
        'checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.n_layers)))
    model.eval()
    if not args.model == "rgcn":
        logits = model(g, feats)
    else:
        logits = model.forward(g, feats, edge_type, edge_norm)
    logits = logits[target_idx]
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = torch.sum(logits[test_idx].argmax(
        dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(
        test_acc, test_loss.item()))
    print()

    print("Mean forward time: {:4f}".format(
        np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean backward time: {:4f}".format(
        np.mean(backward_time[len(backward_time) // 4:])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--model", type=str, default="rgcn",
                        help="use GNN model")
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
                        help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=5000,
                        help="number of training epochs")
    parser.add_argument('--patience', type=int, default=30, help='Patience.')
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
                        help="l2 norm coef")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
                        help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)

    args = parser.parse_args()
    print(args)
    main(args)
