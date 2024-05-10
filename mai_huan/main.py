import os
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import dgl
from dgl.data import DGLDataset
from functions import compute_auc, seed_torch, compute_f1_score_tpr_fpr_full
from log import setup_logging
from model import GraphSAGE
import warnings
warnings.filterwarnings("ignore")


class MyDataset(DGLDataset):
    def __init__(self):
        super(MyDataset, self).__init__(name="my_dataset")

    def process(self):
        threshold = 70
        A = np.load(r"A.npy") # Adjacency matrix
        A = np.where(A >= np.percentile(A, threshold), A, 0)
        A = sp.csr_matrix(A)

        nodes_data = pd.read_csv(r"data.csv") # Data
        nodes_data = nodes_data / nodes_data.max(axis=0)
        node_features = torch.from_numpy(nodes_data.iloc[:, :-1].to_numpy()).to(torch.float32)
        node_labels = torch.from_numpy(nodes_data.iloc[:, -1].to_numpy()).to(torch.long)

        self.graph = dgl.from_scipy(A).add_self_loop()
        self.graph.ndata["feat"] = node_features
        self.graph.ndata["label"] = node_labels
        self.graph.num_classes = 2

        n_nodes = nodes_data.shape[0]
        n_train = int(n_nodes * 0.7)
        n_val = int(n_nodes * 0.1)

        # Generate a random permutation of node indices
        perm = torch.randperm(n_nodes)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train: n_train + n_val]] = True
        test_mask[ perm[n_train + n_val:]] = True

        self.graph.ndata["train_mask"] = train_mask
        self.graph.ndata["val_mask"] = val_mask
        self.graph.ndata["test_mask"] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

def train(g, model, batch_size, num_layer, epochs, lr=0.01, weight_decay=5e-4, early_stopping=None, verbose=True):
    # Create log directory
    log_folder = "./logs/"
    os.makedirs(log_folder, exist_ok=True)
    model_name = re.findall(r"(.*?)[(]", str(model))[0]
    model_folder = "./logs/{}".format(model_name)
    os.makedirs(model_folder, exist_ok=True)
    log_file = "./logs/{}/training_log_{}_L{}_LR{}_WD{}_ES{}.txt".format(model_name, model_name, num_layer, lr, weight_decay,
                                                                     early_stopping)
    setup_logging(log_file)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    best_test_acc = 0
    best_f1_score = 0
    best_auc = 0
    patience = 0

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    train_nid = torch.nonzero(train_mask, as_tuple=False).squeeze()
    val_mask = g.ndata["val_mask"]
    val_nid = torch.nonzero(val_mask, as_tuple=False).squeeze()
    test_mask = g.ndata["test_mask"]
    test_nid = torch.nonzero(test_mask, as_tuple=False).squeeze()

    sample_size = [15, 10, 5]
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_size, prefetch_node_feats=["feat"], prefetch_labels=["label"])

    train_loader = dgl.dataloading.DataLoader(g, train_nid, sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=1)
    val_loader = dgl.dataloading.DataLoader(g, val_nid, sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=1)
    test_loader = dgl.dataloading.DataLoader(g, test_nid, sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=1)


    for e in range(1, epochs + 1):
        # Train
        model.train()
        for input_nodes, output_nodes, blocks in train_loader:
            batch_features = blocks[0].srcdata['feat']
            batch_labels = blocks[-1].dstdata['label']
            # Forward
            logits = model(blocks, batch_features)

            # Compute prediction
            pred = logits.argmax(1)

            # Compute loss
            train_loss = F.cross_entropy(logits, batch_labels)

            # Compute accuracy on training set
            train_acc = (pred == batch_labels).float().mean()

            # Backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss_total = 0
            val_acc_total = 0
            val_f1_score_total = 0
            val_TPR_total = 0
            val_FPR_total = 0
            val_auc_total = 0
            # for batch_features, batch_labels, _, batch_val_mask, _ in val_loader:
            for input_nodes, output_nodes, blocks in val_loader:
                batch_features = blocks[0].srcdata['feat']
                batch_labels = blocks[-1].dstdata['label']
                # Forward
                logits = model(blocks, batch_features)

                # Compute prediction
                pred = logits.argmax(1)

                # Compute loss
                val_loss_total += F.cross_entropy(logits, batch_labels)

                # Compute accuracy on validation set
                val_acc_total += (pred == batch_labels).float().mean()
                val_TP, val_FP, val_FN, val_TN, val_f1_score_tmp, val_TPR_tmp, val_FPR_tmp = compute_f1_score_tpr_fpr_full(pred, batch_labels)
                val_auc_tmp = compute_auc(pred, batch_labels)
                val_f1_score_total += val_f1_score_tmp
                val_TPR_total += val_TPR_tmp
                val_FPR_total += val_FPR_tmp
                val_auc_total += val_auc_tmp

            # Compute average loss and accuracy on validation set
            val_loss = val_loss_total / len(val_loader.dataset)
            val_acc = val_acc_total / len(val_loader.dataset)
            val_f1_score = val_f1_score_total / len(val_loader.dataset)
            val_TPR = val_TPR_total / len(val_loader.dataset)
            val_FPR = val_FPR_total / len(val_loader.dataset)
            val_auc = val_auc_total / len(val_loader.dataset)

        # Test
        model.eval()
        with torch.no_grad():
            test_loss_total = 0
            test_acc_total = 0
            test_f1_score_total = 0
            test_TPR_total = 0
            test_FPR_total = 0
            test_auc_total = 0

            for input_nodes, output_nodes, blocks in test_loader:
                batch_features = blocks[0].srcdata['feat']
                batch_labels = blocks[-1].dstdata['label']
                # Forward
                logits = model(blocks, batch_features)

                # Compute prediction
                pred = logits.argmax(1)

                # Compute loss
                test_loss_total += F.cross_entropy(logits, batch_labels)
                # Compute accuracy on validation set
                test_acc_total += (pred == batch_labels).float().mean()
                test_TP, test_FP, test_FN, test_TN, test_f1_score_tmp, test_TPR_tmp, test_FPR_tmp = compute_f1_score_tpr_fpr_full(pred, batch_labels)
                test_auc_tmp = compute_auc(pred, batch_labels)
                test_f1_score_total += test_f1_score_tmp
                test_TPR_total += test_TPR_tmp
                test_FPR_total += test_FPR_tmp
                test_auc_total += test_auc_tmp


            # Compute average loss and accuracy on test set
            test_loss = test_loss_total / len(test_loader.dataset)
            test_acc = test_acc_total / len(test_loader.dataset)
            test_f1_score = test_f1_score_total / len(test_loader.dataset)
            test_TPR = test_TPR_total / len(test_loader.dataset)
            test_FPR = test_FPR_total / len(test_loader.dataset)
            test_auc = test_auc_total / len(test_loader.dataset)

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_f1_score = val_f1_score
            best_auc = val_auc
        else:
            patience += 1


        if e % 5 == 0:
            logging.info(
                f"In epoch {e}, train loss: {train_loss:.3f}, val loss: {val_loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}), test acc: {test_acc:.3f} (best {best_test_acc:.3f}), val F1: {val_f1_score:.3f}, val TPR: {val_TPR:.3f}, val FPR: {val_FPR:.3f}, val AUC: {val_auc:.3f}, val_TP: {val_TP}, val_FP: {val_FP}, val_FN: {val_FN}, val_TN: {val_TN} ."
            )


        if early_stopping is not None and patience >= early_stopping:
            logging.info(f"Early stopping at epoch {e}!")
            break

        if e % 100 == 0:
            checkpoint_folder = "./checkpoints"
            os.makedirs(checkpoint_folder, exist_ok=True)
            epoch_folder = "./checkpoints/{}_L{}_LR{}_WD{}_ES{}/epoch_{}".format(model_name,num_layer, lr, weight_decay,
                                                                             early_stopping, e)
            os.makedirs(epoch_folder, exist_ok=True)
            state_dict_checkpoint_path = os.path.join(epoch_folder, f"state_dict_checkpoint_epoch_{e}.pt")
            model_checkpoint_path = os.path.join(epoch_folder, f"model_checkpoint_epoch_{e}.pt")
            torch.save(model.state_dict(), state_dict_checkpoint_path)
            torch.save(model, model_checkpoint_path)
    print('Done!')



if __name__ == '__main__':
    seed_torch()
    print('Loading dataset...')
    dataset = MyDataset()
    print(dataset)

    # GraphSAGE
    print('GraphSAGE...')
    epochs = 3000
    in_feats = 13
    hidden_feats = 16
    out_feats = 2
    batch_size = 512
    num = 3

    graph = dataset[0]
    model = GraphSAGE(in_feats, hidden_feats, out_feats, num_layers = num)
    train(graph, model, num_layer = num, epochs = epochs , batch_size = batch_size)
