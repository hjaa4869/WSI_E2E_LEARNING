import pdb
import h5py
import time
import random
import numpy as np
import torch
from utils.utils import *
import os
from models.model_mil import MIL_fc, MIL_fc_mc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.data_all = {'y_true': [], 'y_pred': []}

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

        self.data_all['y_true'].append(Y)
        self.data_all['y_pred'].append(Y_hat)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        self.data_all['y_true'].append(Y)
        self.data_all['y_pred'].append(Y_hat)

        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count

    def get_f1(self):
        y_true = np.asarray(self.data_all['y_true']).reshape(-1, )
        y_pred = np.asarray(self.data_all['y_pred']).reshape(-1, )
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1


class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=70, verbose=False):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(datasets, cur, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets

    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=6)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print('\nInit Model...', end=' ')

    if args.model_type == 'TransMIL':
        from models.TransMIL import TransMIL
        model_dict = {'n_classes': 6}
        model = TransMIL(**model_dict)

    elif args.model_type == 'DSMIL':
        from models.DSMIL import DSMIL
        model_dict = {'n_classes': 6}
        model = DSMIL(**model_dict)

    elif args.model_type == 'ABMIL':
        from models.ABMIL import ABMIL
        model_dict = {'n_classes': 6}
        model = ABMIL(**model_dict)

    elif args.model_type == 'rrtmil':
        from modules.rrtmil import RRTMIL
        model_dict = {
            'input_dim': 1024,
            'n_classes': 6,
            'pool': 'attn',
            'n_layers': 2,
            'da_act': 'tanh',
            'epeg_k': 15,
            'crmsa_k': 1,
            'all_shortcut': True
        }
        model = RRTMIL(**model_dict)

    elif args.model_type == 'e2emil':
        from models.abmilx import DAttentionX
        model_dict = {
            'input_dim': 1024,
            'n_classes': 6,
            'dropout': 0.25,
            'act': 'gelu',
            'mil_norm': 'ln',
            'mil_bias': False,
            'inner_dim': 256,
            'n_heads': 4,
            'proj_drop': 0.,
            'D': None,
            'attn_type': 'mlp',
            'attn_bias': False,
            'attn_plus': True,
            'ffn': True,
            'sdpa_type': 'math',
            'pad_v': False,
            'attn_plus_embed_new': False
        }
        model = DAttentionX(**model_dict)

    else:
        raise NotImplementedError

    model = model.to(device)
    print('Done!')

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample, batchsize=3)
    val_loader = get_split_loader(val_split, testing=args.testing, batchsize=3)
    test_loader = get_split_loader(test_split, testing=args.testing, batchsize=3)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=100, verbose=True)
    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, 6, loss_fn, writer)
        stop = validate(cur, epoch, model, val_loader, 6, early_stopping, writer, loss_fn, args.results_dir)
        torch.cuda.empty_cache()

        if stop:
            break

    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))

    val_error, val_auc, val_logger = summary(model, val_loader, 6)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
    val_f1 = val_logger.get_f1()

    test_error, test_auc, acc_logger = summary(model, test_loader, 6)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    class_accs = {}
    for i in range(6):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
        class_accs[i] = acc

    test_f1 = acc_logger.get_f1()

    test_acc0 = class_accs[0]
    test_acc1 = class_accs[1]
    test_acc2 = class_accs[2]
    test_acc3 = class_accs[3]
    test_acc4 = class_accs[4]
    test_acc5 = class_accs[5]

    if writer:
        writer.add_scalar('final/val_f1', val_f1, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_f1', test_f1, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()

    return test_auc, val_auc, test_f1, val_f1, test_acc0, test_acc1, test_acc2, test_acc3, test_acc4, test_acc5


def train_loop(epoch, model, loader, optimizer, n_classes, loss_fn=None, writer=None):
    loader.dataset.set_epoch(epoch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    scaler = torch.cuda.amp.GradScaler()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')

    optimizer.zero_grad()
    accum_steps = 1

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits, Y_hat, Y_prob = model(data)
            loss = loss_fn(logits, label)

        loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        for y_hat_i, label_i in zip(Y_hat, label):
            acc_logger.log(y_hat_i, label_i)

        loss_value = loss.item() * accum_steps
        train_loss += loss_value

        error = calculate_error(Y_hat, label)
        train_error += error

    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def validate(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    probs = []
    labels = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_hat, Y_prob = model(data)

            probs.append(Y_prob.cpu().numpy())
            labels.append(label.cpu().numpy())

            for y_hat_i, label_i in zip(Y_hat, label):
                acc_logger.log(y_hat_i, label_i)

            loss = loss_fn(logits, label)
            val_loss += loss.item()

            error = calculate_error(Y_hat, label)
            val_error += error

    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)

    val_error /= len(loader)
    val_loss /= len(loader)

    aucs = []
    binary_labels = label_binarize(labels, classes=[0, 1, 2, 3, 4, 5])

    for class_idx in range(6):
        if 0 < binary_labels[:, class_idx].sum() < len(binary_labels):
            fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], probs[:, class_idx])
            aucs.append(calc_auc(fpr, tpr))
        else:
            aucs.append(float('nan'))

    auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    for i in range(6):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_error = 0.

    probs = []
    labels = []
    hats = []

    slide_ids = loader.dataset.slide_data['slide_id']

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        with torch.no_grad():
            logits, Y_hat, Y_prob = model(data)

        for y_hat_i, label_i in zip(Y_hat, label):
            acc_logger.log(y_hat_i, label_i)

        probs.append(Y_prob.cpu().numpy())
        labels.append(label.cpu().numpy())
        hats.append(Y_hat.cpu().numpy())

        error = calculate_error(Y_hat, label)
        test_error += error

    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)
    hats = np.concatenate(hats, axis=0)

    test_error /= len(loader)

    aucs = []
    binary_labels = label_binarize(labels, classes=[0, 1, 2, 3, 4, 5])

    for class_idx in range(6):
        if 0 < binary_labels[:, class_idx].sum() < len(binary_labels):
            fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], probs[:, class_idx])
            aucs.append(calc_auc(fpr, tpr))
        else:
            aucs.append(float('nan'))

    auc = np.nanmean(np.array(aucs))

    print("\n" + "=" * 40)
    for i in range(6):
        mask = labels == i
        if mask.sum() > 0:
            wrong_mask = hats[mask] != i
            wrong_preds = hats[mask][wrong_mask]
            correct = (~wrong_mask).sum()
            total = mask.sum()

            print(f"Class {i}: {correct}/{total} correct", end="")

            if wrong_mask.sum() > 0:
                unique, counts = np.unique(wrong_preds, return_counts=True)
                error_info = ", ".join([f"{count} to {int(u)}" for u, count in zip(unique, counts)])
                print(f" (errors: {error_info})")
            else:
                print()

    print("=" * 40)

    return test_error, auc, acc_logger