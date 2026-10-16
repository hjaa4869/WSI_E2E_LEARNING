#这是二阶段中方法，有配套的coreutils
from __future__ import print_function
import argparse
import os
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
import torch
import pandas as pd
import numpy as np

def main(args):
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_f1 = []
    all_val_f1 = []
    all_test_acc0 = []
    all_test_acc1 = []
    all_test_acc2 = []
    all_test_acc3 = []
    all_test_acc4 = []
    all_test_acc5 = []
    
    folds = np.arange(start, end)#(0,4)
    print('folds:',folds)
    for i in folds:
        seed_torch(args.seed+i)
        
        train_dataset, val_dataset, test_dataset = dataset.return_splits(csv_path='{}/split_{}.csv'.format(args.split_dir, i))
        datasets = (train_dataset, val_dataset, test_dataset)
        test_auc, val_auc, test_f1, val_f1, test_acc0, test_acc1, test_acc2, test_acc3, test_acc4, test_acc5 = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_f1.append(test_f1)
        all_val_f1.append(val_f1)
        all_test_acc0.append(test_acc0)
        all_test_acc1.append(test_acc1)
        all_test_acc2.append(test_acc2)
        all_test_acc3.append(test_acc3)
        all_test_acc4.append(test_acc4)
        all_test_acc5.append(test_acc5)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc,
        'val_auc': all_val_auc, 'test_f1': all_test_f1, 'val_f1' : all_val_f1, 'test_acc0': all_test_acc0, 'test_acc1': all_test_acc1, 'test_acc2': all_test_acc2, 'test_acc3': all_test_acc3, 'test_acc4': all_test_acc4, 'test_acc5': all_test_acc5})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=400,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--reg', type=float, default=1e-2,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=5, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use, '
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd', 'adamw'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'ea_mil', 'ABMIL', 'TransMIL', 'MaxPooling', 'MeanPooling', 'DSMIL', 'rrtmil', 'e2eMIL'], default='TransMIL', help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_e2e', 'task_3_gradding'])
### 针对CLAM
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--bag_weight', type=float, default=1.0,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
#下列参数不手动设置，其中部分同样针对clam
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--no_inst_cluster', action='store_true', default=False, help='disable instance-level clustering')
parser.add_argument('--subtyping', action='store_true', default=False, help='subtyping problem')
parser.add_argument('--vib', action='store_true', default=False, help='enable variational IB')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.data_root_dir = "./data_feat"
args.max_epochs = 60
args.lr = 1e-4
args.reg = 1e-04
args.seed = 2023
args.k = 1
args.k_start = 0
args.k_end = 1
args.results_dir = "./results"
args.split_dir = "task_panda"
args.log_data = False#控制tensorboard记录相关训练数据,日志数据
args.early_stopping = True
args.opt = 'adamw'
args.drop_out=True
args.bag_loss = 'ce'
args.model_type = 'e2emil'
args.weighted_sample =True
args.task = 'task_3_grading'
args.csv_path = './dataset_csv/panda.csv'
sub_feat_dir = 'panda_res50'
args.use_drop_out = True
args.use_lmdb = True
args.lmdb_path = './lmdb/panda/256_80_jpg.lmdb'
args.sampled_number = 128
args.n_classes = 2
#后续参数针对clam
args.inst_loss ="svm"
args.bag_weight = 0.7

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


encoding_size = 1024

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(
                            data_dir = os.path.join(args.data_root_dir, sub_feat_dir),
                            use_lmdb = args.use_lmdb,
                            lmdb_path = args.lmdb_path,
                            sampled_number = args.sampled_number,
                            csv_path = args.csv_path,
                            shuffle = False,
                            label_col = 'label',
                            print_info = True,
                            seed = args.seed,
                            num_classes = 2)    
    
elif args.task == 'task_2_e2e':
    dataset = Generic_MIL_Dataset(
                            data_dir = os.path.join(args.data_root_dir, sub_feat_dir),
                            use_lmdb = args.use_lmdb,
                            lmdb_path = args.lmdb_path,
                            sampled_number = args.sampled_number,
                            csv_path = args.csv_path,
                            shuffle = False,
                            label_col = 'label',
                            print_info = True,
                            seed = args.seed,
                            num_classes = 2)
    
    
elif args.task == 'task_3_grading':
    args.n_classes=6
    dataset = Generic_MIL_Dataset(
                            data_dir = os.path.join(args.data_root_dir, sub_feat_dir),
                            use_lmdb = args.use_lmdb,
                            lmdb_path = args.lmdb_path,
                            sampled_number = args.sampled_number,
                            csv_path = args.csv_path,
                            shuffle = False,
                            label_col = 'label',
                            print_info = True,
                            seed = args.seed,
                            num_classes = 6)#是否启用h5文件可以在代码内部进行选择
    
    
else:
    raise NotImplementedError

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

if __name__ == "__main__":
    # pdb.set_trace()
    results = main(args)
    print("finished!")
    print("end script")

