import argparse
from networkx.exception import PowerIterationFailedConvergence
import numpy as np
import copy
import torch
import dgl
import torch.nn.functional as F
from models.HGT import HGT
from data.process_acm import load_acm, load_acm_simplehgn, load_acm_ycm, load_acm_magnn, load_acm_hgt
from data.process_dblp import load_dblp, load_dblp_simplehgn, load_dblp_ycm, load_dblp_magnn, load_dblp_hgt
from data.process_imdb import load_imdb, load_imdb_simplehgn, load_imdb_ycm, load_imdb_magnn, load_imdb_hgt
from models.Simple_HGN import SimpleHGN
from models.YCM import ycm
from models.HAN import HAN
from models.MAGNN import MAGNN_nc, MAGNN_nc_mb
from trainer import Trainer
from trainer_mb import Trainer_mb
import copy
import scipy.sparse as sp
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser('Main')
parser.add_argument('-s', '--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--device', type=str, default='cuda:2')
parser.add_argument('--dataset', type=str, default='DBLP')
parser.add_argument('--student', type=str, default='MAGNN')

parser.add_argument('--label_ratio', type=int, default=50, 
                    help='train labels per class [20, 50]')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--samples', type=int, default=150)

parser.add_argument('--lr_teacher', type=float, default=0.01, 
                    help='teacher learning rate')#0.01
parser.add_argument('--wd_teacher', type=float, default=0.0005, 
                    help='teacher weight decay')#0.001
parser.add_argument('--lr_student', type=float, default=0.005, 
                    help='student learning rate')#0.005
parser.add_argument('--wd_student', type=float, default=0.0005, 
                    help='student weight decay')#0.0005
parser.add_argument('--optimizer', type=str, default='adam', 
                    help='Optimizer.')
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--loss_para', type=float, default=1, 
                    help='teacher learning rate')#0.01
parser.add_argument('--iter', type=int, default=5, 
                    help='Number of training iterations.')

args = parser.parse_args()

opt = vars(args)
setup_seed(opt['seed'])
device = opt['device']
dataset = opt['dataset']
ratio = opt['label_ratio']
activation = F.relu
if dataset == 'ACM':
    type_num = [4019, 7167, 60]
    sample_rate = [5, 1]
    nei_num = 2
    etypes_list = [[0, 1], [2, 3]]
    metapath_list = 2
    edge_type = 4
    feats, labels, idx_train, idx_val, idx_test, node_num, num_classes, feat_dims = load_acm(ratio, type_num, device)

if dataset =='DBLP':
    type_num = [4057, 14328, 7723, 20]
    sample_rate = [3]
    nei_num = 1
    etypes_list = [[0, 1], [0, 2, 3, 1], [0, 4, 5, 1]]
    metapath_list = 3
    edge_type = 6
    feats, labels, idx_train, idx_val, idx_test, node_num, num_classes, feat_dims = load_dblp(ratio, type_num, device)

if dataset =='IMDB':
    type_num = [4278, 2081, 5257]
    sample_rate = [1, 3]
    nei_num = 2
    etypes_lists = [[[0, 1], [2, 3]]]
    metapath_list = [2]
    edge_type = 4
    feats, labels, idx_train, idx_val, idx_test, node_num, num_classes, feat_dims = load_imdb(ratio, type_num, device)

opt['nei_num'] = nei_num
opt['sample_rate'] = sample_rate

feats = [feat.to(device) for feat in feats]
#G = [g.to(device) for g in G]
labels = labels.to(device)

stu = opt['student']

if dataset == 'ACM':
    adjlists, edge_metapath_indices_list, type_mask = load_acm_magnn(device)
    target_node_indices = np.where(type_mask == 0)[0]
    teacher_model = MAGNN_nc_mb(2, 4, etypes_list, feat_dims, 128, num_classes, 8, 64, 'RotatE0', 0.8)
if dataset == 'DBLP':
    adjlists, edge_metapath_indices_list, type_mask = load_dblp_magnn(device)
    target_node_indices = np.where(type_mask == 0)[0]
    teacher_model = MAGNN_nc_mb(3, 6, etypes_list, feat_dims, 128, num_classes, 8, 64, 'RotatE0', 0.6)
if dataset == 'IMDB':
    g_lists, edge_metapath_indices_lists, type_mask, target_node_indices = load_imdb_magnn(device)
    teacher_model = MAGNN_nc(num_layers=1, num_metapaths_list=[2], num_edge_type=4, etypes_lists=etypes_lists, \
        feats_dim_list=feat_dims, hidden_dim=64, out_dim=num_classes, num_heads=8, attn_vec_dim=64, rnn_type='RotatE0', dropout_rate=0.5)



if stu == 'MAGNN':
    if dataset == 'ACM':
        student_model = MAGNN_nc_mb(2, 4, etypes_list, feat_dims, 128, num_classes, 8, 64, 'RotatE0', 0.8)
    if dataset == 'DBLP':
        student_model = MAGNN_nc_mb(3, 6, etypes_list, feat_dims, 128, num_classes, 8, 64, 'RotatE0', 0.6)
    if dataset == 'IMDB':
        student_model = MAGNN_nc(num_layers=1, num_metapaths_list=[2], num_edge_type=4, etypes_lists=etypes_lists, \
            feats_dim_list=feat_dims, hidden_dim=64, out_dim=num_classes, num_heads=8, attn_vec_dim=64, rnn_type='RotatE0', dropout_rate=0.6)
    student = Trainer_mb(opt, student_model, 'student')
else:
    byte_idx_train = torch.zeros_like(labels_one_hot, dtype=torch.bool).to(device)
    byte_idx_train[idx_train] = True

    student_model = ycm(g=g, graph=graph, byte_idx_train=byte_idx_train, labels_one_hot=labels_one_hot, labels_init=labels_init,  \
        node_num=node_num, features=feats, hidden_dim=128, num_classes=num_classes, layers=10, mlp_layers=4, feat_dims=feat_dims, attn_drop=0.2, dropout=0.6, \
            activation = activation, device=device)
    student = Trainer(opt, student_model, 'student')

teacher = Trainer_mb(opt, teacher_model, 'teacher')

student_model.to(opt['device'])
teacher_model.to(opt['device'])

student_state = dict([('model', copy.deepcopy(student.model.state_dict())), ('optim', copy.deepcopy(student.optimizer.state_dict()))])
teacher_state = dict([('model', copy.deepcopy(teacher.model.state_dict())), ('optim', copy.deepcopy(teacher.optimizer.state_dict()))])

teacher_target = torch.zeros(node_num, num_classes).to(device).detach()
student_target = torch.zeros(node_num, num_classes).to(device).detach()
label_init = torch.zeros(node_num, num_classes).to(device)

pre_dis_s = 0.0
pre_dis_t = 0.0

#pretrain_outputs = np.load('outputs/preds'+ str(opt['teacher']) + str(dataset) + str(ratio) + '.npy')
pretrain_outputs = np.load('outputs/preds'+ 'MAGNN' + str(dataset) + str(ratio) + '.npy')
#pretrain_outputs = np.load('outputs/preds'+ str(dataset) + str(ratio) + '.npy')
pretrain_outputs = torch.FloatTensor(pretrain_outputs).to(device)
idx_no_train = torch.LongTensor(np.setdiff1d(np.array(range(len(labels))), idx_train.cpu())).to(device)

feats = [feat.to(device) for feat in feats]
#G = [g.to(device) for g in G]
labels = labels.to(device)

def train_student(epoches, iter):

    best = [0, 0.0, 0.0, 0.0, 0.0]
    best_loss = 10000000.0
    results_auc = []
    results_micro_f1 = []
    results_macro_f1 = []
    #preds = student.predict(labels_init, feats, nei_idx)
    student_input = feats
    #input = (g_lists, feats, type_mask, edge_metapath_indices_lists)
    if iter > 0:
    #    if t_mini_batch == True:
    #        preds = teacher.predict(feats)
    #    else:
    #        preds = teacher.predict(features)
    #        student_target.copy_(preds)
        preds = teacher.predict(feats, adjlists, edge_metapath_indices_list, type_mask)
        #student_target.copy_(preds)
    if iter == 0:
        preds = pretrain_outputs
        #preds = F.log_softmax(preds, dim=-1)
        #preds = torch.exp(preds)
        #student_target.copy_(preds)
        
    #if s_mini_batch == True:
        #student_input = feats
    #else:
    #    student_input = torch.zeros_like(features).to(device).detach()
    #    student_input.copy_(features)
    student_target = preds
    for epoch in range(epoches):
        #student_input = feats
        if stu == 'LP':
            loss = student.train(idx_train, idx_val, idx_test, student_target, labels, student_input)
            loss_test, auc, micro_f1, macro_f1, att = student.evaluate(epoch, idx_val, idx_test, labels, student_target, student_input)
        else:
            loss = student.train(idx_train, idx_val, idx_test, student_target, labels, feats, adjlists, edge_metapath_indices_list, type_mask)
            loss_test, auc, micro_f1, macro_f1 = student.evaluate(epoch, idx_val, idx_test, labels, student_target, student_input, adjlists, edge_metapath_indices_list, type_mask)
        
        results_auc += auc
        results_micro_f1 += micro_f1
        results_macro_f1 += macro_f1
        acc_v, _ = micro_f1[0]
        macro_v, _ = macro_f1[0]
        auc_v, _ = auc[0]
        if loss_test < best_loss and acc_v > best[2]:
            best = [epoch + 1, loss, acc_v, macro_v, auc_v]
            best_loss = loss_test
            #best_att = att
            state = dict([('model', copy.deepcopy(student.model.state_dict())), ('optim', copy.deepcopy(student.optimizer.state_dict()))])

    print('Epoch {:d} | Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f} | Val AUC {:.4f}'.format(best[0], best[1], \
        best[2], best[3], best[4]))

    student.model.load_state_dict(state['model'])
    student.optimizer.load_state_dict(state['optim'])

    return [results_auc, results_micro_f1, results_macro_f1], best[1]#, best_att

def train_teacher(epoches):

    best = [0, 0.0, 0.0, 0.0, 0.0]
    best_loss = 10000000.0
    results_auc = []
    results_micro_f1 = []
    results_macro_f1 = []
    #preds = student.predict(labels_init, feats, nei_idx)
    #if s_mini_batch == True:
    #    preds = student.predict(feats)
    #else:
    #    preds = student.predict(features)
    #    teacher_target.copy_(preds)
    #if t_mini_batch == True:
    #    teacher_input = feats
    #else:
    #    teacher_input = torch.zeros_like(features).to(device).detach()
    #    teacher_input.copy_(features)
    teacher_input = feats
    #teacher_input = (g_lists, feats, type_mask, edge_metapath_indices_lists, target_node_indices)
    if stu == 'MAGNN':
        preds = student.predict(feats, adjlists, edge_metapath_indices_list, type_mask)
    else:
        preds = student.predict(feats)
    #teacher_target.copy_(preds)
    teacher_target = preds
    #teacher_input = feats
    for epoch in range(epoches):
        loss = teacher.train(idx_train, idx_val, idx_test, teacher_target, labels, teacher_input, adjlists, edge_metapath_indices_list, type_mask)
        loss_test, auc, micro_f1, macro_f1, _ = teacher.evaluate(epoch, idx_val, idx_test, labels, teacher_target, teacher_input, adjlists, edge_metapath_indices_list, type_mask)
        results_auc += auc
        results_micro_f1 += micro_f1
        results_macro_f1 += macro_f1
        acc_v, _ = micro_f1[0]
        macro_v, _ = macro_f1[0]
        auc_v, _ = auc[0]
        if loss_test < best_loss and acc_v > best[2]:
            best = [epoch + 1, loss, acc_v, macro_v, auc_v]
            best_loss = loss_test
            state = dict([('model', copy.deepcopy(teacher.model.state_dict())), ('optim', copy.deepcopy(teacher.optimizer.state_dict()))])

    print('Epoch {:d} | Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f} | Val AUC {:.4f}'.format(best[0], best[1], \
        best[2], best[3], best[4]))
    teacher.model.load_state_dict(state['model'])
    teacher.optimizer.load_state_dict(state['optim'])
    
    return [results_auc, results_micro_f1, results_macro_f1], best[1]

base_results = []

teacher_auc, teacher_micro, teacher_macro = [], [], []
student_auc, student_micro, student_macro = [], [], []
t_auc, t_micro, t_macro = [], [], [] 
s_auc, s_micro, s_macro = [], [], []

def get_result(results):
    best_dev, test_res = 0.0, 0.0
    for d, t in results:
        if d > best_dev:
            best_dev, test_res = d, t
    return test_res
attn = []
for k in range(opt['iter']):
    result, loss_s, best_att = train_student(opt['max_epoch'], k)
    attn.append(best_att)
    pre_dis_s = loss_s
    student_auc += result[0]
    student_micro += result[1]
    student_macro += result[2]
    s_auc.append(get_result(result[0]))
    s_micro.append(get_result(result[1]))
    s_macro.append(get_result(result[2]))
    #teacher.model.load_state_dict(teacher_state['model'])
    #teacher.optimizer.load_state_dict(teacher_state['optim'])
    result, loss_t = train_teacher(opt['max_epoch'])
    pre_dis_s = loss_t
    teacher_auc += result[0]
    teacher_micro += result[1]
    teacher_macro += result[2]
    t_auc.append(get_result(result[0]))
    t_micro.append(get_result(result[1]))
    t_macro.append(get_result(result[2]))
    #student.model.load_state_dict(student_state['model'])
    #student.optimizer.load_state_dict(student_state['optim'])



auc_test = 0.0
micro_test = 0.0
macro_test = 0.0

for i in range(len(s_auc)):
    print('Iteration : {:d} | Teacher test AUC: {:.2f} | Teacher test Micro f1: {:.2f} | Teacher test Macro f1: {:.2f} '.format(i + 1, t_auc[i] * 100, t_micro[i] * 100, t_macro[i] * 100))
    print('Iteration : {:d} | Student test AUC: {:.2f} | Student test Micro f1: {:.2f} | Student test Macro f1: {:.2f} '.format(i + 1, s_auc[i] * 100, s_micro[i] * 100, s_macro[i] * 100))
    auc_test = max(auc_test, t_auc[i], s_auc[i])
    micro_test = max(micro_test, t_micro[i], s_micro[i])
    macro_test = max(macro_test, t_macro[i], s_macro[i])
print('Test AUC: {:.2f} | Test Micro f1: {:.2f} | Test Macro f1: {:.2f} '.format(auc_test * 100, micro_test * 100, macro_test * 100))
print(attn)
