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
from dgl.data.utils import save_graphs
from models.HGCF import HGCF
from models.GTN import GTN
from models.HAN import HAN
from models.MAGNN import MAGNN_nc, MAGNN_nc_mb
from trainer import Trainer

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
parser.add_argument('--teacher', type=str, default='HAN')
parser.add_argument('--student', type=str, default='LP')

parser.add_argument('--label_ratio', type=int, default=20, 
                    help='train labels per class [20, 50]')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--layer', type=int, default=6)
parser.add_argument('--mlp', type=int, default=2)
parser.add_argument('--hidden', type=int, default=128)

parser.add_argument('--lr_teacher', type=float, default=0.01, 
                    help='teacher learning rate')#0.01
parser.add_argument('--wd_teacher', type=float, default=0.0000, 
                    help='teacher weight decay')#0.001
parser.add_argument('--lr_student', type=float, default=0.03, 
                    help='student learning rate')#0.005
parser.add_argument('--wd_student', type=float, default=0.0000, 
                    help='student weight decay')#0.0005
parser.add_argument('--optimizer', type=str, default='adam', 
                    help='Optimizer.')
parser.add_argument('--max_epoch', type=int, default=150)
parser.add_argument('--iter', type=int, default=3, 
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
    feats, labels, idx_train, idx_val, idx_test, node_num, num_classes, feat_dims = load_acm(ratio, type_num, device)

if dataset =='DBLP':
    type_num = [4057, 14328, 7723, 20]
    sample_rate = [3]
    nei_num = 1
    opt['etypes_lists'] = [[0, 1], [0, 2, 3, 1], [0, 4, 5, 1]]
    opt['metapath_list'] = 3
    opt['edge_type'] = 6
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
tea = opt['teacher']

if tea == stu:
    if tea == 'HAN':
        if dataset == 'ACM':
            g, graph, nei_idx, labels_one_hot, labels_init = load_acm_ycm(ratio, device)
        if dataset == 'DBLP':
            g, graph, nei_idx, labels_one_hot, labels_init = load_dblp_ycm(ratio, device)
        if dataset == 'IMDB':
            g, graph, nei_idx, labels_one_hot, labels_init = load_imdb_ycm(ratio, device)

        teacher_model = HAN(num_paths=len(g), in_dim=feats[0].shape[1], hid_dim=64, num_heads=[8], dropout=0.5, num_classes=num_classes,  g=g)
        student_model = HAN(num_paths=len(g), in_dim=feats[0].shape[1], hid_dim=64, num_heads=[8], dropout=0.5, num_classes=num_classes,  g=g)

    if tea == 'MAGNN':
        if dataset == 'ACM':
            adjlists, edge_metapath_indices_list, type_mask = load_acm_magnn(device)
        if dataset == 'DBLP':
            adjlists, edge_metapath_indices_list, type_mask = load_dblp_magnn(device)
        if dataset == 'IMDB':
            g_lists, edge_metapath_indices_lists, type_mask, target_node_indices = load_imdb_magnn(device)
            opt['target_node_indices'] = target_node_indices
            teacher_model = MAGNN_nc(num_layers=1, num_metapaths_list=[2], num_edge_type=4, etypes_lists=etypes_lists, \
                feats_dim_list=feat_dims, hidden_dim=64, out_dim=num_classes, num_heads=8, attn_vec_dim=64, rnn_type='RotatE0', dropout_rate=0.5)
            student_model = MAGNN_nc(num_layers=1, num_metapaths_list=[2], num_edge_type=4, etypes_lists=etypes_lists, \
                feats_dim_list=feat_dims, hidden_dim=64, out_dim=num_classes, num_heads=8, attn_vec_dim=64, rnn_type='RotatE0', dropout_rate=0.5)

    if tea == 'SimpleHGN':
        if dataset == 'ACM':
            gs = load_acm_simplehgn(device)
        if dataset == 'DBLP':
            gs = load_dblp_simplehgn(device)
        if dataset == 'IMDB':
            gs= load_imdb_simplehgn(device)
        teacher_model = SimpleHGN(graph=gs, edge_dim=32, num_etypes=5, in_dim=64, feats_dim_list=feat_dims, num_hidden=64, num_classes=num_classes,\
        num_layers=1, heads=[8, 1], feat_drop=0.6, attn_drop=0.6, negative_slope=0.05, residual=True, alpha=0.05)

        student_model = SimpleHGN(graph=gs, edge_dim=32, num_etypes=5, in_dim=64, feats_dim_list=feat_dims, num_hidden=64, num_classes=num_classes,\
        num_layers=1, heads=[8, 1], feat_drop=0.6, attn_drop=0.6, negative_slope=0.05, residual=True, alpha=0.05)
    if tea == 'HGT':
        if dataset == 'ACM':
            G_hgt, node_dict, edge_dict = load_acm_hgt(device)
            teacher_model = HGT(G_hgt, node_dict, edge_dict, n_inp=[feat_dims[1], feat_dims[0], feat_dims[2]], n_hid=256, n_out=num_classes,n_layers=2, \
            n_heads=8, feat_drop=0.6, dropout=0.8, use_norm = True).to(device)
            student_model = HGT(G_hgt, node_dict, edge_dict, n_inp=[feat_dims[1], feat_dims[0], feat_dims[2]], n_hid=256, n_out=num_classes,n_layers=2, \
            n_heads=8, feat_drop=0.6, dropout=0.8, use_norm = True).to(device)
        if dataset == 'DBLP':
            G_hgt, node_dict, edge_dict = load_dblp_hgt(device)
            teacher_model = HGT(G_hgt, node_dict, edge_dict, n_inp=[feat_dims[0], feat_dims[3], feat_dims[1], feat_dims[2]], n_hid=256, n_out=num_classes, \
            n_layers=2, n_heads=4, feat_drop=0.6,  dropout = 0.6,  use_norm = True).to(device)
            student_model = HGT(G_hgt, node_dict, edge_dict, n_inp=[feat_dims[0], feat_dims[3], feat_dims[1], feat_dims[2]], n_hid=256, n_out=num_classes, \
            n_layers=2, n_heads=4, feat_drop=0.6,  dropout = 0.5,  use_norm = True).to(device)
        if dataset == 'IMDB':
            G_hgt, node_dict, edge_dict = load_imdb_hgt(device)
            teacher_model = HGT(G_hgt, node_dict, edge_dict, n_inp=[feat_dims[2], feat_dims[1], feat_dims[0]], n_hid=256, n_out=num_classes, \
            n_layers=2, n_heads=4, feat_drop=0.6,  dropout = 0.5,  use_norm = True).to(device)
            student_model = HGT(G_hgt, node_dict, edge_dict, n_inp=[feat_dims[2], feat_dims[1], feat_dims[0]], n_hid=256, n_out=num_classes, \
            n_layers=2, n_heads=4, feat_drop=0.6,  dropout = 0.5,  use_norm = True).to(device)

else:
    if dataset == 'ACM':
        g, graph, nei_idx, labels_one_hot, labels_init = load_acm_ycm(ratio, device)
    if dataset == 'DBLP':
        g, graph, nei_idx, labels_one_hot, labels_init = load_dblp_ycm(ratio, device)
    if dataset == 'IMDB':
        g, graph, nei_idx, labels_one_hot, labels_init = load_imdb_ycm(ratio, device)

    if tea == 'GTN':
         edges = np.load("data/acm/edge.npy", allow_pickle=True)
         num_nodes = 4019 + 7167 + 60
    if tea == 'HAN':
        teacher_model = HAN(num_paths=len(g), in_dim=feats[0].shape[1], hid_dim=128, num_heads=[8], dropout=0.6, num_classes=num_classes,  g=g)

    if tea == 'MAGNN':
        if dataset == 'ACM':
            adjlists, edge_metapath_indices_list, type_mask = load_acm_magnn(device)
        if dataset == 'DBLP':
            adjlists, edge_metapath_indices_list, type_mask = load_dblp_magnn(device)
        if dataset == 'IMDB':
            g_lists, edge_metapath_indices_lists, type_mask, target_node_indices = load_imdb_magnn(device)
            opt['target_node_indices'] = target_node_indices
            teacher_model = MAGNN_nc(num_layers=1, num_metapaths_list=[2], num_edge_type=4, etypes_lists=etypes_lists, \
                feats_dim_list=feat_dims, hidden_dim=64, out_dim=num_classes, num_heads=8, attn_vec_dim=64, rnn_type='RotatE0', dropout_rate=0.6)

    if tea == 'SimpleHGN':
        if dataset == 'ACM':
            gs = load_acm_simplehgn(device)
        if dataset == 'DBLP':
            gs = load_dblp_simplehgn(device)
        if dataset == 'IMDB':
            gs= load_imdb_simplehgn(device)
        teacher_model = SimpleHGN(graph=gs, edge_dim=64, num_etypes=6, in_dim=128, feats_dim_list=feat_dims, num_hidden=128, num_classes=num_classes,\
        num_layers=1, heads=[8, 1], feat_drop=0.7, attn_drop=0.6, negative_slope=0.05, residual=True, alpha=0.05)
    if tea == 'HGT':
        if dataset == 'ACM':
            G_hgt, node_dict, edge_dict = load_acm_hgt(device)
            teacher_model = HGT(G_hgt, node_dict, edge_dict, n_inp=[feat_dims[1], feat_dims[0], feat_dims[2]], n_hid=256, n_out=num_classes,n_layers=2, \
            n_heads=4, feat_drop=0.7, dropout=0.3, use_norm = True).to(device)

        if dataset == 'DBLP':
            G_hgt, node_dict, edge_dict = load_dblp_hgt(device)
            teacher_model = HGT(G_hgt, node_dict, edge_dict, n_inp=[feat_dims[0], feat_dims[3], feat_dims[1], feat_dims[2]], n_hid=256, n_out=num_classes, \
            n_layers=2, n_heads=4, feat_drop=0.6,  dropout = 0.5,  use_norm = True).to(device)

        if dataset == 'IMDB':
            G_hgt, node_dict, edge_dict = load_imdb_hgt(device)
            teacher_model = HGT(G_hgt, node_dict, edge_dict, n_inp=[feat_dims[2], feat_dims[1], feat_dims[0]], n_hid=256, n_out=num_classes, \
            n_layers=2, n_heads=4, feat_drop=0.6,  dropout = 0.5,  use_norm = True).to(device)

    byte_idx_train = torch.zeros_like(labels_one_hot, dtype=torch.bool).to(device)
    byte_idx_train[idx_train] = True

    student_model = HGCF(g=g, graph=graph, byte_idx_train=byte_idx_train, labels_one_hot=labels_one_hot, labels_init=labels_init,  \
        node_num=node_num, features=feats, hidden_dim=opt['hidden'], num_classes=num_classes, layers=opt['layer'], mlp_layers=opt['mlp'], feat_dims=feat_dims, attn_drop=0.5, dropout=0.6, \
            activation = activation, device=device)

teacher = Trainer(opt, teacher_model, 'teacher')
student = Trainer(opt, student_model, 'student')

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
pretrain_outputs = np.load('outputs/preds'+ str(opt['teacher']) + str(dataset) + str(ratio) + '.npy')
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
        preds = teacher.predict(feats)
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
        loss = student.train(idx_train, idx_val, idx_test, student_target, labels, student_input)
        loss_test, auc, micro_f1, macro_f1, att, gt = student.evaluate(epoch, idx_val, idx_test, labels, student_target, student_input)
        results_auc += auc
        results_micro_f1 += micro_f1
        results_macro_f1 += macro_f1
        acc_v, _ = micro_f1[0]
        macro_v, _ = macro_f1[0]
        auc_v, _ = auc[0]
        if loss_test < best_loss and acc_v > best[2]:
            best = [epoch + 1, loss, acc_v, macro_v, auc_v]
            best_loss = loss_test
            best_att = att
            best_gt = gt
            state = dict([('model', copy.deepcopy(student.model.state_dict())), ('optim', copy.deepcopy(student.optimizer.state_dict()))])

    print('Epoch {:d} | Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f} | Val AUC {:.4f}'.format(best[0], best[1], \
        best[2], best[3], best[4]))

    student.model.load_state_dict(state['model'])
    student.optimizer.load_state_dict(state['optim'])

    return [results_auc, results_micro_f1, results_macro_f1], best[1], best_att, best_gt

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
    preds = student.predict(feats)
    #teacher_target.copy_(preds)
    teacher_target = preds
    #teacher_input = feats
    for epoch in range(epoches):
        loss = teacher.train(idx_train, idx_val, idx_test, teacher_target, labels, teacher_input)
        loss_test, auc, micro_f1, macro_f1, d, pred = teacher.evaluate(epoch, idx_val, idx_test, labels, teacher_target, teacher_input)
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
    np.save('outputs/DBLP_HGCF_20.npy', pred.cpu().numpy())
    
    return [results_auc, results_micro_f1, results_macro_f1], best[1]

base_results = []

teacher_auc, teacher_micro, teacher_macro = [], [], []
student_auc, student_micro, student_macro = [], [], []
t_auc, t_micro, t_macro = [], [], [] 
s_auc, s_micro, s_macro = [], [], []



def get_result(results):
    best_dev, test_res = 0.0, 0.0
    for d, t in results:
        if d >= best_dev:
            best_dev, test_res = d, t
    return test_res, best_dev
attn = []
for k in range(opt['iter']):
    result, loss_s, best_att, best_gt = train_student(opt['max_epoch'], k)
    attn = best_att
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

save_graphs('data/g.bin', best_gt)
    



auc_test = 0.0
micro_test = 0.0
macro_test = 0.0

for i in range(len(s_auc)):
    print('Iteration : {:d} | Teacher test AUC: {:.2f} | Teacher test Micro f1: {:.2f} | Teacher test Macro f1: {:.2f} '.format(i + 1, t_auc[i][0] * 100, t_micro[i][0] * 100, t_macro[i][0] * 100))
    print('Iteration : {:d} | Student test AUC: {:.2f} | Student test Micro f1: {:.2f} | Student test Macro f1: {:.2f} '.format(i + 1, s_auc[i][0] * 100, s_micro[i][0] * 100, s_macro[i][0] * 100))
auc_test = get_result(student_auc+teacher_auc)
micro_test = get_result(student_micro+teacher_micro)
macro_test = get_result(student_macro+teacher_macro)
print('Test AUC: {:.2f} | Test Micro f1: {:.2f} | Test Macro f1: {:.2f} '.format(auc_test[0] * 100, micro_test[0] * 100, macro_test[0] * 100))
#(att_t, att_c) = attn


#att_t = att_t.detach().cpu().numpy()
#att_c = att_c.detach().cpu().numpy()
#np.save('outputs/'+ str(opt['teacher']) + 'attt.npy', att_t)
#np.save('outputs/'+ str(opt['teacher']) + 'attc.npy', att_c)