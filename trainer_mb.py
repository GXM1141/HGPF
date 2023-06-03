import time
import torch
import torch.nn.functional as F
import numpy as np
from utils import my_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from utils import index_generator, parse_minibatch

def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

class Trainer_mb(object):
    def __init__(self, opt, model, Mtype):
        self.opt = opt
        self.model = model
        self.loss_fcn = torch.nn.CrossEntropyLoss()
        self.Mtype = Mtype
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.batch_size = opt['batch_size']
        if Mtype == 'student' :
            self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr_student'], self.opt['wd_student'])
        else :
            self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr_teacher'], self.opt['wd_teacher'])
    
    def reset(self):
        self.model.reset()
        if self.Mtype == 'student' :
            self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr_student'], self.opt['wd_student'])
        else :
            self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr_teacher'], self.opt['wd_teacher'])
    
    def train(self, idx_train, idx_val, idx_test, soft_target, target, features_list, adjlists, edge_metapath_indices_list, type_mask):
        train_idx_generator = index_generator(batch_size=self.batch_size, indices=idx_train.cpu())
        val_idx_generator = index_generator(batch_size=self.batch_size, indices=idx_val.cpu(), shuffle=False)
        test_idx_generator = index_generator(batch_size = self.batch_size, indices= idx_test.cpu(), shuffle = False)
        device = self.opt['device']
        neighbor_samples = 10
        total_loss = 0
        t_start = time.time()
        self.model.train()
        if self.Mtype == 'teacher':
            for iteration in range(train_idx_generator.num_iterations()):
                # forward
                train_idx_batch = train_idx_generator.next()
                train_idx_batch.sort()
                train_g_list, train_indices_list, train_idx_batch_mapped_list = parse_minibatch(
                    adjlists, edge_metapath_indices_list, train_idx_batch, device, neighbor_samples)

                logits = self.model(
                    (train_g_list, features_list, type_mask, train_indices_list, train_idx_batch_mapped_list))
                logp = F.log_softmax(logits, dim = -1)
                logp = torch.exp(logp)
                loss = self.loss_fcn(logp, target[train_idx_batch])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss = total_loss + loss
        for iteration in range(val_idx_generator.num_iterations()):
            # forward
            val_idx_batch = train_idx_generator.next()
            val_idx_batch.sort()
            val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                adjlists, edge_metapath_indices_list, val_idx_batch, device, neighbor_samples)

            logits = self.model(
                (val_g_list, features_list, type_mask, val_indices_list, val_idx_batch_mapped_list))
            logp = F.log_softmax(logits, dim = -1)
            logp = torch.exp(logp)
            if self.Mtype == 'student':
                loss = my_loss(logp, soft_target[val_idx_batch])
            else:
                loss = F.kl_div(logp, soft_target[val_idx_batch], reduction='batchmean')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss = total_loss + loss
        for iteration in range(test_idx_generator.num_iterations()):
                # forward
            test_idx_batch = test_idx_generator.next()
            test_g_list, test_indices_list, test_idx_batch_mapped_list = parse_minibatch(adjlists,
                edge_metapath_indices_list, test_idx_batch, device, neighbor_samples)
            logits = self.model((test_g_list, features_list, type_mask, test_indices_list, test_idx_batch_mapped_list))
            logp = F.log_softmax(logits, dim = -1)
            logp = torch.exp(logp)
            if self.Mtype == 'student':
                loss = my_loss(logp, soft_target[test_idx_batch])
            else:
                loss = F.kl_div(logp, soft_target[test_idx_batch], reduction='batchmean')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss = total_loss + loss
        return total_loss

    def evaluate(self, epoch, idx_val, idx_test, target, soft_target, feats, adjlists, edge_metapath_indices_list, type_mask):
        self.model.eval()
        with torch.no_grad():
            logits = self.predict(feats, adjlists, edge_metapath_indices_list, type_mask)
        logp = torch.softmax(logits, dim=-1)
        loss_val = self.loss_fcn(logp[idx_val], target[idx_val])
        if self.Mtype == 'pretrain':
            loss_test = 0.0
        else :
            loss_test = F.kl_div(logp[idx_test], soft_target[idx_test], reduction='batchmean')
        _, val_indices = torch.max(logits[idx_val], dim=1)
        val_prediction = val_indices.long().cpu().numpy()
        val_logp = logp[idx_val].float().cpu().numpy()
        t = target.long().cpu().numpy()
        auc_val = roc_auc_score(t[idx_val.long().cpu().numpy()], val_logp, multi_class='ovo')
        val_micro_f1 = f1_score(t[idx_val.long().cpu().numpy()], val_prediction, average='micro')
        val_macro_f1 = f1_score(t[idx_val.long().cpu().numpy()], val_prediction, average='macro')
        _, test_indices = torch.max(logits[idx_test], dim=1)
        test_prediction = test_indices.long().cpu().numpy()
        test_logp = logp[idx_test].float().cpu().numpy()
        test_logp[np.isinf(test_logp)] = np.nan
        auc_test = roc_auc_score(t[idx_test.long().cpu().numpy()], test_logp, multi_class='ovo')
        test_micro_f1 = f1_score(t[idx_test.long().cpu().numpy()], test_prediction, average='micro')
        test_macro_f1 = f1_score(t[idx_test.long().cpu().numpy()], test_prediction, average='macro')
        print('Epoch {:d} | Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f} | Val AUC {:.4f}'.format(epoch + 1, loss_val.item(), val_micro_f1, val_macro_f1, auc_val.item()))

        return loss_test, [(auc_val.item(), auc_test.item())], [(val_micro_f1.item(), test_micro_f1.item())], [(val_macro_f1.item(), test_macro_f1.item())]

    def predict(self, features_list, adjlists, edge_metapath_indices_list, type_mask, tau = 1):
        self.model.eval()
        with torch.no_grad():
            node_num = features_list[0].shape[0]
            idx = torch.LongTensor(np.array(range(node_num))).to(self.opt['device'])
            idx_generator = index_generator(batch_size=500, indices=idx.cpu(), shuffle=False)
            total_logits = []
            device = self.opt['device']
            neighbor_samples = self.opt['samples']
            for iteration in range(idx_generator.num_iterations()):
            # forward
                idx_batch = idx_generator.next()
                idx_batch.sort()
                g_list, indices_list, idx_batch_mapped_list = parse_minibatch(
                    adjlists, edge_metapath_indices_list, idx_batch, device, neighbor_samples)

                logits = self.model(
                    (g_list, features_list, type_mask, indices_list, idx_batch_mapped_list))
                total_logits.append(logits)
            logits = torch.cat(total_logits, dim = 0)
            logits = F.log_softmax(logits, dim=-1)
            logits = torch.exp(logits)
        return logits.detach()
