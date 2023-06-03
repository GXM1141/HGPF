import time
import torch
import torch.nn.functional as F
import numpy as np
from utils import my_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

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

class Trainer(object):
    def __init__(self, opt, model, Mtype):
        self.opt = opt
        self.model = model
        self.loss_fcn = torch.nn.CrossEntropyLoss()
        self.Mtype = Mtype
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
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
    
    def train(self, idx_train, idx_val, idx_test, soft_target, target, inputs):
        t0 = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        if self.Mtype == 'student':
            logits, a, b, att, gt = self.model(inputs)
        else:
            logits, a, b, _, _ = self.model(inputs)
        #logp = F.log_softmax(logits, dim = -1)
        #logp = torch.exp(logp)
        logp = logits
        idx_no_train = torch.LongTensor(np.setdiff1d(np.array(range(len(target))), idx_train.cpu())).to(self.opt['device'])
        if self.Mtype == 'student':
            logp = F.log_softmax(logits, dim = -1)
            logp = torch.exp(logp)
            #loss = F.kl_div(logp[idx_no_train], soft_target[idx_no_train], reduction='batchmean')
            
            loss = my_loss(logp[idx_no_train], soft_target[idx_no_train])
            #loss = F.kl_div(logp[idx], soft_target[idx], reduction='batchmean')
        else:
            logp = F.log_softmax(logits, dim = -1)
            logp = torch.exp(logp)
            #loss = my_loss(logp[idx_no_train], soft_target[idx_no_train]) + self.loss_fcn(logp[idx_train], target[idx_train])
            loss = F.kl_div(logp[idx_no_train], soft_target[idx_no_train], reduction='batchmean') + self.loss_fcn(logp[idx_train], target[idx_train])
        #-torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, epoch, idx_val, idx_test, target, soft_target, inputs):
        self.model.eval()
        with torch.no_grad():
            if self.Mtype == 'student':
                logits, a, b, att, gt = self.model(inputs)
            else:
                logits, _, _, _, _ = self.model(inputs)
                att = 0
                gt = 0
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

        return loss_test, [(auc_val.item(), auc_test.item())], [(val_micro_f1.item(), test_micro_f1.item())], [(val_macro_f1.item(), test_macro_f1.item())], att, gt

    def predict(self, inputs, tau = 1):
        self.model.eval()

        if self.Mtype == 'student':
            logits, a, b, _, _ = self.model(inputs)
        else:
            logits, _, _, _, _ = self.model(inputs)
        logits = F.log_softmax(logits, dim=-1)
        logits = torch.exp(logits)
        return logits.detach()

