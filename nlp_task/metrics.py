from sklearn.metrics import roc_auc_score, f1_score, hamming_loss
import numpy as np

def set_score_pre(input_batch, target_batch, predict_batch, str2tok):
    s = []
    s2 = []
    for b in range(target_batch.shape[0]):
        trim_target = []
        trim_predict = []
        for t in target_batch[b]:
            if t > 1:
                trim_target.append(t)
        for t in predict_batch[b]:
            if t > 1:
                trim_predict.append(t)
        if np.random.rand()>1:
            print('{} vs {}'.format(trim_target, trim_predict))
        acc = len(set(trim_target).intersection(set(trim_predict)))/len(set(trim_target))
        acc2=0
        if len(set(trim_predict))>0:
            acc2 = len(set(trim_target).intersection(set(trim_predict))) / len(trim_predict)
        s.append(acc)
        s2.append(acc2)
    return np.mean(s2), np.mean(s)#prec, recall

def roc_auc(input_batch, target_batch, predict_batch, str2tok):
    all_auc_macro=[]
    all_auc_micro = []
    for b in range(target_batch.shape[0]):
        target = np.zeros(predict_batch.shape[-1])
        for t in target_batch[b]:
            if t>1:
                target[t]=1
        all_auc_macro.append(roc_auc_score(target, predict_batch[b], average='macro'))
        all_auc_micro.append(roc_auc_score(target, predict_batch[b], average='micro'))
    return np.mean(all_auc_macro),np.mean(all_auc_micro)


def fscore(input_batch, target_batch, predict_batch, str2tok):
    all_auc_macro=[]
    all_auc_micro = []
    nlabel=len(str2tok)
    for b in range(target_batch.shape[0]):
        target = np.zeros(nlabel)
        predict = np.zeros(nlabel)
        for t in target_batch[b]:
            if t>1:
                target[t]=1
        for t in predict_batch[b]:
            if t>1:
                predict[t]=1
        all_auc_macro.append(f1_score(target, predict, average='macro'))
        all_auc_micro.append(f1_score(target, predict, average='micro'))
    return np.mean(all_auc_macro),np.mean(all_auc_micro)

def hloss(input_batch, target_batch, predict_batch, str2tok):
    ham=[]
    nlabel=len(str2tok)
    for b in range(target_batch.shape[0]):
        target = np.zeros(nlabel)
        predict = np.zeros(nlabel)
        for t in target_batch[b]:
            if t>1:
                target[t]=1
        for t in predict_batch[b]:
            if t>1:
                predict[t]=1
        ham.append(hamming_loss(target, predict))

    return np.mean(ham)