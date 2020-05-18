import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import pickle
import getopt
import time
import sys
import os
import pandas
import nltk

from keras.models import load_model
import csv
from sklearn import metrics
from datetime import datetime, timedelta
from collections import Counter
import distance
from jellyfish._jellyfish import damerau_levenshtein_distance

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')

from dnc import DNC
from recurrent_controller import StatelessRecurrentController







def set_score_pre(target_batch, predict_batch):
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
        # if np.random.rand()>0.999:
        #     print('{} vs {}'.format(trim_target, trim_predict))
        acc = len(set(trim_target).intersection(set(trim_predict)))/len(set(trim_target))
        acc2=0
        if len(set(trim_predict))>0:
            acc2 = len(set(trim_target).intersection(set(trim_predict))) / len(trim_predict)
        s.append(acc)
        s2.append(acc2)
    return np.mean(s), np.mean(s2)


def batch_mae(reals, preds, pprint=0.999):
    avgs=0
    c=0
    for i,real in enumerate(reals):
        # if np.random.rand() > pprint:
        #     print('{} vs {}'.format(reals[i], preds[i]))
        for r, p in zip(reals[i],preds[i]):
            avgs += np.abs(r-p)
        c+=1

    return avgs/c



def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def load(path):
    return pickle.load(open(path, 'rb'))

def onehot(index, size):
    # print('-----')
    # print(index)
    vec = np.zeros(size, dtype=np.float32)
    vec[int(index)] = 1.0
    return vec

def prepare_sample(dig_list, proc_list, word_space_size_input, word_space_size_output, index=-1):
    if index<0:
        index = int(np.random.choice(len(dig_list),1))

    # print('\n{}'.format(index))
    ins=dig_list[index]
    ose=proc_list[index]
    seq_len = len(ins) + 1 + len(ose)
    input_vec = np.zeros(seq_len)
    for iii, token in enumerate(ins):
        input_vec[iii] = token
    input_vec[len(ins)] = 1
    output_vec = np.zeros(seq_len)
    decoder_point = len(ins) + 1
    for iii, token in enumerate(ose):
        output_vec[decoder_point + iii] = token
    input_vec = np.array([[onehot(code, word_space_size_input) for code in input_vec]])
    output_vec = np.array([[onehot(code, word_space_size_output) for code in output_vec]])
    return input_vec, output_vec, seq_len, decoder_point, index

def prepare_sample_batch(dig_list,proc_list,word_space_size_input,word_space_size_output, bs, lm_train=False):
    if isinstance(bs, int):
        indexs = np.random.choice(len(dig_list),bs,replace=False)
    else:
        #print('from {} to {}'.format(bs[0],bs[1]))
        indexs=list(range(bs[0],bs[1]))
    minlen=0
    moutlne=0
    for index in indexs:
        minlen=max(len(dig_list[index]),minlen)
        moutlne = max(len(proc_list[index]+[0]), moutlne)
    # moutlne*=2
    input_vecs=[]
    output_vecs=[]
    seq_len = minlen + 1 + moutlne
    decoder_point = minlen + 1
    out_list=[]
    masks=[]
    for index in indexs:
        # print('\n{}'.format(index))
        ins=dig_list[index]
        ose=proc_list[index]+[0]
        out_list.append(ose)
        input_vec = np.zeros(seq_len)
        output_vec = np.zeros(seq_len)
        mask=np.zeros(seq_len, dtype=np.bool)
        for iii, token in enumerate(ins):
            input_vec[minlen-len(ins)+iii] = token
            if lm_train:
                output_vec[minlen - len(ins) + iii+1] = token
                mask[minlen - len(ins) + iii+1] = True
        input_vec[minlen] = 1




        for iii, token in enumerate(ose):
            output_vec[decoder_point + iii] = token
            mask[decoder_point + iii]=True

        # print(ins)
        # print(ose)
        # print(input_vec)
        # print(output_vec)
        # print('====')
        input_vec = [onehot(code, word_space_size_input) for code in input_vec]
        output_vec = [onehot(code, word_space_size_output) for code in output_vec]
        input_vecs.append(input_vec)
        output_vecs.append(output_vec)
        masks.append(mask)

    # raise False
    return np.asarray(input_vecs), np.asarray(output_vecs), seq_len, decoder_point, np.asarray(masks), out_list

def prepare_sample_batch_feature(X,y, bs):
    if isinstance(bs, int):
        indexs = np.random.choice(len(X),bs,replace=False)
    else:
        #print('from {} to {}'.format(bs[0],bs[1]))
        indexs=list(range(bs[0],bs[1]))


    minlen=X.shape[1]
    moutlen=y.shape[1]
    # moutlne*=2
    input_vecs=[]
    output_vecs=[]
    seq_len = minlen + 1 + moutlen
    decoder_point = minlen + 1
    out_list=[]
    masks=[]

    for index in indexs:
        # print('\n{}'.format(index))
        ins=X[index]
        if y.shape[2]>1:
            ose=np.zeros((moutlen, y.shape[2]+2))
            ose[:moutlen,2:]=y[index]
            ro=[]
            for l in range(moutlen):
                ro.append(np.argmax(ose[l],axis=-1))
        else:
            ose = np.zeros((moutlen, 1))
            ose[:moutlen] = y[index]
            ro = []
            for l in range(moutlen):
                ro.append(y[index][l])
        # print(ro)
        # print(y[index])
        # print(ose[:moutlen])
        # print(np.argmax(ose[l],axis=-1))
        # raise  False
        out_list.append(ro)
        input_vec = np.zeros((seq_len, X.shape[2]))
        if y.shape[2]>1:
            output_vec = np.zeros((seq_len, y.shape[2]+2))
        else:
            output_vec = np.zeros((seq_len,1))
        mask=np.zeros(seq_len, dtype=np.bool)
        for iii, token in enumerate(ins):
            input_vec[minlen-len(ins)+iii] = token
        input_vec[minlen] = np.ones(X.shape[2])




        for iii, token in enumerate(ose):
            output_vec[decoder_point + iii] = token
            mask[decoder_point + iii]=True

        # print(ins)
        # print(ose)
        # print(input_vec)
        # print(output_vec)
        # print('====')
        #
        # raise  False

        input_vecs.append(input_vec)
        output_vecs.append(output_vec)
        masks.append(mask)

    # raise False
    return np.asarray(input_vecs), np.asarray(output_vecs), seq_len, decoder_point, np.asarray(masks), out_list


def prepare_sample_batch_feature_mix(X,y1,y2,bs):
    if isinstance(bs, int):
        indexs = np.random.choice(len(X),bs,replace=False)
    else:
        #print('from {} to {}'.format(bs[0],bs[1]))
        indexs=list(range(bs[0],bs[1]))


    minlen=X.shape[1]
    moutlen=y1.shape[1]
    # moutlne*=2
    input_vecs=[]
    output_vecs=[]
    seq_len = minlen + 1 + moutlen
    decoder_point = minlen + 1
    out_list1 = []
    out_list2 = []
    masks=[]

    for index in indexs:
        # print('\n{}'.format(index))
        ins=X[index]
        ose1=np.zeros((moutlen, y1.shape[2]+2))
        ose1[:moutlen,2:]=y1[index]
        ro1=[]
        for l in range(moutlen):
            ro1.append(np.argmax(ose1[l],axis=-1))
        ose2 = np.zeros((moutlen, 1))
        ose2[:moutlen] = y2[index]
        ro2 = []
        for l in range(moutlen):
            ro2.append(y2[index][l])
        # print(ro)
        # print(y[index])
        # print(ose[:moutlen])
        # print(np.argmax(ose[l],axis=-1))
        # raise  False
        out_list1.append(ro1)
        out_list2.append(ro2)
        input_vec = np.zeros((seq_len, X.shape[2]))
        output_vec1 = np.zeros((seq_len, y1.shape[2]+2))
        output_vec2 = np.zeros((seq_len,1))
        mask=np.zeros(seq_len, dtype=np.bool)
        for iii, token in enumerate(ins):
            input_vec[minlen-len(ins)+iii] = token
        input_vec[minlen] = np.ones(X.shape[2])




        for iii, token in enumerate(ose1):
            output_vec1[decoder_point + iii] = token
            mask[decoder_point + iii]=True
        for iii, token in enumerate(ose2):
            output_vec2[decoder_point + iii] = token

        output_vec = np.concatenate([output_vec1,output_vec2],axis=-1)

        # print(ins)
        # print(ose)
        # print(input_vec)
        # print(output_vec)
        # print('====')
        #
        # raise  False

        input_vecs.append(input_vec)
        output_vecs.append(output_vec)
        masks.append(mask)

    # raise False
    return np.asarray(input_vecs), np.asarray(output_vecs), seq_len, decoder_point, np.asarray(masks), out_list1, out_list2

def load_dict(dir='./data/BusinessProcess/Moodle'):
    return pickle.load(open(dir+'/event_vocab.pkl','rb'))

def load_single_sequence(fname):
    seqs=[]
    rl=''
    for l in open(fname):
        if l.strip()[-1]==']':
            if rl!='':
                l=rl
            s=l.strip()[1:-1].strip().split()
            seqs.append([int(x)+1 for x in s])
            rl=''
        else:
            rl+=l+' '
    return seqs




def load_np_data(dir):
    X=pickle.load(open(dir+'/ps_Xsmall.pkl','rb'),encoding='latin1')
    y_a = pickle.load(open(dir + '/ps_yasmall.pkl','rb'),encoding='latin1')
    y_t = pickle.load(open(dir + '/ps_ytsmall.pkl','rb'),encoding='latin1')

    return X, y_a, y_t


def get_nb_params_shape(shape):

    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params

def count_number_trainable_params():
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params



def bpi_task_mix():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_bpi_mix')


    X, y_a, y_t = load_np_data('./data/bpi/')

    if len(y_a.shape)<3:
        y_a= np.reshape(y_a,[y_a.shape[0],1,y_a.shape[1]])

    if len(y_t.shape) <2:
        y_t = np.reshape(y_t, [y_t.shape[0], 1, 1])
    batch_size = X.shape[1]
    print('bs {}'.format(batch_size))
    all_index = list(range(len(X)))

    # np.random.shuffle(all_index)

    train_index = all_index[:int(len(X) * 0.8)]
    test_index = all_index[int(len(X) * 0.8):]

    # print(X.shape[1])
    # raise False

    X_train = X[train_index]
    X_test = X[test_index]

    y_a_train = y_a[train_index]
    y_a_test = y_a[test_index]

    y_t_train = y_t[train_index]
    y_t_test = y_t[test_index]



    dim_in = X_train.shape[2]
    dim_out = y_a_train.shape[2]+ y_t_train.shape[1]

    print('num train {}'.format(len(X_train)))
    print('num test {}'.format(len(X_test)))
    print('dim in  {}'.format(dim_in))
    print('dim out {}'.format(dim_out))


    input_size = dim_in
    output_size = dim_out+2
    sequence_max_length = 100

    words_count = 5
    word_size = 20
    read_heads = 1



    iterations = 20000
    start_step = 0


    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim = 100,
                use_emb=False,
                use_mem=True,
                decoder_mode=True,
                dual_controller=True,
                write_protect=True,
                attend_dim=0
            )

            output1, output2, loss, apply_gradients = ncomputer.build_mix_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            print(count_number_trainable_params())
            llprint("Done!\n")


            last_100_losses = []

            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1
            minloss=1000
            start_time_100 = time.time()
            avg_100_time = 0.
            avg_counter = 0
            train_writer = tf.summary.FileWriter('./data/log_bpi/', session.graph)
            for i in range(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))
                    input_vec, output_vec, seq_len, decoder_point, masks, rout_list1, rout_list2 = \
                        prepare_sample_batch_feature_mix(X_train,y_a_train,y_t_train, bs=batch_size)




                    summerize = (i % 100 == 0)
                    take_checkpoint = (i != 0) and (i % end == 0)
                    if i<=1000000:
                        loss_value, _ = session.run([
                            loss,
                            apply_gradients
                        ], feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks
                        })
                    else:
                        loss_value = session.run(loss, feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks
                        })

                    last_100_losses.append(loss_value)

                    if summerize:
                        llprint("\n\t episode %d -->Avg. Cross-Entropy: %.7f\n" % (i, np.mean(last_100_losses)))
                        summary = tf.Summary()
                        summary.value.add(tag='batch_train_loss', simple_value=np.mean(last_100_losses))
                        trscores1 = []
                        trscores2 = []
                        for ii in range(10):
                            input_vec, output_vec, seq_len, decoder_point, masks, rout_list1, rout_list2 =\
                                prepare_sample_batch_feature_mix(X_train,y_a_train,y_t_train, bs=batch_size)


                            out1,out2 = session.run([output1,output2], feed_dict={ncomputer.input_data: input_vec,
                                                                                  ncomputer.target_output: output_vec,
                                                                 ncomputer.decoder_point: decoder_point,
                                                                 ncomputer.sequence_length: seq_len,
                                                                 ncomputer.mask:masks})

                            out1 = np.reshape(np.asarray(out1), [-1, seq_len, output_size-1])
                            out1 = np.argmax(out1, axis=-1)
                            bout_list1 = []
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out1.shape[0]):
                                out_list1 = []
                                for io in range(decoder_point, out1.shape[1]):
                                    out_list1.append(out1[b][io])
                                bout_list1.append(out_list1)
                            # print(rout_list)
                            # print(bout_list)
                            # raise False
                            pre, rec = set_score_pre(np.asarray(rout_list1), np.asarray(bout_list1))
                            trscores1.append(pre)

                            out2 = np.reshape(np.asarray(out2), [-1, seq_len])
                            rout_list2 = np.reshape(np.asarray(rout_list2),[-1,y_t_train.shape[1]])
                            bout_list2=[]
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out2.shape[0]):
                                out_list2 = []
                                for io in range(decoder_point, out2.shape[1]):
                                    out_list2.append(out2[b][io])
                                bout_list2.append(out_list2)
                            # print(rout_list)
                            # print(bout_list)
                            # raise False
                            trscores2.append(batch_mae(rout_list2, bout_list2,0.95))



                        print('-----')

                        tescores1 = []
                        tescores2 = []

                        losses = []
                        ntb=len(X_test)//batch_size+1
                        for ii in range(ntb):
                            if ii*batch_size==len(X_test):
                                break
                            bs=[ii*batch_size, min((ii+1)*batch_size,len(X_test))]
                            rs = bs[1] - bs[0]
                            if bs[1]>=len(X_test):
                                bs=[len(X_test)-batch_size, len(X_test)]

                            input_vec, output_vec, seq_len, decoder_point, masks, rout_list1, rout_list2 = \
                                prepare_sample_batch_feature_mix(X_test, y_a_test, y_t_test, bs=bs)

                            out1, out2, loss_v = session.run([output1, output2, loss], feed_dict={ncomputer.input_data: input_vec,
                                                                                    ncomputer.decoder_point: decoder_point,
                                                                                    ncomputer.target_output: output_vec,
                                                                                    ncomputer.sequence_length: seq_len,
                                                                                    ncomputer.mask: masks})

                            losses.append(loss_v)

                            out1 = np.reshape(np.asarray(out1), [-1, seq_len, output_size - 1])
                            out1 = np.argmax(out1, axis=-1)
                            bout_list1 = []
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out1.shape[0]):
                                out_list1 = []
                                for io in range(decoder_point, out1.shape[1]):
                                    out_list1.append(out1[b][io])
                                bout_list1.append(out_list1)
                            pre, rec = set_score_pre(np.asarray(rout_list1[:rs]), np.asarray(bout_list1[:rs]))
                            tescores1.append(pre)


                            out2 = np.reshape(np.asarray(out2), [-1, seq_len])
                            rout_list2 = np.reshape(np.asarray(rout_list2), [-1, y_t_train.shape[1]])
                            bout_list2 = []
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out2.shape[0]):
                                out_list2 = []
                                for io in range(decoder_point, out2.shape[1]):
                                    out_list2.append(out2[b][io])
                                bout_list2.append(out_list2)

                            tescores2.append(batch_mae(rout_list2[:rs], bout_list2[:rs], 0.995))


                        tloss = np.mean(losses)
                        print('test lost {} vs min loss {}'.format(tloss,minloss))
                        print('tr pre {} vs te pre {}'.format(np.mean(trscores1), np.mean(tescores1)))
                        print('tr mae {} vs te mae {}'.format(np.mean(trscores2), np.mean(tescores2)))
                        summary.value.add(tag='test_pre', simple_value=np.mean(tescores1))
                        summary.value.add(tag='test_mae', simple_value=np.mean(tescores2))
                        summary.value.add(tag='test_loss', simple_value=tloss)

                        train_writer.add_summary(summary, i)
                        train_writer.flush()

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                        print("\tApprox. time to completion: %.2f hours" % (estimated_time))

                        start_time_100 = time.time()
                        last_100_losses = []

                    if minloss>tloss:
                        minloss=tloss
                        ncomputer.save(session, ckpts_dir, ncomputer.print_config())
                        print("\nSaving Checkpoint ...\n"),



                except KeyboardInterrupt:

                    llprint("\nSaving Checkpoint ... ")

def exact_bpi_test_mix():
    chars = pickle.load(open('./data/bpi/bpi_tmp/chars.pkl', 'rb'), encoding='latin1')
    char_indices = pickle.load(open('./data/bpi/bpi_tmp/char_indices.pkl', 'rb'), encoding='latin1')
    target_indices_char = pickle.load(open('./data/bpi/bpi_tmp/target_indices_char.pkl', 'rb'), encoding='latin1')
    target_char_indices = pickle.load(open('./data/bpi/bpi_tmp/target_char_indices.pkl', 'rb'), encoding='latin1')
    target_chars = pickle.load(open('./data/bpi/bpi_tmp/target_chars.pkl', 'rb'), encoding='latin1')
    maxlen = pickle.load(open('./data/bpi/bpi_tmp/maxlen.pkl', 'rb'), encoding='latin1')
    divisor = pickle.load(open('./data/bpi/bpi_tmp/divisor.pkl', 'rb'), encoding='latin1')
    divisor2 = pickle.load(open('./data/bpi/bpi_tmp/divisor2.pkl', 'rb'), encoding='latin1')
    lines = pickle.load(open('./data/bpi/bpi_tmp/lines.pkl', 'rb'), encoding='latin1')
    lines_t = pickle.load(open('./data/bpi/bpi_tmp/lines_t.pkl', 'rb'), encoding='latin1')
    lines_t2 = pickle.load(open('./data/bpi/bpi_tmp/lines_t2.pkl', 'rb'), encoding='latin1')
    lines_t3 = pickle.load(open('./data/bpi/bpi_tmp/lines_t3.pkl', 'rb'), encoding='bytes')

    # lines_t4 = fold1_t4 + fold2_t4

    # set parameters
    predict_size = 1


    # define helper functions
    def encode(sentence, times, times3, maxlen=maxlen):
        num_features = len(chars) + 5
        X = np.zeros((1, maxlen, num_features), dtype=np.float32)
        leftpad = maxlen - len(sentence)
        times2 = np.cumsum(times)
        for t, char in enumerate(sentence):
            midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = times3[t] - midnight
            multiset_abstraction = Counter(sentence[:t + 1])
            for c in chars:
                if c == char:
                    X[0, t + leftpad, char_indices[c]] = 1
            X[0, t + leftpad, len(chars)] = t + 1
            X[0, t + leftpad, len(chars) + 1] = times[t] / divisor
            X[0, t + leftpad, len(chars) + 2] = times2[t] / divisor2
            X[0, t + leftpad, len(chars) + 3] = timesincemidnight.seconds / 86400
            X[0, t + leftpad, len(chars) + 4] = times3[t].weekday() / 7
        return X

    def getSymbol(predictions):
        maxPrediction = 0
        symbol = ''
        i = 0;
        print(predictions)
        for prediction in predictions:
            if (prediction >= maxPrediction):
                maxPrediction = prediction
                symbol = target_indices_char[i]
            i += 1
        return symbol

    one_ahead_gt = []
    one_ahead_pred = []

    two_ahead_gt = []
    two_ahead_pred = []

    three_ahead_gt = []
    three_ahead_pred = []

    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_bpi_mix')
    batch_size = 1

    X, y_a, y_t = load_np_data('./data/bpi/')

    if len(y_a.shape) < 3:
        y_a = np.reshape(y_a, [y_a.shape[0], 1, y_a.shape[1]])

    if len(y_t.shape) <2:
        y_t = np.reshape(y_t, [y_t.shape[0], 1, 1])

    num_sample = len(X)
    X_train = X[:int(0.8 * num_sample)]
    X_test = X[int(0.8 * num_sample):]

    y_a_train = y_a[:int(0.8 * num_sample)]
    y_a_test = y_a[int(0.8 * num_sample):]

    y_t_train = y_t[:int(0.8 * num_sample)]
    y_t_test = y_t[int(0.8 * num_sample):]

    dim_in = X_train.shape[2]
    dim_out = y_a_train.shape[2] + y_t_train.shape[1]

    print('num train {}'.format(len(X_train)))
    print('num test {}'.format(len(X_test)))
    print('dim in  {}'.format(dim_in))
    print('dim out {}'.format(dim_out))

    input_size = dim_in
    output_size = dim_out + 2
    sequence_max_length = 100

    words_count = 5
    word_size = 20
    read_heads = 1
    test_data=[]
    iterations = 10000
    start_step = 0

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim=100,
                use_emb=False,
                use_mem=True,
                decoder_mode=True,
                dual_controller=True,
                write_protect=True,
                attend_dim=0
            )

            output1, output2, loss, apply_gradients = ncomputer.build_mix_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            ncomputer.restore(session, ckpts_dir, ncomputer.print_config())
            llprint("Done!\n")

            print('-----')

            eventlog = "bpi.csv"
            # make predictions
            with open('./data/bpi/next_activity_and_time_%s' % eventlog, 'w') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(
                    ["Prefix length", "Groud truth", "Predicted", "Levenshtein", "Damerau", "Jaccard", "Ground truth times",
                     "Predicted times", "RMSE", "MAE", "Median AE"])
                for prefix_size in range(2, maxlen):
                    print(prefix_size)
                    for line, times, times3 in zip(lines, lines_t, lines_t3):
                        times.append(0)
                        cropped_line = ''.join(line[:prefix_size])
                        cropped_times = times[:prefix_size]
                        cropped_times3 = times3[:prefix_size]
                        if '!' in cropped_line:
                            continue  # make no prediction for this case, since this case has ended already
                        ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
                        ground_truth_t = times[prefix_size:prefix_size + predict_size]
                        predicted = ''
                        predicted_t = []
                        for i in range(predict_size):
                            if len(ground_truth) <= i:
                                continue
                            enc = encode(cropped_line, cropped_times, cropped_times3)
                            test_data.append(enc)

                            input_vec, output_vec, seq_len, decoder_point, masks, rout_list1, rout_list2 = \
                                prepare_sample_batch_feature_mix(enc,np.asarray([y_a_test[0]]), np.asarray([y_t_test[0]]), 1)

                            out1, out2, loss_v = session.run([output1, output2, loss],
                                                             feed_dict={ncomputer.input_data: input_vec,
                                                                        ncomputer.decoder_point: decoder_point,
                                                                        ncomputer.target_output: output_vec,
                                                                        ncomputer.sequence_length: seq_len,
                                                                        ncomputer.mask: masks})

                            out1 = np.reshape(np.asarray(out1), [-1, seq_len, output_size-1])
                            out1 = np.argmax(out1, axis=-1)
                            bout1 = []
                            for io in range(decoder_point, out1.shape[1]):
                                bout1.append(max(out1[0][io] - 2, 1))
                            y_char = bout1[0]

                            out2 = np.reshape(np.asarray(out2), [-1, seq_len])
                            bout2 = []
                            for io in range(decoder_point, out2.shape[1]):
                                bout2.append(out2[0][io])

                            # print(y_char)
                            y_t = bout2[0]
                            prediction = target_indices_char[y_char]
                            cropped_line += prediction
                            if y_t < 0:
                                y_t = 0
                            cropped_times.append(y_t)
                            y_t = y_t * divisor
                            cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
                            predicted_t.append(y_t)
                            if i == 0:
                                if len(ground_truth_t) > 0:
                                    one_ahead_pred.append(y_t)
                                    one_ahead_gt.append(ground_truth_t[0])
                            if i == 1:
                                if len(ground_truth_t) > 1:
                                    two_ahead_pred.append(y_t)
                                    two_ahead_gt.append(ground_truth_t[1])
                            if i == 2:
                                if len(ground_truth_t) > 2:
                                    three_ahead_pred.append(y_t)
                                    three_ahead_gt.append(ground_truth_t[2])
                            if prediction == '!':  # end of case was just predicted, therefore, stop predicting further into the future
                                print('! predicted, end case')
                                break
                            predicted += prediction
                        output = []
                        if len(ground_truth) > 0:
                            output.append(prefix_size)
                            output.append((ground_truth).encode("utf-8"))
                            output.append((predicted).encode("utf-8"))
                            output.append(1 - distance.nlevenshtein(predicted, ground_truth))
                            dls = 1 - (
                            damerau_levenshtein_distance((predicted), (ground_truth)) / max(len(predicted), len(ground_truth)))
                            if dls < 0:
                                dls = 0  # we encountered problems with Damerau-Levenshtein Similarity on some linux machines where the default character encoding of the operating system caused it to be negative, this should never be the case
                            output.append(dls)
                            output.append(1 - distance.jaccard(predicted, ground_truth))
                            output.append('; '.join(str(x) for x in ground_truth_t))
                            output.append('; '.join(str(x) for x in predicted_t))
                            if len(predicted_t) > len(
                                    ground_truth_t):  # if predicted more events than length of case, only use needed number of events for time evaluation
                                predicted_t = predicted_t[:len(ground_truth_t)]
                            if len(ground_truth_t) > len(
                                    predicted_t):  # if predicted less events than length of case, put 0 as placeholder prediction
                                predicted_t.extend(range(len(ground_truth_t) - len(predicted_t)))
                            if len(ground_truth_t) > 0 and len(predicted_t) > 0:
                                output.append('')
                                output.append(metrics.mean_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                                output.append(metrics.median_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                            else:
                                output.append('')
                                output.append('')
                                output.append('')
                            spamwriter.writerow(output)


if __name__ == '__main__':
    bpi_task_mix()
    # exact_bpi_test_mix()
