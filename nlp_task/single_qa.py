import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import pickle
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from dnc_v2 import DNC
from recurrent_controller import StatelessRecurrentController
import mimic_processing
import beam_search

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


def metric_score(brin,brout,bout_list, tok2str):


def prepare_sample_batch(diag_list,word_space_size_input,word_space_size_output, bs):
    if isinstance(bs, int):
        indexs = np.random.choice(len(diag_list),bs,replace=True)
    else:
        #print('from {} to {}'.format(bs[0],bs[1]))
        indexs=list(range(bs[0],bs[1]))
        bs = bs[1]-bs[0]
    minlen=0
    moutlne=0

    for index in indexs:
        index2=index
        if index<0:
            index2 = (index+bs) % len(diag_list)
        minlen=max(len(diag_list[index2][0]),minlen)
        moutlne = max(len(diag_list[index2][1]), moutlne)
    # moutlne*=2
    input_vecs=[]
    output_vecs=[]
    seq_len = minlen + 1
    decoder_length = moutlne+2
    out_list=[]
    in_list=[]
    masks=[]
    for index in indexs:
        index2 = index
        if index < 0:
            index2 = (index+bs) % len(diag_list)
        # print('\n{}'.format(index))
        ins=diag_list[index2][0]
        in_list.append(ins)
        ose=[1]+diag_list[index2][1]+[2]
        out_list.append(ose)
        input_vec = np.zeros(seq_len)
        output_vec = np.zeros(decoder_length)
        mask=np.zeros(decoder_length, dtype=np.bool)
        for iii, token in enumerate(ins):
            input_vec[minlen-len(ins)+iii] = token
            # if lm_train:
            #     output_vec[minlen - len(ins) + iii+1] = token
            #     mask[minlen - len(ins) + iii+1] = True
        input_vec[minlen] = 2




        for iii, token in enumerate(ose):
            output_vec[iii] = token
            mask[iii]=True

        # print(ins)
        # print(ose)
        # print(input_vec)
        # print(output_vec)
        # print('====')

        if SAMPLED_SOFTMAX == 0:
            output_vec = np.array([onehot(code, word_space_size_output) for code in output_vec])
        else:
            output_vec = np.reshape(np.asarray(output_vec), (-1, 1))

        input_vec = [onehot(code, word_space_size_input) for code in input_vec]
        input_vecs.append(input_vec)
        output_vecs.append(output_vec)
        masks.append(mask)

    # raise False
    return np.asarray(input_vecs), np.asarray(output_vecs), seq_len, decoder_length, np.asarray(masks), out_list, in_list



def single_qa_task(args):
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/save/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_qa_{}_single_in_single_out_persit'.format(args.task))

    llprint("Loading Data ... ")

    llprint("Done!\n")
    str2tok, tok2str, dialogs = pickle.load(open(args.data_dir, 'rb'))

    all_index = list(range(len(dialogs)))
    train_index = all_index[:int(len(dialogs)-5000)]
    valid_index = all_index[int(len(dialogs) -2500):int(len(dialogs) * 1)]
    test_index = all_index[int(len(dialogs) - 5000):int(len(dialogs) -2500)]

    dialogs_list_train = [dialogs[i] for i in train_index]

    dialogs_list_valid = [dialogs[i] for i in valid_index]

    dialogs_list_test = [dialogs[i] for i in test_index]

    print('num_dialogs {}'.format(len(dialogs)))
    print('num train {}'.format(len(dialogs_list_train)))
    print('num valid {}'.format(len(dialogs_list_valid)))
    print('num test {}'.format(len(dialogs_list_test)))
    print('dim in  {} {}'.format(len(str2tok), len(str2tok)))
    print('dim out {}'.format(len(str2tok)))

    batch_size = args.batch_size
    input_size = len(str2tok)
    output_size = len(str2tok)

    words_count = args.mem_size
    word_size = args.word_size
    read_heads = args.read_heads

    learning_rate = args.learning_rate
    momentum = 0.9

    iterations = args.iterations
    start_step = 0


    config = tf.ConfigProto(device_count={'CPU': args.cpu_num})
    config.intra_op_parallelism_threads = args.cpu_num

    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = args.gpu_ratio
    graph = tf.Graph()
    with graph.as_default():
        tf.contrib.framework.get_or_create_global_step()
        with tf.Session(graph=graph, config=config) as session:

            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                output_size,
                words_count,
                word_size,
                read_heads,
                batch_size,
                use_mem=args.use_mem,
                dual_emb=False,
                use_emb_encoder=True,
                use_emb_decoder=True,
                decoder_mode=True,
                dual_controller=True,
                write_protect=True,
                emb_size=args.emb_dim,
                hidden_controller_dim=args.hidden_dim,
                use_teacher=args.use_teacher,
                attend_dim=args.attend,
                sampled_loss_dim=args.sampled_loss_dim,
                enable_drop_out=args.drop_out_keep>0,
                nlayer=args.nlayer,
                name='vanila'
            )
            # optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # lr = tf.train.exponential_decay(
            #     learning_rate,
            #     tf.train.get_global_step(),
            #     args.lr_decay_step,
            #     args.lr_decay_rate,
            #     staircase=True,
            #     name="learning_rate")
            # optimizer = tf.train.AdamOptimizer(lr)
            if args.sampled_loss_dim==0:
                _, prob, loss, apply_gradients = ncomputer.build_loss_function_mask(optimizer, clip_s=[-5, 5])
            else:
                _, prob, loss, apply_gradients = ncomputer.build_sampled_loss_mask(optimizer, sampled_sm=5000)
            llprint("Done!\n")
            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")

            if args.from_checkpoint is not '':
                if args.from_checkpoint=='default':
                    from_checkpoint = ncomputer.print_config()
                else:
                    from_checkpoint = args.from_checkpoint
                llprint("Restoring Checkpoint %s ... " % from_checkpoint)
                ncomputer.restore(session, ckpts_dir, from_checkpoint)
                llprint("Done!\n")


            last_100_losses = []

            start = 1 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1
            if args.mode == 'test':
                start=0
                end = start
                dialogs_list_valid = dialogs_list_test


            start_time_100 = time.time()

            avg_100_time = 0.
            avg_counter = 0
            if args.mode=='train':
                log_dir = './data/summary/log_{}_{}/'.format(args.task, args.use_pretrain_emb)
                if not os.path.isdir(log_dir):
                    os.mkdir(log_dir)
                log_dir = '{}/{}/'.format(log_dir,ncomputer.print_config())
                if not os.path.isdir(log_dir):
                    os.mkdir(log_dir)
                train_writer = tf.summary.FileWriter(log_dir, session.graph)
            min_tloss=0
            for i in range(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))
                    input_data, target_output, seq_len, decoder_length, masks,_,_ = \
                        prepare_sample_batch(dialogs_list_train, input_size, output_size, batch_size)

                    summerize = (i % args.valid_time == 0)
                    if args.mode == 'train':
                        loss_value, _= session.run([
                            loss,
                            apply_gradients
                        ], feed_dict={
                            ncomputer.input_encoder: input_data,
                            ncomputer.input_decoder: target_output,
                            ncomputer.target_output: target_output,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decode_length: decoder_length,
                            ncomputer.mask: masks,
                            ncomputer.teacher_force: ncomputer.get_bool_rand_incremental(decoder_length, prob_true_max=0.5),
                            ncomputer.drop_out_keep: args.drop_out_keep
                        })

                        last_100_losses.append(loss_value)

                    tloss=10000000
                    tpre=0
                    if summerize:
                        llprint("\n\t episode %d -->Avg. Cross-Entropy: %.7f\n" % (i, np.mean(last_100_losses)))
                        trscores = []
                        tescores = []
                        tescores4 = []
                        distinct2 = []
                        bows=[]
                        if args.mode=='train':
                            summary = tf.Summary()
                            summary.value.add(tag='batch_train_loss', simple_value=np.mean(last_100_losses))

                            ncomputer.clear_current_mem(session)
                            for ii in range(5):
                                input_data, target_output, seq_len, decoder_length,masks, brout, brin = \
                                    prepare_sample_batch(dialogs_list_train, input_size, output_size, batch_size)

                                out = session.run([prob],  feed_dict={
                                    ncomputer.input_encoder: input_data,
                                    ncomputer.input_decoder: target_output,
                                    ncomputer.target_output: target_output,
                                    ncomputer.sequence_length: seq_len,
                                    ncomputer.decode_length: decoder_length,
                                    ncomputer.mask: masks,
                                    ncomputer.teacher_force: ncomputer.get_bool_rand_incremental(decoder_length, prob_true_max=0),
                                    ncomputer.drop_out_keep: args.drop_out_keep
                                                })

                                out = np.reshape(np.asarray(out),[-1, decoder_length, output_size])
                                out = np.argmax(out, axis=-1)
                                bout_list = []
                                for b in range(out.shape[0]):
                                    out_list = []
                                    for io in range(out.shape[1]):
                                        if out[b][io]==2:
                                            break
                                        out_list.append(out[b][io])
                                    bout_list.append(out_list)

                                trscores.append(bleu_score(np.asarray(brin),np.asarray(brout),np.asarray(bout_list), tok2str))
                            print('done quick test train...')

                        losses = []
                        all_out=[]
                        all_label=[]
                        all_res_in=[]
                        all_res_out=[]
                        all_res_pred=[]
                        all_res_score=[]
                        ntb = len(dialogs_list_valid) // batch_size + 1
                        for ii in range(ntb):
                            # llprint("\r{}/{}".format(ii, ntb))
                            if ii * batch_size == len(dialogs_list_valid):
                                break
                            bs = [ii * batch_size, min((ii + 1) * batch_size, len(dialogs_list_valid))]
                            rs = bs[1] - bs[0]
                            if bs[1] >= len(dialogs_list_valid):
                                bs = [len(dialogs_list_valid) - batch_size, len(dialogs_list_valid)]

                            input_data, target_output, seq_len, decoder_length, masks, rout_list, rin_list = \
                                prepare_sample_batch(dialogs_list_valid, input_size, output_size, bs)
                            out, loss_v = session.run([prob, loss], feed_dict={ncomputer.input_encoder: input_data,
                                                                               ncomputer.input_decoder: target_output,
                                                                               ncomputer.target_output: target_output,
                                                                               ncomputer.sequence_length: seq_len,
                                                                               ncomputer.decode_length: decoder_length,
                                                                               ncomputer.mask: masks,
                                                                               ncomputer.teacher_force: ncomputer.get_bool_rand_incremental(decoder_length, prob_true_max=0),
                                                                               ncomputer.drop_out_keep: 1
                                                                               })

                            losses.append(loss_v)
                            pout = np.reshape(np.asarray(out), [-1, decoder_length, output_size])
                            out = np.argmax(pout, axis=-1)
                            bout_list = []

                            for b in range(rs):
                                if args.beam_size == 0:
                                    out_list = []
                                    for io in range(out.shape[1]):
                                        if out[b][io]==2:
                                            break
                                        out_list.append(out[b][io])

                                else:
                                    out_list = beam_search.leap_beam_search(pout[b],
                                                                            beam_size=args.beam_size,
                                                                            is_set=True,
                                                                            is_fix_length=False,
                                                                            stop_char=2)
                                bout_list.append(out_list)
                            tescores.append(bleu_score(np.asarray(rin_list)[:rs], np.asarray(rout_list)[:rs], np.asarray(bout_list)[:rs], tok2str))



                        tloss=np.mean(losses)
                        tpre=np.mean(tescores)
                        print('tr score {} vs te store {}'.format(np.mean(trscores),np.mean(tescores)))
                        print('test loss {}'.format(tloss))
                        if args.mode=='train':
                            summary.value.add(tag='train_acc', simple_value=np.mean(trscores))
                            summary.value.add(tag='test_acc', simple_value=np.mean(tescores))
                            summary.value.add(tag='test_loss', simple_value=tloss)
                            train_writer.add_summary(summary, i)
                            train_writer.flush()

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print ("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                        print ("\tApprox. time to completion: %.2f hours" % (estimated_time))

                        start_time_100 = time.time()
                        last_100_losses = []



                    if args.mode=='train' and tpre>min_tloss:
                        min_tloss=tpre
                        llprint("\nSaving Checkpoint ... "),
                        ncomputer.save(session, ckpts_dir, ncomputer.print_config())
                        llprint("Done!\n")

                except KeyboardInterrupt:
                    sys.exit(0)



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train")
    parser.add_argument('--use_mem', default=True, type=str2bool)
    parser.add_argument('--use_teacher', default=False, type=str2bool)
    parser.add_argument('--task', default="cornell20_validpointer_20000_10_clean")
    parser.add_argument('--data_dir', default="./data/cornell20_20000_10/trim_20qa_single.pkl")
    parser.add_argument('--from_checkpoint', default="")
    parser.add_argument('--hidden_dim', default=768, type=int)
    parser.add_argument('--sampled_loss_dim', default=0, type=int)
    parser.add_argument('--emb_dim', default=96, type=int)
    parser.add_argument('--attend', default=0, type=int)
    parser.add_argument('--mem_size', default=16, type=int)
    parser.add_argument('--word_size', default=64, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--read_heads', default=1, type=int)
    parser.add_argument('--beam_size', default=0, type=int)
    parser.add_argument('--nlayer', default=3, type=int)
    parser.add_argument('--drop_out_keep', default=-1, type=float)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--lr_decay_step', default=10000, type=float)
    parser.add_argument('--lr_decay_rate', default=0.9, type=float)
    parser.add_argument('--iterations', default=1000000, type=int)
    parser.add_argument('--valid_time', default=100, type=int)
    parser.add_argument('--gpu_ratio', default=0.4, type=float)
    parser.add_argument('--cpu_num', default=10, type=int)
    parser.add_argument('--gpu_device', default="1,2,3", type=str)
    parser.add_argument('--use_pretrain_emb', default="word2vec", type=str)
    parser.add_argument('--persist_mode', default=False, type=str2bool)
    parser.add_argument('--test_file', default="./data/cornell20/test_single.txt", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    #
    # args.mode='test_file'
    # args.from_checkpoint = 'default'
    # args.use_mem=False
    # args.beam_size = 3
    # args.attend=64
    # args.task = 'cornell20_x2'

    print(args)



    single_qa_task(args)

