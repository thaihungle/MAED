import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
import pickle
from datetime import datetime




def load_mimic_emb():
    start = datetime.now()
    emb_matrix=pickle.load(open('./data/mimic/emb_mat.p','rb'))
    print(emb_matrix.shape)
    print("Finish loading emb, took {} s".format(datetime.now() - start))

def load_mimic_txt():
    start = datetime.now()
    f = open('./data/mimic/txt_top10.p', 'rb')
    loaded_data = []
    for i in range(7):  # [reverse_dictionary, train_sequence, test_sequence, train_label, test_label]:
        loaded_data.append(pickle.load(f))
    f.close()
    print("Finish loading data, took {} s".format(datetime.now()-start))
    tok2str, trainx, validx, testx, trainy, validy, testy=loaded_data
    return tok2str, trainx, validx, testx, trainy, validy, testy



if __name__ == '__main__':
    load_mimic_emb()
    # load_mimic_txt()