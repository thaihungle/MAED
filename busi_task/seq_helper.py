import numpy as np


def batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used

    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """
    # print(inputs)
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths

def batch_mask(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used

    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """
    # print(inputs)
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    max_sequence_length+=1
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD
    mask = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.bool)  # == PAD
    for i, seq in enumerate(inputs):
        seq=seq+[0]
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element
            mask[i,j]=True

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)
    mask = mask.swapaxes(0, 1)
    return inputs_time_major, sequence_lengths, mask

def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
        raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]

def even_odd_sample(vocab_lower, vocab_upper, length_from, length_to):
    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    seed = np.random.choice(list(range(int(vocab_lower),int(vocab_upper))),
                      int(random_length()), replace=False)
    odd = (seed*2+1).tolist()
    even = (seed*4).tolist()
    even2 = even
    for i in range(len(even) // 2 + 1, len(even)):
        even2[i] = even2[i - 1] + 2
    # return odd, even
    return odd, even

def even_od_sequence(vocab_lower, vocab_upper, length_from, length_to, batch_size):
    while True:
        ret=[]
        for i in range(batch_size):
            ret.append(even_odd_sample(vocab_lower, vocab_upper, length_from, length_to))
        yield ret

def mimic_sequence(dig_list, proc_list, batch_size):
    while True:
        indexs=np.random.choice(len(dig_list), batch_size, replace=False)
        # print('sss {}'.format(indexs))
        ret=[]
        for i in range(batch_size):
            ret.append((dig_list[indexs[i]],proc_list[indexs[i]]))
        yield ret

def mimic_sequence_all(batch_size, dig_list, proc_list):
    ret_b = []
    for i in range(len(dig_list)//batch_size+1):
        if i>=len(dig_list):
            break
        ret = []
        for j in range(i*batch_size, min((i+1)*batch_size, len(dig_list))):
            ret.append((dig_list[j],proc_list[j]))

        ret_b.append(ret)
    return ret_b