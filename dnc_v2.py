import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from memory import Memory
import utility
import os

class ProjectionOp:
    """ Single layer perceptron
    Project input tensor on the output dimension
    """
    def __init__(self, shape, scope=None, dtype=None):
        """
        Args:
            shape: a tuple (input dim, output dim)
            scope (str): encapsulate variables
            dtype: the weights type
        """
        assert len(shape) == 2

        self.scope = scope

        # Projection on the keyboard
        with tf.variable_scope('weights_' + self.scope):
            self.W_t = tf.get_variable(
                'weights',
                shape,
                # initializer=tf.truncated_normal_initializer()  # TODO: Tune value (fct of input size: 1/sqrt(input_dim))
                dtype=dtype
            )
            self.b = tf.get_variable(
                'bias',
                shape[0],
                initializer=tf.constant_initializer(),
                dtype=dtype
            )
            self.W = tf.transpose(self.W_t)#[deim, nout]

    def getWeights(self):
        """ Convenience method for some tf arguments
        """
        return self.W, self.b

    def project(self, X):
        """ Project the output of the decoder into the vocabulary space
        Args:
            X (tf.Tensor): input value
        """
        with tf.name_scope(self.scope):
            return tf.matmul(X, self.W) + self.b

class DNC:

    def __init__(self, controller_class, input_encoder_size, input_decoder_size, output_size,
                 memory_words_num = 256, memory_word_size = 64, memory_read_heads = 4,
                 batch_size = 1,hidden_controller_dim=256, use_emb_encoder=True, use_emb_decoder=True,
                 use_mem=True, decoder_mode=False, emb_size=64,
                 write_protect=False, dual_controller=False, dual_emb=True,
                 use_teacher=False, attend_dim=0, persist_mode=False,
                 pointer_mode=0, use_encoder_output=False,
                 pass_encoder_state=True, sampled_loss_dim=0,
                 memory_read_heads_decode=None, enable_drop_out=False,
                 nlayer=1, name='DNCv2'):
        """
        constructs a complete DNC architecture as described in the DNC paper
        http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html
        Parameters:
        -----------
        controller_class: BaseController
            a concrete implementation of the BaseController class
        input_size: int
            the size of the input vector
        output_size: int
            the size of the output vector
        max_sequence_length: int
            the maximum length of an input sequence
        memory_words_num: int
            the number of words that can be stored in memory
        memory_word_size: int
            the size of an individual word in memory
        memory_read_heads: int
            the number of read heads in the memory
        batch_size: int
            the size of the data batch
        """
        saved_args = locals()
        print("saved_args is", saved_args)
        self.name=name
        self.input_encoder_size = input_encoder_size
        self.input_decoder_size = input_decoder_size
        self.output_size = output_size
        self.words_num = memory_words_num
        self.word_size = memory_word_size
        self.read_heads = memory_read_heads
        if memory_read_heads_decode is None:
            self.read_heads_decode = memory_read_heads
        else:
            self.read_heads_decode = memory_read_heads_decode
        self.batch_size = batch_size
        self.unpacked_input_encoder_data = None
        self.unpacked_input_decoder_data = None
        self.packed_output = None
        self.packed_memory_view_encoder = None
        self.packed_memory_view_decoder = None
        self.decoder_mode = decoder_mode
        self.emb_size = emb_size
        self.emb_size2 = emb_size
        self.dual_emb = dual_emb
        self.use_mem = use_mem
        self.use_emb_encoder = use_emb_encoder
        self.use_emb_decoder = use_emb_decoder
        self.hidden_controller_dim = hidden_controller_dim
        self.attend_dim = attend_dim
        self.use_teacher = use_teacher
        self.teacher_force = tf.placeholder(tf.bool,[None], name='teacher')
        self.persist_mode = persist_mode
        self.pointer_mode = pointer_mode# 0 no pointer, 1 full pointer, 2 assist pointer
        self.use_encoder_output=use_encoder_output
        self.pass_encoder_state=pass_encoder_state
        self.clear_mem = tf.placeholder(tf.bool,None, name='clear_mem')
        self.drop_out_keep = tf.placeholder(tf.float32, name='drop_out_keep')
        self.nlayer=nlayer
        drop_out_v = 1
        if enable_drop_out:
            drop_out_v = self.drop_out_keep

        self.sampled_loss_dim = sampled_loss_dim
        self.outputProjection = None

        self.controller_out = self.output_size
        if self.sampled_loss_dim > 0:
            self.controller_out = self.sampled_loss_dim
            self.outputProjection = ProjectionOp(
                (self.output_size, self.sampled_loss_dim),
                scope='softmax_projection',
            )


        if self.use_emb_encoder is False:
            self.emb_size=input_encoder_size

        if self.use_emb_decoder is False:
            self.emb_size2=input_decoder_size #pointer mode not use

        if self.attend_dim>0:
            self.W_a = tf.get_variable('W_a', [hidden_controller_dim, self.attend_dim],
                                      initializer=tf.random_normal_initializer(stddev=0.1))

            value_size = self.hidden_controller_dim
            if self.use_mem:
                value_size = self.word_size
            self.U_a = tf.get_variable('U_a', [value_size, self.attend_dim],
                                  initializer=tf.random_normal_initializer(stddev=0.1))
            if self.use_mem:
                self.V_a = tf.get_variable('V_a', [self.read_heads_decode*self.word_size, self.attend_dim],
                                       initializer=tf.random_normal_initializer(stddev=0.1))
            self.v_a = tf.get_variable('v_a', [self.attend_dim],
                                  initializer=tf.random_normal_initializer(stddev=0.1))

        # DNC (or NTM) should be structurized into 2 main modules:
        # all the graph is setup inside these twos:
        self.W_emb_encoder = tf.get_variable('embe_w', [self.input_encoder_size, self.emb_size],
                                             initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
        self.W_emb_decoder = tf.get_variable('embd_w', [self.output_size, self.emb_size],
                                             initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

        self.memory = Memory(self.words_num, self.word_size, self.read_heads, self.batch_size)
        self.controller = controller_class(self.emb_size, self.controller_out, self.read_heads,
                                           self.word_size, self.batch_size, use_mem,
                                           hidden_dim=hidden_controller_dim, drop_out_keep=drop_out_v, nlayer=nlayer)
        self.dual_controller = dual_controller
        if self.dual_controller:
            with tf.variable_scope('controller2_scope'):
                if attend_dim==0 or pointer_mode==1 or use_mem:
                    self.controller2 = controller_class(self.emb_size2, self.controller_out, self.read_heads_decode,
                                                       self.word_size, self.batch_size, use_mem,
                                                        hidden_dim=hidden_controller_dim, drop_out_keep=drop_out_v, nlayer=nlayer)
                else:
                    self.controller2 = controller_class(self.emb_size2+hidden_controller_dim, self.controller_out, self.read_heads_decode,
                                                        self.word_size, self.batch_size, use_mem,
                                                        hidden_dim=hidden_controller_dim, drop_out_keep=drop_out_v, nlayer=nlayer)
        self.write_protect = write_protect
        if pointer_mode==2:
            wsize = self.hidden_controller_dim + self.emb_size2 + self.hidden_controller_dim
            if self.use_mem:
                wsize=self.hidden_controller_dim + self.emb_size2 + self.read_heads_decode*self.word_size
            self.W_pointer_gate = tf.get_variable('W_pg', [wsize, 1],
                                                  initializer=tf.random_uniform_initializer(minval=-1, maxval=1))


        # input data placeholders
        if pointer_mode==1:
            self.pointer_output_size = tf.placeholder(tf.int32, name='pointer_output_size')
            self.target_output = tf.placeholder(tf.float32, [batch_size, None, None], name='targets')
        elif sampled_loss_dim>0:
            self.target_output = tf.placeholder(tf.float32, [batch_size, None, 1], name='targets')
        else:
            self.target_output = tf.placeholder(tf.float32, [batch_size, None, output_size], name='targets')

        self.input_encoder = tf.placeholder(tf.float32, [batch_size, None, input_encoder_size], name='input_encoder')
        if sampled_loss_dim > 0:
            self.input_decoder = tf.placeholder(tf.float32, [batch_size, None, 1], name='input_decoder')
        else:
            self.input_decoder = tf.placeholder(tf.float32, [batch_size, None, input_decoder_size], name='input_decoder')

        self.mask = tf.placeholder(tf.bool, [batch_size, None], name='mask')
        self.sequence_length = tf.placeholder(tf.int32, name='sequence_length')# variant length?
        self.decode_length = tf.placeholder(tf.int32, name='decode_length')  # variant length?

        if persist_mode:
            self.cur_mem_content = tf.get_variable('cur_mc', [self.batch_size, self.words_num, self.word_size],trainable=False)
            self.assign_op_cur_mem = self.cur_mem_content.assign(np.ones([self.batch_size, self.words_num, self.word_size])*1e-6)
            self.cur_u = tf.get_variable('cur_u', [self.batch_size, self.words_num],trainable=False) # initial usage vector u
            self.assign_op_cur_u = self.cur_u.assign(np.zeros([self.batch_size, self.words_num]))
            self.cur_p = tf.get_variable('cur_p',[self.batch_size, self.words_num], trainable=False) # initial precedence vector p
            self.assign_op_cur_p = self.cur_p.assign(np.zeros([self.batch_size, self.words_num]))
            self.cur_L = tf.get_variable('cur_L',[self.batch_size, self.words_num, self.words_num], trainable=False) # initial link matrix L
            self.assign_op_cur_L = self.cur_L.assign(np.ones([self.batch_size, self.words_num, self.words_num])*1e-6)
            self.cur_ww = tf.get_variable('cur_ww',[self.batch_size, self.words_num], trainable=False) # initial write weighting
            self.assign_op_cur_ww = self.cur_ww.assign(np.ones([self.batch_size, self.words_num])*1e-6)
            self.cur_rw = tf.get_variable('cur_rw',[self.batch_size, self.words_num, self.read_heads], trainable=False) # initial read weightings
            self.assign_op_cur_rw = self.cur_rw.assign(np.ones([self.batch_size, self.words_num, self.read_heads])*1e-6)
            self.cur_rv = tf.get_variable('cur_rv',[self.batch_size, self.word_size, self.read_heads], trainable=False)  # initial read vectors
            self.assign_op_cur_rv = self.cur_rv.assign(np.ones([self.batch_size, self.word_size, self.read_heads])*1e-6)
            self.cur_encoder_rnn_state = self.controller.zero_state()
            self.cur_c = tf.get_variable('cur_c', [self.batch_size, hidden_controller_dim],
                                         trainable=False)
            self.assign_op_cur_c = self.cur_c.assign(np.ones([self.batch_size, hidden_controller_dim]) * 1e-6)
            self.cur_h = tf.get_variable('cur_h', [self.batch_size, hidden_controller_dim],
                                         trainable=False)
            self.assign_op_cur_h = self.cur_h.assign(np.ones([self.batch_size, hidden_controller_dim]) * 1e-6)
        self.build_graph()


    def pointer_weight(self, fout, controller_state, read_vectors, step2, alphas):
        if self.pointer_mode==2 and self.attend_dim>0:
            if self.nlayer > 1:
                try:
                    ns = controller_state[-1][-1]
                    print('multilayer state include c and h')
                except:
                    ns = controller_state[-1]
                    print('multilayer state include only h')
            else:
                ns = controller_state[-1]
            if self.use_mem:
                contex=tf.concat([ns,step2,tf.reshape(read_vectors,[self.batch_size,-1])],axis=-1)
            else:
                contex=tf.concat([ns,step2],axis=-1)
            gp = tf.sigmoid(tf.matmul(contex,self.W_pointer_gate))
            fout = fout*(gp*tf.reshape(tf.matmul(tf.transpose(tf.expand_dims(alphas,2),[0,2,1]),
                                         self.input_encoder),
                               [self.batch_size,self.output_size]) +\
                         (1.0-tf.reduce_max(self.input_encoder,axis=1)*(1-gp)))
            return fout
        else:
            return fout

    # The nature of DNC is to process data by step and remmeber data at each time step when necessary
    # If input has sequence format --> suitable with RNN core controller --> each time step in RNN equals 1 time step in DNC
    # or just feed input to MLP --> each feed is 1 time step
    def _step_op_encoder(self,  step, memory_state, controller_state=None):
        """
        performs a step operation on the input step data
        Parameters:
        ----------
        step: Tensor (batch_size, input_size)
        memory_state: Tuple
            a tuple of current memory parameters
        controller_state: Tuple
            the state of the controller if it's recurrent
        Returns: Tuple
            output: Tensor (batch_size, output_size)
            memory_view: dict
        """

        last_read_vectors = memory_state[6] # read values from memory
        pre_output, interface, nn_state = None, None, None

        # compute oututs from controller
        if self.controller.has_recurrent_nn:
            # controller state is the rnn cell state pass through each time step
            if not self.use_emb_encoder:
                step2 = tf.reshape(step, [-1, self.input_encoder_size])
                pre_output, interface, nn_state=  self.controller.process_input(step2, last_read_vectors, controller_state)
            else:
                pre_output, interface, nn_state = self.controller.process_input(step, last_read_vectors, controller_state)
        else:
            pre_output, interface = self.controller.process_input(step, last_read_vectors)

        # memory_matrix isthe copy of memory for reading process later
        # do the write first

        usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = self.memory.write(
            memory_state[0], memory_state[1], memory_state[5],
            memory_state[4], memory_state[2], memory_state[3],
            interface['write_key'],
            interface['write_strength'],
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],
            interface['write_vector'],
            interface['erase_vector']
        )

        # then do the read, read after write because the write weight is needed to produce temporal linklage to guide the reading
        read_weightings, read_vectors = self.memory.read(
            memory_matrix,
            memory_state[5],
            interface['read_keys'],
            interface['read_strengths'],
            link_matrix,
            interface['read_modes'],
        )
        fout=None
        if self.use_encoder_output:
            fout = self.controller.final_output(pre_output, read_vectors)


        return [
            # report new memory state to be updated outside the condition branch
            memory_matrix, #0

            # neccesary for next step to compute memory stuffs
            usage_vector, #1
            precedence_vector, #2
            link_matrix, #3
            write_weighting, #4
            read_weightings, #5
            read_vectors, #6

            # the final output of dnc
            fout, #7

            # the values public info to outside
            interface['read_modes'], #8
            interface['allocation_gate'], #9
            interface['write_vector'], #10

            # report new state of RNN if exists, neccesary for next step to compute inner controller stuff
            nn_state if nn_state is not None else tf.zeros(1), #11
        ]

    def _step_op_decoder(self, step, memory_state,
                         controller_state=None, controller_hiddens=None):
        """
        performs a step operation on the input step data
        Parameters:
        ----------
        step: Tensor (batch_size, input_size)
        memory_state: Tuple
            a tuple of current memory parameters
        controller_state: Tuple
            the state of the controller if it's recurrent
        Returns: Tuple
            output: Tensor (batch_size, output_size)
            memory_view: dict
        """

        last_read_vectors = memory_state[6]  # read values from memory
        pre_output, interface, nn_state = None, None, None

        if self.dual_controller:
            controller=self.controller2
        else:
            controller=self.controller
        alphas = None
        # compute outputs from controller
        if controller.has_recurrent_nn:
            if not self.use_emb_decoder:
                if self.pointer_mode==1:
                    step2 = tf.reshape(step, [-1, self.input_encoder_size])
                elif self.sampled_loss_dim > 0:
                    step2 = tf.one_hot(tf.argmax(step, axis=-1), depth=self.output_size)
                else:
                    step2 = tf.reshape(step, [-1, self.output_size])
            else:
                step2 = step
            # attention

            if self.attend_dim>0:
                values = utility.pack_into_tensor(controller_hiddens,axis=1)
                value_size = self.hidden_controller_dim
                if self.use_mem:
                    value_size = self.word_size
                # values = controller_hiddens.gather(tf.range(self.sequence_length))
                encoder_outputs = \
                    tf.reshape(values, [self.batch_size, -1, value_size])  # bs x Lin x h
                v = tf.reshape(tf.matmul(tf.reshape(encoder_outputs, [-1, value_size]), self.U_a),
                               [self.batch_size, -1, self.attend_dim])


                if self.use_mem:
                    v+= tf.reshape(
                        tf.matmul(tf.reshape(last_read_vectors, [-1, self.read_heads_decode*self.word_size]), self.V_a),
                        [self.batch_size, 1, self.attend_dim])

                if self.nlayer>1:
                    try:
                        ns=controller_state[-1][-1]
                        print('multilayer state include c and h')
                    except:
                        ns = controller_state[-1]
                        print('multilayer state include only h')
                else:
                    ns = controller_state[-1]
                    print('single layer')
                print(ns)
                v += tf.reshape(
                    tf.matmul(tf.reshape(ns, [-1, self.hidden_controller_dim]), self.W_a),
                    [self.batch_size, 1, self.attend_dim])  # bs.Lin x h_att
                print('state include only h')

                v = tf.reshape(tf.tanh(v), [-1, self.attend_dim])
                eijs = tf.matmul(v, tf.expand_dims(self.v_a, 1))  # bs.Lin x 1
                eijs = tf.reshape(eijs, [self.batch_size, -1])  # bs x Lin
                alphas = tf.nn.softmax(eijs)
                # exps = tf.exp(eijs)
                # alphas = exps /(tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])+1e-2)  # bs x Lin

                if self.pointer_mode!=1 and not self.use_mem:
                    att = tf.reduce_sum(encoder_outputs * tf.expand_dims(alphas, 2), 1)  # bs x h x 1
                    att = tf.reshape(att, [self.batch_size, self.hidden_controller_dim])  # bs x h
                    step2 = tf.concat([step2, att], axis=-1)  # bs x (decoder_input_size + h)

            pre_output, interface, nn_state = controller.process_input(step2, last_read_vectors, controller_state)

        else:
            pre_output, interface = controller.process_input(step, last_read_vectors)

        # memory_matrix isthe copy of memory for reading process later
        # do the write first
        if self.write_protect:
            usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector \
                =memory_state[1], memory_state[4], memory_state[0], memory_state[3], memory_state[2]

        else:
            usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = self.memory.write(
                memory_state[0], memory_state[1], memory_state[5],
                memory_state[4], memory_state[2], memory_state[3],
                interface['write_key'],
                interface['write_strength'],
                interface['free_gates'],
                interface['allocation_gate'],
                interface['write_gate'],
                interface['write_vector'],
                interface['erase_vector']
            )

        # then do the read, read after write because the write weight is needed to produce temporal linklage to guide the reading
        read_weightings, read_vectors = self.memory.read(
            memory_matrix,
            memory_state[5],
            interface['read_keys'],
            interface['read_strengths'],
            link_matrix,
            interface['read_modes'],
        )
        if  self.pointer_mode!=1:
            fout = controller.final_output(pre_output, read_vectors) # bs x output_size
        else:
            # if self.use_mem and self.attend_dim==0:
            #     # pointer with mem
            #     values = utility.pack_into_tensor(controller_hiddens, axis=1)# bs x Lin x mem_size
            #     # values = controller_hiddens.gather(tf.range(0,self.sequence_length)) #write_weights of encoder
            #     encoder_outputs = \
            #         tf.reshape(values, [self.batch_size, -1, self.words_num])  # bs x Lin x mem_size
            #     fout = controller.final_output_pointer(read_weightings, encoder_outputs)
            # elif self.attend_dim > 0:
            #     # pointer without mem
            fout = controller.final_output_pointer(None, alphas)
        fout = self.pointer_weight(fout, controller_state, read_vectors, step2, alphas)


        return [
            # report new memory state to be updated outside the condition branch
            memory_matrix,  # 0

            # neccesary for next step to compute memory stuffs
            usage_vector,  # 1
            precedence_vector,  # 2
            link_matrix,  # 3
            write_weighting,  # 4
            read_weightings,  # 5
            read_vectors,  # 6

            # the final output of dnc
            fout,  # 7

            # the values public info to outside
            interface['read_modes'],  # 8
            interface['allocation_gate'],  # 9
            interface['write_gate'],  # 10

            # report new state of RNN if exists, neccesary for next step to compute inner controller stuff
            nn_state if nn_state is not None else tf.zeros(1),  # 11
        ]

    '''
    THIS WRAPPER FOR ONE STEP OF COMPUTATION --> INTERFACE FOR SCAN/WHILE LOOP
    '''
    def _loop_body_encoder(self, time, memory_state, outputs, free_gates, allocation_gates, write_gates,
                   read_weightings, write_weightings, usage_vectors, controller_state,
                   outputs_cache, controller_hiddens):
        """
        the body of the DNC sequence processing loop
        Parameters:
        ----------
        time: Tensor
        outputs: TensorArray
        memory_state: Tuple
        free_gates: TensorArray
        allocation_gates: TensorArray
        write_gates: TensorArray
        read_weightings: TensorArray,
        write_weightings: TensorArray,
        usage_vectors: TensorArray,
        controller_state: Tuple
        Returns: Tuple containing all updated arguments
        """

        # dynamic tensor array input

        if self.use_emb_encoder:
            step_input = tf.matmul(self.unpacked_input_encoder_data.read(time), self.W_emb_encoder)
        else:
            step_input = self.unpacked_input_encoder_data.read(time)


        # compute one step of controller
        output_list = self._step_op_encoder(step_input, memory_state, controller_state)
        # update memory parameters

        # new_controller_state = tf.zeros(1)
        new_memory_state = tuple(output_list[0:7])
        new_controller_state = output_list[11] #state  hidden values

        if self.nlayer>1:
            try:
                controller_hiddens = controller_hiddens.write(time, new_controller_state[-1][-1])
                print('state include c and h')
            except:
                controller_hiddens = controller_hiddens.write(time, new_controller_state[-1])
                print('state include only h')
        else:
            controller_hiddens = controller_hiddens.write(time, new_controller_state[-1])
            print('single layer')

        if self.use_encoder_output:
            outputs = outputs.write(time, output_list[7])# new output is updated
            outputs_cache = outputs_cache.write(time, output_list[7])# new output is updated
        # collecting memory view for the current step
        free_gates = free_gates.write(time, output_list[8])
        allocation_gates = allocation_gates.write(time, output_list[9])
        write_gates = write_gates.write(time, output_list[10])
        read_weightings = read_weightings.write(time, output_list[5])
        write_weightings = write_weightings.write(time, output_list[10])
        usage_vectors = usage_vectors.write(time, output_list[1])

        # all variables have been updated should be return for next step reference
        return (
            time + 1, #0
            new_memory_state, #1
            outputs, #2
            free_gates,allocation_gates, write_gates, #3 4 5
            read_weightings, write_weightings, usage_vectors, #6 7 8
            new_controller_state, #9
            outputs_cache,  #10
            controller_hiddens, #11
        )

    def _loop_body_decoder(self, time, memory_state, outputs, free_gates, allocation_gates, write_gates,
                           read_weightings, write_weightings, usage_vectors, controller_state,
                           outputs_cache, controller_hiddens,
                           encoder_write_weightings, encoder_controller_hiddens):
        """
        the body of the DNC sequence processing loop
        Parameters:
        ----------
        time: Tensor
        outputs: TensorArray
        memory_state: Tuple
        free_gates: TensorArray
        allocation_gates: TensorArray
        write_gates: TensorArray
        read_weightings: TensorArray,
        write_weightings: TensorArray,
        usage_vectors: TensorArray,
        controller_state: Tuple
        Returns: Tuple containing all updated arguments
        """

        # dynamic tensor array input
        if self.decoder_mode:
            def fn1():
                if self.pointer_mode==1:
                    return tf.zeros([self.batch_size, self.input_encoder_size])
                else:
                    return tf.zeros([self.batch_size, self.output_size])
            def fn2():
                def fn2_1():
                    if self.pointer_mode==1:
                        inds=tf.argmax(self.target_output[:, time - 1, :],axis=-1)
                        v = tf.one_hot(inds, depth=self.pointer_output_size)
                        # return  tf.gather(self.input_encoder, inds, axis=1)
                        return tf.matmul(tf.expand_dims(v,axis=1), self.input_encoder)
                    elif self.sampled_loss_dim > 0:
                        out = self.target_output[:, time - 1, :]
                        out = tf.one_hot(tf.argmax(out, axis=-1), depth=self.output_size)
                        return out
                    else:
                        return self.target_output[:, time - 1, :]

                def fn2_2():
                    if self.sampled_loss_dim>0:
                        out = outputs_cache.read(time - 1)
                        out = self.outputProjection.project(
                            tf.reshape(out, [-1, self.sampled_loss_dim]))
                        return out
                    else:
                        inds = tf.argmax(outputs_cache.read(time - 1), axis=-1)
                        if self.pointer_mode==1:
                            v = tf.one_hot(inds, depth=self.pointer_output_size)
                            # return tf.gather(self.input_encoder, inds, axis=1)
                            return tf.matmul(tf.expand_dims(v,axis=1), self.input_encoder)
                        else:
                            return tf.one_hot(inds, depth=self.output_size)

                if self.use_teacher:
                    return tf.cond(self.teacher_force[time - 1], fn2_1, fn2_2)
                else:
                    return  fn2_2()

            feed_value = tf.cond(time>0,fn2,fn1)


            if not self.use_emb_decoder:
                r = tf.reshape(feed_value, [self.batch_size, self.input_decoder_size])
                step_input = r
            elif self.dual_emb:
                step_input = tf.matmul(feed_value, self.W_emb_decoder)
            else:
                step_input = tf.matmul(feed_value, self.W_emb_encoder)

        else:
            if self.use_emb_decoder:
                if self.dual_emb:
                    step_input = tf.matmul(self.unpacked_input_decoder_data.read(time), self.W_emb_decoder)
                else:
                    step_input = tf.matmul(self.unpacked_input_decoder_data.read(time), self.W_emb_encoder)
            else:
                step_input = self.unpacked_input_decoder_data.read(time)
                print(step_input.shape)
                print('ssss')

        # compute one step of controller
        if self.pointer_mode==1 and self.use_mem and self.attend_dim==0:
            print('pointer full memory')
            output_list = self._step_op_decoder(step_input, memory_state, controller_state, encoder_write_weightings)
        elif not self.use_mem and self.attend_dim > 0:
            print('normal attention or mix pointer mode without memory')
            output_list = self._step_op_decoder(step_input, memory_state, controller_state, encoder_controller_hiddens)
        elif self.use_mem and self.attend_dim > 0:
            print('attention and mix pointer mode with memory')
            output_list = self._step_op_decoder(step_input, memory_state, controller_state, encoder_write_weightings)
        else:
            output_list = self._step_op_decoder(step_input, memory_state, controller_state)
            # update memory parameters
        # update memory parameters

        # new_controller_state = tf.zeros(1)
        new_memory_state = tuple(output_list[0:7])
        new_controller_state = output_list[11] # state hidden  values

        if self.nlayer>1:
            try:
                controller_hiddens = controller_hiddens.write(time, new_controller_state[-1][-1])
                print('state include c and h')
            except:
                controller_hiddens = controller_hiddens.write(time, new_controller_state[-1])
                print('state include only h')
        else:
            controller_hiddens = controller_hiddens.write(time, new_controller_state[-1])
            print('single layer')
        outputs = outputs.write(time, output_list[7])  # new output is updated
        outputs_cache = outputs_cache.write(time, output_list[7])  # new output is updated
        # collecting memory view for the current step
        free_gates = free_gates.write(time, output_list[8])
        allocation_gates = allocation_gates.write(time, output_list[9])
        write_gates = write_gates.write(time, output_list[10])
        read_weightings = read_weightings.write(time, output_list[5])
        write_weightings = write_weightings.write(time, output_list[4])
        usage_vectors = usage_vectors.write(time, output_list[1])

        # all variables have been updated should be return for next step reference
        return (
            time + 1,  # 0
            new_memory_state,  # 1
            outputs,  # 2
            free_gates, allocation_gates, write_gates,  # 3 4 5
            read_weightings, write_weightings, usage_vectors,  # 6 7 8
            new_controller_state,  # 9
            outputs_cache,  # 10
            controller_hiddens,  # 11
            encoder_write_weightings, #12
            encoder_controller_hiddens, #13
        )

    def build_graph(self):
        """
        builds the computational graph that performs a step-by-step evaluation
        of the input data batches
        """

        # make dynamic time step length tensor
        self.unpacked_input_encoder_data = utility.unpack_into_tensorarray(self.input_encoder, 1, self.sequence_length)

        # want to store all time step values of these variables
        eoutputs = tf.TensorArray(tf.float32, self.sequence_length)
        eoutputs_cache = tf.TensorArray(tf.float32, self.sequence_length)
        efree_gates = tf.TensorArray(tf.float32, self.sequence_length)
        eallocation_gates = tf.TensorArray(tf.float32, self.sequence_length)
        ewrite_gates = tf.TensorArray(tf.float32, self.sequence_length)
        eread_weightings = tf.TensorArray(tf.float32, self.sequence_length, clear_after_read=False)
        ewrite_weightings = tf.TensorArray(tf.float32, self.sequence_length, clear_after_read=False)
        eusage_vectors = tf.TensorArray(tf.float32, self.sequence_length, clear_after_read=False)
        econtroller_hiddens = tf.TensorArray(tf.float32, self.sequence_length, clear_after_read=False)

        # make dynamic time step length tensor
        self.unpacked_input_decoder_data = utility.unpack_into_tensorarray(self.input_decoder, 1, self.decode_length)

        # want to store all time step values of these variables
        doutputs = tf.TensorArray(tf.float32, self.decode_length)
        doutputs_cache = tf.TensorArray(tf.float32, self.decode_length)
        dfree_gates = tf.TensorArray(tf.float32, self.decode_length)
        dallocation_gates = tf.TensorArray(tf.float32, self.decode_length)
        dwrite_gates = tf.TensorArray(tf.float32, self.decode_length)
        dread_weightings = tf.TensorArray(tf.float32, self.decode_length)
        dwrite_weightings = tf.TensorArray(tf.float32, self.decode_length, clear_after_read=False)
        dusage_vectors = tf.TensorArray(tf.float32, self.decode_length)
        dcontroller_hiddens = tf.TensorArray(tf.float32, self.decode_length, clear_after_read=False)

        # inital state for RNN controller
        controller_state = self.controller.zero_state()
        print(controller_state)
        memory_state = self.memory.init_memory()
        if self.persist_mode:
            def p1():
                return memory_state, controller_state
            def p2():
                return (self.cur_mem_content, self.cur_u, self.cur_p,
                        self.cur_L, self.cur_ww, self.cur_rw, self.cur_rv), \
                       tuple(self.cur_encoder_rnn_state)
            memory_state, controller_state=tf.cond(self.clear_mem, p1, p2)


        # final_results = None
        with tf.variable_scope("sequence_encoder_loop"):
            time = tf.constant(0, dtype=tf.int32)

            # use while instead of scan --> suitable with dynamic time step
            encoder_results = tf.while_loop(
                cond=lambda time, *_: time < self.sequence_length,
                body=self._loop_body_encoder,
                loop_vars=(
                    time, memory_state, eoutputs,
                    efree_gates, eallocation_gates, ewrite_gates,
                    eread_weightings, ewrite_weightings,
                    eusage_vectors, controller_state,
                    eoutputs_cache, econtroller_hiddens
                ), # do not need to provide intial values, the initial value lies in the variables themselves
                parallel_iterations=1,
                swap_memory=True
            )

        memory_state2 = self.memory.init_memory(self.read_heads_decode)
        if self.read_heads_decode!=self.read_heads:
            encoder_results_state=(encoder_results[1][0],encoder_results[1][1],encoder_results[1][2],
                                encoder_results[1][3],encoder_results[1][4], memory_state2[5],memory_state2[6])
        else:
            encoder_results_state=encoder_results[1]
        with tf.variable_scope("sequence_decoder_loop"):
            time = tf.constant(0, dtype=tf.int32)
            nstate = controller_state
            if self.pass_encoder_state:
                nstate = encoder_results[9]
            # use while instead of scan --> suitable with dynamic time step
            final_results = tf.while_loop(
                cond=lambda time, *_: time < self.decode_length,
                body=self._loop_body_decoder,
                loop_vars=(
                    time, encoder_results_state, doutputs,
                    dfree_gates, dallocation_gates, dwrite_gates,
                    dread_weightings, dwrite_weightings,
                    dusage_vectors, nstate,
                    doutputs_cache, dcontroller_hiddens,
                    encoder_results[7], encoder_results[11]
                ),  # do not need to provide intial values, the initial value lies in the variables themselves
                parallel_iterations=1,
                swap_memory=True
            )

        if self.persist_mode:

            self.cur_mem_content, self.cur_u, self.cur_p, \
            self.cur_L, self.cur_ww, self.cur_rw, self.cur_rv = encoder_results[1]
            try:
                self.cur_c = encoder_results[9][0][0]
                self.cur_h = encoder_results[9][0][1]
                self.cur_encoder_rnn_state = list(self.controller.zero_state())
                self.cur_encoder_rnn_state[0][0]=self.cur_c
                self.cur_encoder_rnn_state[0][1] = self.cur_h
            except:
                self.cur_c = encoder_results[9][0]
                self.cur_h = encoder_results[9][0]
                self.cur_encoder_rnn_state = list(self.controller.zero_state())
                self.cur_encoder_rnn_state[0] = self.cur_c




        dependencies = []
        if self.controller.has_recurrent_nn:
            # tensor array of pair of hidden and state values of rnn
            dependencies.append(self.controller.update_state(final_results[9]))

        with tf.control_dependencies(dependencies):
            # convert output tensor array to normal tensor
            self.packed_output = utility.pack_into_tensor(final_results[2], axis=1)
            self.packed_memory_view_encoder = {
                'free_gates': utility.pack_into_tensor(encoder_results[3], axis=1),
                'allocation_gates': utility.pack_into_tensor(encoder_results[4], axis=1),
                'write_gates': utility.pack_into_tensor(encoder_results[5], axis=1),
                'read_weightings': utility.pack_into_tensor(encoder_results[6], axis=1),
                'write_weightings': utility.pack_into_tensor(encoder_results[7], axis=1),
                'usage_vectors': utility.pack_into_tensor(encoder_results[8], axis=1),
                'final_controller_ch': encoder_results[9],
            }
            self.packed_memory_view_decoder = {
                'free_gates': utility.pack_into_tensor(final_results[3], axis=1),
                'allocation_gates': utility.pack_into_tensor(final_results[4], axis=1),
                'write_gates': utility.pack_into_tensor(final_results[5], axis=1),
                'read_weightings': utility.pack_into_tensor(final_results[6], axis=1),
                'write_weightings': utility.pack_into_tensor(final_results[7], axis=1),
                'usage_vectors': utility.pack_into_tensor(final_results[8], axis=1),
                'final_controller_ch':final_results[9],
            }


    def get_outputs(self):
        """
        returns the graph nodes for the output and memory view
        Returns: Tuple
            outputs: Tensor (batch_size, time_steps, output_size)
            memory_view: dict
        """
        return self.packed_output, self.packed_memory_view_encoder, self.packed_memory_view_decoder

    def assign_pretrain_emb_encoder(self, sess, lookup_mat):
        assign_op_W_emb_encoder = self.W_emb_encoder.assign(lookup_mat)
        sess.run([assign_op_W_emb_encoder])

    def assign_pretrain_emb_decoder(self, sess, lookup_mat):
        assign_op_W_emb_decoder = self.W_emb_decoder.assign(lookup_mat)
        sess.run([assign_op_W_emb_decoder])


    def build_loss_function(self, optimizer=None, clip_s=10):
        print('build loss....')
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer()
        output, _, _ = self.get_outputs()

        prob = tf.nn.softmax(output, dim=-1)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.target_output,
            logits=output, dim=-1))


        gradients = optimizer.compute_gradients(loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_value(grad, -clip_s, clip_s), var)


        apply_gradients = optimizer.apply_gradients(gradients)
        return output, prob, loss, apply_gradients


    def build_loss_function_mask(self, optimizer=None, clip_s=10, learning_params=None):

        # train_arg.add_argument('--lr_start', type=float, default=0.001, help='')
        # train_arg.add_argument('--lr_decay_step', type=int, default=5000, help='')
        # train_arg.add_argument('--lr_decay_rate', type=float, default=0.96, help='')
        # train_arg.add_argument('--max_grad_norm', type=float, default=2.0, help='')

        print('build loss mask....')
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer()

        output, _, _ = self.get_outputs()
        prob = tf.nn.softmax(output, dim=-1)

        score=tf.nn.softmax_cross_entropy_with_logits(
            labels=self.target_output,
            logits=output, dim=-1)
        score_flatten=tf.reshape(score,[-1])
        mask_flatten=tf.reshape(self.mask,[-1])
        mask_score=tf.boolean_mask(score_flatten, mask_flatten)


        loss = tf.reduce_mean(mask_score)


        gradients = optimizer.compute_gradients(loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                if isinstance(clip_s, list):
                    gradients[i] = (tf.clip_by_value(grad, clip_s[0], clip_s[1]), var)
                else:
                    gradients[i] = (tf.clip_by_norm(grad, clip_s), var)


        apply_gradients = optimizer.apply_gradients(gradients)
        return output, prob, loss, apply_gradients

    def build_sampled_loss_mask(self, optimizer=None,clip_s=10, sampled_sm=100):
        print('build loss sampled mask....')

        if optimizer is None:
            optimizer = tf.train.AdamOptimizer()

        output, _, _ = self.get_outputs()#[bs. L, sampled_loss_dim]
        print(output.shape)
        prob = tf.reshape(tf.nn.softmax(self.outputProjection.project(
                                       tf.reshape(output,[-1, self.sampled_loss_dim])),
                             dim=-1),[self.batch_size,-1, self.output_size])


        labels = tf.reshape(self.target_output, [-1, 1])

        logits = tf.reshape(output, [-1, self.sampled_loss_dim])

        def sampledSoftmax(labels, inputs):

            # We need to compute the sampled_softmax_loss using 32bit floats to
            # avoid numerical instabilities.
            localWt = tf.cast(self.outputProjection.W_t, tf.float32)
            localB = tf.cast(self.outputProjection.b, tf.float32)
            localInputs = tf.cast(inputs, tf.float32)

            return tf.nn.sampled_softmax_loss(
                localWt,  # Should have shape [num_classes, dim]
                localB,
                labels,
                localInputs,
                sampled_sm,  # The number of classes to randomly sample per batch
                self.output_size)
            # The number of classes

        score = sampledSoftmax(labels, logits)
        score_flatten = tf.reshape(score, [-1])
        mask_flatten = tf.reshape(self.mask, [-1])
        mask_score = tf.boolean_mask(score_flatten, mask_flatten)

        loss = tf.reduce_mean(mask_score)


        gradients = optimizer.compute_gradients(loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_value(grad, -clip_s, clip_s), var)

        apply_gradients = optimizer.apply_gradients(gradients)
        return output, prob, loss, apply_gradients


    def print_config(self):
        return '{}.{}mem_{}dec_{}dua_{}wrp_{}wsz_{}msz_{}tea_{}att_{}per_{}hid_{}nread_{}nlayer'.\
            format(self.name, self.use_mem,
                   self.decoder_mode,
                   self.dual_controller,
                   self.write_protect,
                   self.words_num,
                   self.word_size,
                   self.use_teacher,
                   self.attend_dim,
                   self.persist_mode,
                   self.hidden_controller_dim,
                   self.read_heads_decode,
                   self.nlayer)


    @staticmethod
    def save(session, ckpts_dir, name):
        """
        saves the current values of the model's parameters to a checkpoint
        Parameters:
        ----------
        session: tf.Session
            the tensorflow session to save
        ckpts_dir: string
            the path to the checkpoints directories
        name: string
            the name of the checkpoint subdirectory
        """
        checkpoint_dir = os.path.join(ckpts_dir, name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        tf.train.Saver(tf.trainable_variables()).save(session, os.path.join(checkpoint_dir, 'model.ckpt'))

    def clear_current_mem(self,sess):
        if self.persist_mode:
            sess.run([self.assign_op_cur_mem, self.assign_op_cur_u, self.assign_op_cur_p,
                      self.assign_op_cur_L, self.assign_op_cur_ww, self.assign_op_cur_rw, self.assign_op_cur_rv])

            sess.run([self.assign_op_cur_c, self.assign_op_cur_h])



    @staticmethod
    def restore(session, ckpts_dir, name):
        """
        session: tf.Session
            the tensorflow session to restore into
        ckpts_dir: string
            the path to the checkpoints directories
        name: string
            the name of the checkpoint subdirectory
        """
        tf.train.Saver(tf.trainable_variables()).restore(session, os.path.join(ckpts_dir, name, 'model.ckpt'))

    @staticmethod
    def get_bool_rand(size_seq, prob_true=0.1):
        ret = []
        for i in range(size_seq):
            if np.random.rand() < prob_true:
                ret.append(True)
            else:
                ret.append(False)
        return np.asarray(ret)

    @staticmethod
    def get_bool_rand_incremental(size_seq, prob_true_min=0, prob_true_max=0.25):
        ret = []
        for i in range(size_seq):
            prob_true=(prob_true_max-prob_true_min)/size_seq*i
            if np.random.rand() < prob_true:
                ret.append(True)
            else:
                ret.append(False)
        return np.asarray(ret)

    @staticmethod
    def get_bool_rand_curriculum(size_seq, epoch, k=0.99, type='exp'):
        if type=='exp':
            prob_true = k**epoch
        elif type=='sig':
            prob_true = k / (k + np.exp(epoch / k))
        ret = []
        for i in range(size_seq):
            if np.random.rand() < prob_true:
                ret.append(True)
            else:
                ret.append(False)
        return np.asarray(ret)
