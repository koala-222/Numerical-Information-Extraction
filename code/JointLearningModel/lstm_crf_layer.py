# encoding=utf-8

"""
bert-blstm-crf layer
@Author:Macan
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf


def BiLSTM(rnn_inputs, keep_prop, seq_lenths, hidden_size, time_major = False, return_outputs = True, type = 'concat'):
    '''
    构建模型
    :param data:placeholder
    :param FLAGS.mem_dim:
    :return:
    '''
    if not time_major:
        rnn_inputs  = tf.transpose(rnn_inputs, [1, 0, 2])   ##time major
    cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob=keep_prop)
    cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob=keep_prop)
    '''If time_major == True, this must be a Tensor of shape: [max_time, batch_size, ...], or a nested tuple of such elements.     '''
    (fw_outputs,bw_outputs), _  = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, sequence_length = seq_lenths,
                                                      dtype=tf.float32, time_major = True)
    if return_outputs:
        if type == 'concat':
            time_major_results = tf.concat((fw_outputs,bw_outputs), 2)
            return tf.transpose(time_major_results, [1, 0, 2])
        else:
            time_major_results = tf.add(fw_outputs, bw_outputs)
            return tf.transpose(time_major_results, [1, 0, 2])
    else:
        if type == 'concat':
            return tf.reduce_mean(tf.concat((fw_outputs,bw_outputs), 2), 0)
            #return tf.reduce_max(tf.concat((fw_outputs, bw_outputs), 2), 0)
        else:
            hidden = tf.add(fw_outputs, bw_outputs)
            return tf.reduce_max(hidden, axis=0)

class DIST_FN:
    '''  function for the same operation on a list of tensors with not the same length'''

    def __init__(self, fns, names, args=None):
        self.first_use = 1
        assert isinstance(fns, list), "'fns' must be a list of function."
        self.fns = fns
        self.names = names
        assert len(self.fns) == len(self.names), "'fns' lenght must be with names."
        self.args = args
        self.args_length = 0
        if  self.args:
            self.args_length = len(self.args)
            for item in self.args:
                assert isinstance(self.args[0], list), "'args'inside part must be a list of params."

    def use_fns(self, inputs):
            arg = list()
            results = inputs
            if self.first_use  == 1:
                self.first_use = 0
                for i in range(len(self.fns)):
                    if i+1 <= self.args_length:
                        arg = self.args[i]
                    results = self.fns[i](results, *arg, name=self.names[i])
                return results
            else:
                for i in range(len(self.fns)):
                    if self.args_length and i+1 <= self.args_length:
                        arg = self.args[i]
                    try:
                        results = self.fns[i](results, *arg, reuse=True,name=self.names[i])
                    except:
                        results = self.fns[i](results, *arg, name=self.names[i])
                return results

def time_distributed(fns, incoming, names, time_major = False, args=None):
    '''
    incoming = [tf.constant([[[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]]]),tf.constant([[[0.3,0.4],[0.5,0.6],[0.7,0.8]]])]
    :param fn:
    :param incoming:
    :param name:
    :param args:
    :return:
    '''
    if not time_major:
        incoming = tf.transpose(incoming, [1, 0, 2])
    fn_dis = DIST_FN(fns, names, args)
    results = tf.map_fn(fn_dis.use_fns, incoming)

    return tf.transpose(results, [1, 0, 2])



class BLSTM_CRF(object):
    def __init__(self, embedded_chars, hidden_unit, cell_type, num_layers, dropout_rate,
                 initializers, num_labels, seq_length, labels, lengths, is_training):
        """
        BLSTM-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: LSTM的隐含单元个数
        :param cell_type: RNN类型（LSTM OR GRU DICNN will be add in feature）
        :param num_layers: RNN的层数
        :param droupout_rate: droupout rate
        :param initializers: variable init class
        :param num_labels: 标签数量
        :param seq_length: 序列最大长度
        :param labels: 真实标签
        :param lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """
        self.hidden_unit = hidden_unit
        self.dropout_rate = dropout_rate
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths
        self.embedding_dims = embedded_chars.shape[-1].value
        self.is_training = is_training

    def add_blstm_crf_layer(self, crf_only):
        """
        blstm-crf网络
        :return:
        """
        if self.is_training:
            # lstm input dropout rate i set 0.9 will get best score
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, rate=self.dropout_rate)

        if crf_only:
            logits = self.project_crf_layer(self.embedded_chars)
        else:
            # blstm
            lstm_output = self.blstm_layer(self.embedded_chars)#batch_size*max_length*2
            # project
            logits = self.project_bilstm_layer(lstm_output)#batch_size*max_length*label_num
        '''学习CRF的使用'''
        # crf
        per_example_loss, trans = self.crf_layer(logits)
        # CRF decode, pred_ids 是一条最大概率的标注路径
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
        return (per_example_loss, logits, trans, pred_ids)

    def _witch_cell(self):
        """
        RNN 类型
        :return:
        """
        cell_tmp = None
        if self.cell_type == 'lstm':
            cell_tmp = rnn.LSTMCell(self.hidden_unit)
        elif self.cell_type == 'gru':
            cell_tmp = rnn.GRUCell(self.hidden_unit)
        return cell_tmp

    def _bi_dir_rnn(self):
        """
        双向RNN
        :return:
        """
        cell_fw = self._witch_cell()
        cell_bw = self._witch_cell()
        if self.dropout_rate is not None:
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=1-self.dropout_rate)
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=1-self.dropout_rate)
        return cell_fw, cell_bw

    def blstm_layer(self, embedding_chars):
        """
        tensorflow API底层又复杂
        :return:
        """
        with tf.variable_scope('rnn_layer'):
            cell_fw, cell_bw = self._bi_dir_rnn()
            if self.num_layers > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding_chars,
                                                         dtype=tf.float32, sequence_length = self.lengths)
            outputs = tf.concat(outputs, axis=2)
        return outputs

    def project_bilstm_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.hidden_unit * 2, self.hidden_unit],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.hidden_unit], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_unit * 2])
                hidden = tf.nn.xw_plus_b(output, W, b)

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.hidden_unit, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def project_crf_layer(self, embedding_chars, name=None):
        """
        全连接 + 非线性激活
        hidden layer between input layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("logits"):
                # 初始化方法
                W = tf.get_variable("W", shape=[self.embedding_dims, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(self.embedded_chars,
                                    shape=[-1, self.embedding_dims])  # [batch_size, embedding_dims]
                pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("fast_lr_crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=self.initializers.xavier_initializer())
            if self.labels is None:
                return None, trans
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=self.labels,
                    transition_params=trans,
                    sequence_lengths=self.lengths)
                return -log_likelihood, trans
