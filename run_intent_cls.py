#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:yuanxiao
from: https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification
改动自run_highly_joint_with_lstm_crf.py，去除序列标注任务，只进行触发词分类
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import pickle
import shutil
import tensorflow as tf
import sys

sys.path.append('..')
sys.path.append('../..')
sys.path.append('.')

import JointLearningModel.calculate_model_scores as tf_metrics
from JointLearningModel.lstm_crf_layer import BLSTM_CRF, BiLSTM, DIST_FN, time_distributed
from tensorflow.contrib.layers.python.layers import initializers
from Bert import modeling
from Bert import optimization
from Bert import tokenization


# 配置log
logger = tf.compat.v1.get_logger()
logger.propagate = False

# 配置GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config.gpu_options.allow_growth = True

# tf会话
session = tf.compat.v1.Session(config=config)
flags = tf.flags
FLAGS = flags.FLAGS

# important ==========================================================
BERT_MODEL_DIR = '../wwm_cased_L-24_H-1024_A-16/'
DATA_DIR = '../standard_dataset/split_files/'
OUTPUT_DIR = '../results/split_result_highly_lstm_crf/'
# ====================================================================

flags.DEFINE_string("data_dir", DATA_DIR, "The input datadir.")
flags.DEFINE_string("task_name", 'quantity', "The name of the task to train.")
# 配置BERT
flags.DEFINE_string("vocab_file", os.path.join(BERT_MODEL_DIR, 'vocab.txt'), "The vocabulary file")
flags.DEFINE_string("bert_config_file", os.path.join(BERT_MODEL_DIR, 'bert_config.json'), "The BERT config json file.")
flags.DEFINE_string("init_checkpoint", os.path.join(BERT_MODEL_DIR, 'bert_model.ckpt'),
                    "Initial checkpoint from a pre-trained BERT model.")
flags.DEFINE_string("output_dir", OUTPUT_DIR, "The output directory where the model checkpoints will be written.")

flags.DEFINE_bool("do_lower_case", False, "Whether to lower case the input text.")
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")
flags.DEFINE_bool("calculate_model_score", True, "Calculate model score on test data")

flags.DEFINE_integer("max_seq_length", 80, "The maximum total input sequence length after WordPiece tokenizing.")
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_integer("fast_lr_ratio", 1, "The ratio for layers need a fast learning rate.")
flags.DEFINE_float("slot_loss_ratio", 0.75, "The ratio for slot task in ratio.")
flags.DEFINE_float("num_train_epochs", 10.0, "Total number of training epochs to perform.")
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate')
flags.DEFINE_integer('lstm_size', 384, 'LSTM size')
flags.DEFINE_float("warmup_proportion", 0.1,
                   "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
flags.DEFINE_integer("save_checkpoints_steps", 50, "How often to save the model checkpoint.")
flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")

# 用于预测API ===========================================================================
flags.DEFINE_string("saved_checkpoint", "", "Saved checkpoint for slot prediction.")
# ======================================================================================

# not used
flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_integer("num_tpu_cores", 8, "Only used if `use_tpu` is True. Total number of TPU cores to use.")
flags.DEFINE_string("tpu_name", None, "The Cloud TPU to use for training.")
flags.DEFINE_string("tpu_zone", None, "[Optional] GCE zone where the Cloud TPU is located in.")
flags.DEFINE_string("gcp_project", None, "[Optional] Project name for the Cloud TPU-enabled project.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the input sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the labeling sequence(Slot Filling).
            specified for train and dev examples, but not for test examples.
          label: (Optional) string. The label(Intent Prediction) of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a  # 文本字符串，用空格分割
        self.text_b = text_b  # 序列标签字符串，用空格分隔
        self.label = label  # 类别标签


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    batches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 input_ids,
                 slot_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_value_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.slot_ids = slot_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_value_ids = is_value_ids
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class Quantity_Joint_LSTM_CRF_Processor(DataProcessor):
    def get_examples(self, data_dir):
        '''
        加载整个数据集
        :param data_dir: 数据集文件夹
        :return: [(text, CRF_label, class_label), ...]
        '''
        path_seq_in = os.path.join(data_dir, "seq.in")
        path_seq_out = os.path.join(data_dir, "seq.out")
        path_label = os.path.join(data_dir, "label")
        seq_in_list, seq_out_list, label_list = [], [], []
        with open(path_seq_in) as seq_in_f:
            with open(path_seq_out) as seq_out_f:
                with open(path_label) as label_f:
                    for seqin, seqout, label in zip(seq_in_f.readlines(), seq_out_f.readlines(), label_f.readlines()):
                        '''单条数据'''
                        seqin_words = [word for word in seqin.split() if len(word) > 0]  # 按照空白符手动进行分割
                        seqout_words = [word for word in seqout.split() if len(word) > 0]
                        assert len(seqin_words) == len(seqout_words)

                        seq_in_list.append(" ".join(seqin_words))
                        seq_out_list.append(" ".join(seqout_words))
                        label_list.append(label.strip())
            lines = list(zip(seq_in_list, seq_out_list, label_list))  # list of tuples
            return lines

    def get_train_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "valid")), "valid")

    def get_test_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "test")), "test")

    def get_slot_labels_from_files(self, data_dir):
        label_set = set()
        for f_type in ["train", "valid", "test"]:
            seq_out_dir = os.path.join(os.path.join(data_dir, f_type), "seq.out")
            with open(seq_out_dir) as data_f:
                seq_sentence_list = [seq.split() for seq in data_f.readlines()]
                seq_word_list = [word for seq in seq_sentence_list for word in seq]
                label_set = label_set | set(seq_word_list)
        label_list = list(label_set)
        label_list.sort()
        return ["[Padding]", "[##WordPiece]", "[CLS]", "[SEP]"] + label_list

    def get_slot_labels(self):
        '''
        :return: CRF标签
        '''
        return ['[Padding]', '[##WordPiece]', '[CLS]', '[SEP]',
                'O', 'B-Attribute', 'I-Attribute', 'B-Modifier', 'I-Modifier',
                'B-Value', 'I-Value', 'B-Unit', 'I-Unit',
                'B-Vtype', 'I-Vtype', 'B-Whole', 'I-Whole',
                'B-Qobject', 'I-Qobject']

    def get_intent_labels(self):
        '''
        :return: 获取触发词
        '''
        return ['Measure', 'Counting', 'Reference', 'Proportion',
                'Ordinal', 'Naming', 'Other']

    def _create_example(self, lines, set_type):
        """
        添加unique id并转化为unicode编码
        :param lines: [(text, CRF_label, class_label), ...]
        :param set_type: 'train', 'valid' or 'test'
        :return:
        """
        examples = []
        for i, sample in enumerate(lines):
            guid = "%s-%s" % (set_type, i)  # unique id
            text_a = tokenization.convert_to_unicode(sample[0])
            text_b = tokenization.convert_to_unicode(sample[1])
            label = tokenization.convert_to_unicode(sample[2])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_single_example(ex_index, example, slot_label_list, intent_label_list, max_seq_length,
                           tokenizer):
    """
    将一个样本（字符串）转化为特征（数），类比于__get_item__()
    Converts a single InputExample into a single TFRecord InputFeatures.
    """
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            slot_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_value_ids=[0] * max_seq_length,
            is_real_example=False
        )

    slot_label_map = {}
    for i, label in enumerate(slot_label_list):
        slot_label_map[label] = i
    with open(os.path.join(FLAGS.output_dir, "slot_label2id.pkl"), 'wb') as w:
        pickle.dump(slot_label_map, w)

    intent_label_map = {}
    for i, label in enumerate(intent_label_list):
        intent_label_map[label] = i
    with open(os.path.join(FLAGS.output_dir, "intent_label2id.pkl"), 'wb') as w:
        pickle.dump(intent_label_map, w)

    text_a_list = example.text_a.split(" ")
    text_b_list = example.text_b.split(" ")

    # 分词 =============================================
    # 先按空格分词保证对应关系，然后tokenize分词，适应bert
    tokens_a = []
    slots_b = []
    for i, word in enumerate(text_a_list):
        token_a = tokenizer.tokenize(word)
        tokens_a.extend(token_a)
        slot_i = text_b_list[i]
        for m in range(len(token_a)):
            if m == 0:
                slots_b.append(slot_i)
            else:
                slots_b.append("[##WordPiece]")
    # ==================================================
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:max_seq_length - 2]
        slots_b = slots_b[0:max_seq_length - 2]

    # convert to id
    tokens = []
    slot_ids = []  # 序列标注id
    segment_ids = []  # 分块id，均设为0
    is_value_ids = []  # 是否为触发词，0或1
    tokens.append("[CLS]")
    slot_ids.append(slot_label_map["[CLS]"])
    is_value_ids.append(0)
    segment_ids.append(0)
    for i, token in enumerate(tokens_a):
        tokens.append(token)
        slot_ids.append(slot_label_map[slots_b[i]])
        segment_ids.append(0)
        if 'Value' in slots_b[i]:  # mark trigger
            is_value_ids.append(1)
        else:
            is_value_ids.append(0)

    tokens.append("[SEP]")
    # append "O" or append "[SEP]" not sure!
    # [SEP]比较合理
    slot_ids.append(slot_label_map["[SEP]"])
    segment_ids.append(0)
    is_value_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens get attention
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        is_value_ids.append(0)
        slot_ids.append(0)
        tokens.append("[Padding]")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(slot_ids) == max_seq_length
    assert len(is_value_ids) == max_seq_length

    label_id = intent_label_map[example.label]

    # print 5 examples
    if ex_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % example.guid)
        logger.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("slots_ids: %s" % " ".join([str(x) for x in slot_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("is_value_ids: %s" % " ".join([str(x) for x in is_value_ids]))
        logger.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        slot_ids=slot_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_value_ids=is_value_ids,
        is_real_example=True
    )

    return feature


def file_based_convert_examples_to_features(
        examples, slot_label_list, intent_label_list, max_seq_length, tokenizer, output_file):
    """
    将InputExamples转成tf_record，并写入文件
    Convert a set of InputExample to a TFRecord file.
    :param examples: [(text, CRF_label, class_label), ...]
    :param slot_label_list: CRF标签列表(String)
    :param intent_label_list: 触发词类别列表(String)
    :param max_seq_length:
    :param tokenizer:
    :param output_file: TFRecord file
    :return:
    """
    writer = tf.io.TFRecordWriter(output_file)

    for ex_index, example in enumerate(examples):
        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        if ex_index % 10000 == 0:
            logger.info("Writing example %d of length %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, slot_label_list, intent_label_list,
                                         max_seq_length, tokenizer)

        # convert to tensorflow format
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["slot_ids"] = create_int_feature(feature.slot_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features['is_value_ids'] = create_int_feature(feature.is_value_ids)
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())  # 写入一个样本到tf_record
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """
    类比于DataLoader
    Creates an `input_fn` closure to be passed to TPUEstimator.
    :param input_file:
    :param seq_length:
    :param is_training:
    :param drop_remainder:
    :return:
    """
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "slot_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_value_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """
        将record转化为tf张量格式，并转化为int32
        Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32. So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder)
        )
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, is_value_ids,
                 slot_label_ids, intent_label_ids, num_slot_labels, num_intent_labels,
                 use_one_hot_embeddings):
    """
    Creates a sequence labeling and classification model.
    BERT + BiLSTM + CRF(or MLP)
    """
    model = modeling.BertModel(  # Google Bert model
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    # Gets final hidden layer of encoder.
    # Returns:
    #   float Tensor of shape [batch_size, seq_length, hidden_size]
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value
    is_value_ids = tf.reshape(is_value_ids, [-1, max_seq_length, 1])
    is_value_ids = tf.cast(is_value_ids, tf.float32)
    embedding = embedding + is_value_ids  # 需要知道is_value的信息!!!

    used = tf.sign(tf.abs(input_ids))
    hidden_unit = FLAGS.lstm_size
    lengths = tf.reduce_sum(used, reduction_indices=1)  # 当前batch中的序列真实长度, [batch_size]大小的向量
    dropout_rate = FLAGS.dropout_rate

    if is_training:
        embedding = tf.nn.dropout(embedding, rate=dropout_rate)

    # LSTM layer
    with tf.variable_scope("fast_lr_lstm_layer"):
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_unit)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_unit)

        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=1 - dropout_rate)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=1 - dropout_rate)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw, inputs=embedding, dtype=tf.float32, sequence_length=lengths)
        lstm_outputs = tf.concat(outputs, axis=2)

    # 触发词分类损失
    with tf.variable_scope("fast_lr_intent_loss"):
        with tf.variable_scope("pooler"):
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token. We assume that this has been pre-trained.
            # float Tensor of shape [batch_size, hidden_size]
            first_token_tensor = tf.squeeze(lstm_outputs[:, 0:1, :], axis=1)
            pooled_output = tf.layers.dense(  # 全连接层
                first_token_tensor,
                hidden_unit,
                activation=tf.tanh,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        intent_output_layer = pooled_output
        # print(intent_output_layer.shape) # batch_size * 384
        intent_hidden_size = intent_output_layer.shape[-1].value

        # 下面是全连接 + softmax
        intent_output_weights = tf.get_variable(
            "intent_output_weights", [num_intent_labels, intent_hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        intent_output_bias = tf.get_variable(
            "intent_output_bias", [num_intent_labels], initializer=tf.zeros_initializer())
        if is_training:
            intent_output_layer = tf.nn.dropout(intent_output_layer, rate=dropout_rate)
        intent_logits = tf.matmul(intent_output_layer, intent_output_weights, transpose_b=True)
        intent_logits = tf.nn.bias_add(intent_logits, intent_output_bias)
        intent_probabilities = tf.nn.softmax(intent_logits, axis=-1)

        intent_log_probs = tf.nn.log_softmax(intent_logits, axis=-1)
        intent_predictions = tf.argmax(intent_logits, axis=-1)
        intent_one_hot_labels = tf.one_hot(intent_label_ids, depth=num_intent_labels, dtype=tf.float32)

        # 手动计算交叉熵损失
        intent_per_example_loss = -tf.reduce_sum(intent_one_hot_labels * intent_log_probs, axis=-1)
        intent_loss = tf.reduce_mean(intent_per_example_loss)
        # return (intent_loss, intent_per_example_loss, intent_logits, intent_probabilities, intent_predictions)

    # 序列标注损失
    with tf.variable_scope("fast_lr_slot_loss"):
        # CRF output layer
        with tf.variable_scope("project_layer"):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[hidden_unit * 2, hidden_unit],
                                    dtype=tf.float32, initializer=initializers.xavier_initializer())
                b = tf.get_variable("b", shape=[hidden_unit], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, hidden_unit * 2])
                hidden = tf.nn.xw_plus_b(output, W, b)

            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[hidden_unit, num_slot_labels],  # project to score of tags
                                    dtype=tf.float32, initializer=initializers.xavier_initializer())
                b = tf.get_variable("b", shape=[num_slot_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                slot_logits = tf.nn.xw_plus_b(hidden, W, b)

        slot_logits = tf.reshape(slot_logits, [-1, FLAGS.max_seq_length, num_slot_labels])

        with tf.variable_scope("fast_lr_crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[num_slot_labels, num_slot_labels],
                initializer=initializers.xavier_initializer())
            if slot_label_ids is None:
                log_likelihood = None
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=slot_logits,
                    tag_indices=slot_label_ids,
                    transition_params=trans,
                    sequence_lengths=lengths)
            slot_per_example_loss = -log_likelihood  # 对数似然损失
            slot_predictions, _ = tf.contrib.crf.crf_decode(
                potentials=slot_logits,
                transition_params=trans,
                sequence_length=lengths)
            slot_loss = tf.reduce_mean(slot_per_example_loss)

    # modified total_loss
    loss = intent_loss + FLAGS.slot_loss_ratio * slot_loss
    # loss = intent_loss + 0.00001 * slot_loss  # BAD RESULTS

    return (
        loss,
        intent_loss,
        intent_per_example_loss,
        intent_logits,
        intent_predictions,
        slot_loss,
        slot_per_example_loss,
        slot_logits,
        slot_predictions)


def model_fn_builder(bert_config, num_slot_labels, num_intent_labels, init_checkpoint,
                     learning_rate, num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        """
        一个完整的训练过程
        :param features: 输入的一批特征
        :param labels: 输入的标签数据
        :param mode: train, eval or predict
        :param params: 超参数，由Estimator传来
        :return:
        """
        logger.info("*** Features ***")
        for name in sorted(features.keys()):
            logger.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        slot_label_ids = features["slot_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        intent_label_ids = features["label_ids"]
        is_value_ids = features["is_value_ids"]
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(intent_label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (  # 投入模型
            total_loss,
            intent_loss,
            intent_per_example_loss,
            intent_logits,
            intent_predictions,
            slot_loss,
            slot_per_example_loss,
            slot_logits,
            slot_predictions
        ) = create_model(bert_config, is_training, input_ids, input_mask, segment_ids, is_value_ids,
                         slot_label_ids, intent_label_ids, num_slot_labels, num_intent_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()  # 获取模型中所有的训练参数
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:  # 加载预训练的Bert模型
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        logger.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logger.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps,
                                                     num_warmup_steps, use_tpu, fast_lr_ratio=FLAGS.fast_lr_ratio)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn
            )
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(intent_per_example_loss, intent_label_ids, intent_logits,
                          slot_per_example_loss, slot_label_ids, slot_logits, is_real_example):
                """
                计算评价指标函数
                :param intent_per_example_loss:
                :param intent_label_ids:
                :param intent_logits:
                :param slot_per_example_loss:
                :param slot_label_ids:
                :param slot_logits:
                :param is_real_example:
                :return:
                """
                intent_predictions = tf.argmax(intent_logits, axis=-1, output_type=tf.int32)
                intent_accuracy = tf.metrics.accuracy(
                    labels=intent_label_ids, predictions=intent_predictions, weights=is_real_example)
                # print("=============================")
                # print(intent_label_ids.numpy())  # ERROR
                # print(intent_predictions.numpy())
                # print(type(intent_label_ids))
                # print(type(intent_predictions))
                # print("=============================")
                intent_loss = tf.metrics.mean(values=intent_per_example_loss, weights=is_real_example)

                slot_predictions = tf.argmax(slot_logits, axis=-1, output_type=tf.int32)
                slot_pos_indices_list = list(range(num_slot_labels))[
                                        4:]  # ["[Padding]","[##WordPiece]", "[CLS]", "[SEP]"] + seq_out_set
                pos_indices_list = slot_pos_indices_list[:-1]  # do not care "O"

                # 自定义的指标计算方法，是包括"O"类型在内的
                slot_precision_macro = tf_metrics.precision(slot_label_ids, slot_predictions, num_slot_labels,
                                                            slot_pos_indices_list, average="macro")
                slot_recall_macro = tf_metrics.recall(slot_label_ids, slot_predictions, num_slot_labels,
                                                      slot_pos_indices_list, average="macro")
                slot_f_macro = tf_metrics.f1(slot_label_ids, slot_predictions, num_slot_labels, slot_pos_indices_list,
                                             average="macro")

                slot_precision_micro = tf_metrics.precision(slot_label_ids, slot_predictions, num_slot_labels,
                                                            slot_pos_indices_list, average="micro")
                slot_recall_micro = tf_metrics.recall(slot_label_ids, slot_predictions, num_slot_labels,
                                                      slot_pos_indices_list, average="micro")
                slot_f_micro = tf_metrics.f1(slot_label_ids, slot_predictions, num_slot_labels, slot_pos_indices_list,
                                             average="micro")

                slot_loss = tf.metrics.mean(values=slot_per_example_loss, weights=is_real_example)  # 去除padding

                return {
                    "eval_intent_accuracy": intent_accuracy,
                    "eval_intent_loss": intent_loss,
                    "eval_slot_precision(macro)": slot_precision_macro,
                    "eval_slot_recall(macro)": slot_recall_macro,
                    "eval_slot_f(macro)": slot_f_macro,
                    "eval_slot_precision(micro)": slot_precision_micro,
                    "eval_slot_recall(micro)": slot_recall_micro,
                    "eval_slot_f(micro)": slot_f_micro,
                    "eval_slot_loss": slot_loss,
                }

            eval_metrics = (metric_fn,
                            [intent_per_example_loss, intent_label_ids, intent_logits,
                             slot_per_example_loss, slot_label_ids, slot_logits, is_real_example])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"intent_predictions": intent_predictions,
                             "slot_predictions": slot_predictions},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def main(_):
    # ----------------for code test------------------
    if FLAGS.do_train:
        if os.path.exists(FLAGS.output_dir):
            try:
                os.removedirs(FLAGS.output_dir)
                os.makedirs(FLAGS.output_dir)
            except:
                logger.info("***** Running evaluation *****")
                logger.warning(FLAGS.output_dir + " is not empty, here use shutil.rmtree(FLAGS.output_dir)!")
                shutil.rmtree(FLAGS.output_dir)  # remove dir recurrently
                os.makedirs(FLAGS.output_dir)
        else:
            os.makedirs(FLAGS.output_dir)
    # ----------------for code test------------------

    logger.setLevel('INFO')
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of 'do_train, 'do_eval' or 'do_predict' must be True.")

    # Bert.modeling
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings)
        )

    tf.gfile.MakeDirs(FLAGS.output_dir)

    # 数据加载器
    processor = Quantity_Joint_LSTM_CRF_Processor()
    intent_label_list = processor.get_intent_labels()  # 触发词
    slot_label_list = processor.get_slot_labels()  # CRF标签

    intent_id2label = {}
    for i, label in enumerate(intent_label_list):
        intent_id2label[i] = label
    slot_id2label = {}
    for i, label in enumerate(slot_label_list):
        slot_id2label[i] = label

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project
        )

    # https: // tensorflow.google.cn / api_docs / python / tf / compat / v1 / estimator / tpu / TPUConfig?hl = en
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=1000,
        model_dir=FLAGS.saved_checkpoint,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        )
    )

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)  # list of InputExample()
        num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)  # 所有epoch的总步数
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_slot_labels=len(slot_label_list),
        num_intent_labels=len(intent_label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu
    )

    # If TPU is not available, this will fall back to normal Estimator on GPU or CPU.
    # 分类器
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        model_dir=FLAGS.saved_checkpoint,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size
    )

    '''训练、验证、测试'''
    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(  # 将样本转化为feature，并存入TF_RECORD文件
            train_examples, slot_label_list, intent_label_list, FLAGS.max_seq_length, tokenizer, train_file)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", FLAGS.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(  # 类比于DataLoader
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True
        )
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        """TPU占用了大量的篇幅"""
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number of examples must be a multiple
            # of the batch size, or else examples will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics support a per-instance weight, and
            # these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, slot_label_list, intent_label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        logger.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder
        )

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number of examples must be a multiple
            # of the batch size, or else examples will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics support a per-instance weight, and
            # these get a weight of 0.0).
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, slot_label_list, intent_label_list,
                                                FLAGS.max_seq_length, tokenizer, predict_file)

        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        logger.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        intent_output_predict_file = os.path.join(FLAGS.output_dir, "intent_prediction_test_results.txt")
        slot_output_predict_file = os.path.join(FLAGS.output_dir, "slot_filling_test_results.txt")

        # 解析预测结果
        with tf.gfile.GFile(intent_output_predict_file, "w") as intent_writer:
            with tf.gfile.GFile(slot_output_predict_file, "w") as slot_writer:
                num_written_lines = 0
                logger.info("***** Intent Predict and Slot Filling results *****")
                for i, prediction in enumerate(result):
                    intent_prediction = prediction["intent_predictions"]
                    slot_predictions = prediction["slot_predictions"]
                    if i >= num_actual_predict_examples:
                        break

                    intent_output_line = str(intent_id2label[intent_prediction]) + "\n"  # 分类结果
                    intent_writer.write(intent_output_line)

                    slot_output_line = " ".join(  # 序列标注结果
                        slot_id2label[id] for id in slot_predictions) + "\n"  # if id != 0 0--->"[Padding]"
                    slot_writer.write(slot_output_line)

                    num_written_lines += 1
                assert num_written_lines == num_actual_predict_examples

        if FLAGS.calculate_model_score:
            path_to_label_file = os.path.join(FLAGS.data_dir, "test")
            path_to_predict_label_file = FLAGS.output_dir
            log_out_file = path_to_predict_label_file
            if FLAGS.task_name.lower() == "quantity":  # 计算评价指标
                intent_slot_reports = tf_metrics.Quantity_Slot_Filling_and_Intent_Detection_Calculate(
                    path_to_label_file, path_to_predict_label_file, log_out_file)
            else:
                raise ValueError("Not this calculate_model_score")
            result_intent = intent_slot_reports.show_intent_prediction_report(store_report=True)
            result_slot = intent_slot_reports.show_slot_filling_report(store_report=True)
            with open("./log.log", "a", encoding="utf-8") as f:
                f.write("=" * 30 + "\n")
                f.write(FLAGS.saved_checkpoint + "\n")
                for k, v in result_intent.items():
                    f.write(str(k) + "\t" + str(v) + "\n")
                f.write("=" * 30 + "\n")


if __name__ == "__main__":
    tf.compat.v1.app.run()


"""
2021-07-04 01:58:23
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_intent_accuracy = 0.8545455
INFO:tensorflow:  eval_intent_loss = 0.5996723
INFO:tensorflow:  eval_slot_f(macro) = 0.7628071
INFO:tensorflow:  eval_slot_f(micro) = 0.99333096
INFO:tensorflow:  eval_slot_loss = 2.0719206
INFO:tensorflow:  eval_slot_precision(macro) = 0.80826795
INFO:tensorflow:  eval_slot_precision(micro) = 0.9935447
INFO:tensorflow:  eval_slot_recall(macro) = 0.73739433
INFO:tensorflow:  eval_slot_recall(micro) = 0.9931173
INFO:tensorflow:  global_step = 689
INFO:tensorflow:  loss = 2.1158006
"""

"""
---show_intent_prediction_report---
2021-07-04 02:10:16
准确率: 0.20938628158844766
宏平均精确率: 0.07711442786069651
微平均精确率: 0.20938628158844766
加权平均精确率: 0.08320760816854356
宏平均召回率: 0.16498520312079634
微平均召回率: 0.20938628158844766
加权平均召回率: 0.20938628158844766
/home/klhao/.conda/envs/klhao-tensorflow1/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1465: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no true nor predicted samples. Use `zero_division` parameter to control this behavior.
  average, "true nor predicted", 'F-score is', len(true_sum)
宏平均F1-score: 0.05601458772579634
微平均F1-score: 0.20938628158844766
加权平均F1-score: 0.08084799384827177


------------------------------------------------------------
---show_slot_filling_report---
2021-07-04 02:10:16
准确率: 0.002181691247054717
宏平均精确率: 0.0025871310070408097
微平均精确率: 0.002181691247054717
加权平均精确率: 0.0006620915527031852
/home/klhao/.conda/envs/klhao-tensorflow1/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1221: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
宏平均召回率: 0.035830052654219145
微平均召回率: 0.002181691247054717
加权平均召回率: 0.002181691247054717
宏平均F1-score: 0.004831093499376966
微平均F1-score: 0.005387350501023597
加权平均F1-score: 0.01751689691147007

"""

"""
最终采用689模型，暂时解决问题
---show_intent_prediction_report---
2021-07-04 03:37:53
准确率: 0.8592057761732852
宏平均精确率: 0.8384156423409471
微平均精确率: 0.8592057761732852
加权平均精确率: 0.8594680741966007
宏平均召回率: 0.8182231732353297
微平均召回率: 0.8592057761732852
加权平均召回率: 0.8592057761732852
/home/klhao/.conda/envs/klhao-tensorflow1/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1465: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no true nor predicted samples. Use `zero_division` parameter to control this behavior.
  average, "true nor predicted", 'F-score is', len(true_sum)
宏平均F1-score: 0.7083456671708459
微平均F1-score: 0.8592057761732852
加权平均F1-score: 0.8585093788286966


------------------------------------------------------------
---show_slot_filling_report---
2021-07-04 03:37:53
准确率: 0.8504983388704319
宏平均精确率: 0.776906844956044
微平均精确率: 0.8504983388704319
加权平均精确率: 0.8922870306760866
/home/klhao/.conda/envs/klhao-tensorflow1/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1221: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
宏平均召回率: 0.7239335076235502
微平均召回率: 0.8504983388704319
加权平均召回率: 0.8504983388704319
宏平均F1-score: 0.7996160286720917
微平均F1-score: 0.9102222222222222
加权平均F1-score: 0.9082880824212634

"""

