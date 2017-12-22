import numpy as np
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import time
import json
import pickle

#DATA_PATH = "data/rnn_samples_new_repeat.json"
DATA_PATH = "data/rnn_samples_new.json"
HIDDEN_SIZE = 100
EMBED_SIZE = 200
BATCH_SIZE = 128
NUM_STEPS = 17
SKIP_STEP = 50
TEMPRATURE = 0.7
LR = 0.001
LEN_GENERATED = 20


def make_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass

def getCandidateMap(candidate):
    assert type(candidate) == list
    topic2index = {}
    index2topic = {}
    for i, j in enumerate(candidate):
        topic2index[j] = i+1
        index2topic[i+1] = j
    """
    tmp = i+1
    topic2index["EOS"] = tmp+1
    index2topic[tmp+1] = "EOS"
    """

    #"sentences include '0', it's a problem"
    topic2index["0"] = 0
    index2topic[0] = "0"
    return topic2index, index2topic

def words_encode(text, t2i):
    """
    :param text: ["A", "b1", "b2", ... , "bn"]
    :param t2i: topics_to_index for all words in words table
    :return: [123, 23, 1, 34, 53,..., 6301]
    """
    return [t2i[x] for x in text if x in t2i]

def words_decode(encode_text, i2t):
    return [i2t[x] for x in encode_text]

def read_data(filename, t2i):
    with open(filename, "r") as f:
        data = json.load(f)
    for text in data:
        text = words_encode(text, t2i)
        text += [0] * (NUM_STEPS - len(text))
        yield text

def read_batch(stream, batch_size=BATCH_SIZE):
    batch = []
    for element in stream:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch

def get_length(sequence):
    # batch_size * NUM_STEP * VOCAB_SIZE
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

def create_rnn(seq, hidden_size=HIDDEN_SIZE):
    #cell = tf.contrib.rnn.GRUCell(hidden_size)
    with tf.variable_scope("GRU", reuse=True) as scope:
        cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    in_state = tf.placeholder_with_default(
                cell.zero_state(tf.shape(seq)[0], tf.float32), [None, hidden_size])
    #length = tf.reduce_sum(tf.reduce_max(tf.sign(seq), 2), 1) + 1
    length = get_length(seq)
    output, out_state = tf.nn.dynamic_rnn(cell, seq, length, in_state, scope="GRU")
    return output, in_state, out_state


def create_model(seq, temp, i2t, hidden=HIDDEN_SIZE, num_samples=512):
    one_hot = tf.one_hot(seq, len(i2t))
    embed_matrix = tf.Variable(tf.constant(0.0, shape=[len(i2t), EMBED_SIZE]), trainable=False, name="embed_matrix")
    embedding_placehoder = tf.placeholder(dtype=tf.float32, shape=[len(i2t), EMBED_SIZE])
    embedding_init = embed_matrix.assign(embedding_placehoder)
    seq = tf.nn.embedding_lookup(embed_matrix, seq, name="embed")
    output, in_state, out_state = create_rnn(seq, hidden)
    #full softmax
    logits = tf.contrib.layers.fully_connected(output, len(i2t), None)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, :-1], labels=one_hot[:, 1:]))
    mask = tf.sign(tf.reduce_max(tf.abs(one_hot[:, 1:]), 2))
    loss *= mask
    loss = tf.reduce_sum(loss, 1) * 1.0 / tf.reduce_sum(mask, 1)

    #sampled-softmax-loss
    proj_w = tf.get_variable("proj_w", [len(i2t), HIDDEN_SIZE], dtype=tf.float32)
    proj_b = tf.get_variable("proj_b", [len(i2t)], dtype=tf.float32)
    sampled_loss = tf.nn.sampled_softmax_loss(
        weights=proj_w,
        biases=proj_b,
        labels=seq[:, 1:],
        inputs=output,
        num_sampled=512,
        num_classes=len(i2t)
    )
    sampled_loss *= mask
    sampled_loss = tf.reduce_mean(sampled_loss, 1) * 1.0 / tf.reduce_sum(mask, 1)

    y_hat = tf.nn.softmax(logits)
    sample = tf.argmax(y_hat, 0)
    #sample = tf.multinomial(tf.exp(logits[:, -1]/temp), 1)[:, 0]

    return sampled_loss, loss, sample, in_state, out_state, embedding_init, embedding_placehoder

class RNN(object):
    def __init__(self, path_vocab="data/vocab_table.json", path_emb="data/vocab_emb.pkl"):
        with open(path_vocab, "r") as f:
            self.candidate = json.load(f)
        with open(path_emb, "rb") as f:
            self.embedding = pickle.load(f)
        self.embedding = [list(np.zeros(200))] + self.embedding # add 0 vector

        self.t2i, self.i2t = getCandidateMap(self.candidate)
        self.seq = tf.placeholder(tf.int32, [None, None])
        self.one_hot = tf.one_hot(self.seq, len(self.i2t), dtype=tf.int32)
        self.embed_matrix = tf.Variable(tf.constant(0.0, shape=[len(self.i2t), EMBED_SIZE]), trainable=False, name="embed_matrix")
        self.embedding_placehoder = tf.placeholder(dtype=tf.float32, shape=[len(self.i2t), EMBED_SIZE])
        self.embedding_init = self.embed_matrix.assign(self.embedding_placehoder)
        self.seq2 = tf.nn.embedding_lookup(self.embed_matrix, self.seq, name="embed")

        """create_rnn content"""
        """
        cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
        in_state = tf.placeholder_with_default(
            cell.zero_state(tf.shape(seq2)[0], tf.float32), [None, HIDDEN_SIZE])
        length = tf.reduce_sum(tf.reduce_max(tf.sign(seq2), 2), 1) + 1
        output, out_state = tf.nn.dynamic_rnn(cell, seq2, length, in_state)
        #return output, in_state, out_state
        """

        self.output, self.in_state, self.out_state = create_rnn(self.seq2, HIDDEN_SIZE)

        #full softmax
        self.mask = tf.cast(tf.sign(tf.reduce_max(tf.abs(self.one_hot[:, 1:]), 2)), dtype=tf.float32)
        #self.loss *= self.mask
        #self.loss = tf.reduce_sum(self.loss, 1) * 1.0 / tf.reduce_sum(self.mask, 1)

        #sampled-softmax-loss
        self.proj_w = tf.get_variable("proj_w", [len(self.i2t), HIDDEN_SIZE], dtype=tf.float32)
        self.proj_b = tf.get_variable("proj_b", [len(self.i2t)], dtype=tf.float32)
        self.full_sample_loss = 0
        for i in range(NUM_STEPS-1):
            sampled_loss = tf.nn.sampled_softmax_loss(
                weights=self.proj_w,
                biases=self.proj_b,
                labels=tf.reshape(self.seq[:, i+1], [-1, 1]),
                inputs=self.output[:,i,:],
                num_classes=len(self.i2t),
                num_sampled=512,
            )
            self.full_sample_loss+=sampled_loss * self.mask[:, i]

        #self.full_sample_loss *= self.mask
        self.full_sample_loss_final = tf.reduce_sum(self.full_sample_loss) * 1.0 / tf.reduce_sum(self.mask)

        self.logits = tf.matmul(tf.reshape(self.output, shape=[-1, HIDDEN_SIZE]), tf.transpose(self.proj_w)) + self.proj_b
        #self.logits = tf.contrib.layers.fully_connected(self.output, len(self.i2t), None)
        #self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits[:, :-1], labels=self.one_hot[:, 1:])
        self.y_hat = tf.nn.softmax(self.logits)
        self.sample = tf.argmax(self.y_hat, 1)
        #sample = tf.multinomial(tf.exp(logits[:, -1]/temp), 1)[:, 0]

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        #-----optimizer-------
        self.optimizer = tf.train.GradientDescentOptimizer(LR).minimize(self.full_sample_loss_final, global_step=self.global_step)
        make_dir("checkpoints")
        make_dir("checkpoints/rnn")

        #*********************************************************
        #                       training model
        #*********************************************************

        #output, in_state, out_state = create_rnn(seq2, HIDDEN_SIZE)

    def train(self, isTraining=True, area="deep_learning"):
        self.saver = tf.train.Saver()
        start = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.embedding_init, feed_dict={self.embedding_placehoder:self.embedding})

            ckpt = tf.train.get_checkpoint_state(os.path.dirname("checkpoints/rnn/checkpoints"))
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            if isTraining:
                iteration = self.global_step.eval()
                for epoch in range(20):
                    epoch_time = time.time()
                    epoch_loss = 0
                    for batch in read_batch(read_data(DATA_PATH, t2i=self.t2i)):
                        batch_loss, _ = sess.run([self.full_sample_loss_final, self.optimizer], {self.seq: batch})
                        epoch_loss += batch_loss

                        if (iteration + 1) % SKIP_STEP == 0:
                            print("Iter {}. \n LOSS {}. Time{}".format(iteration, batch_loss, time.time() - start))
                            sentence = ["machine_learning"]
                            #prob = [1.0]
                            state = None
                            for _ in range(LEN_GENERATED):
                                batch = [words_encode([sentence[-1]], t2i=self.t2i)]
                                feed = {self.seq:batch}

                                if state is not None:
                                    feed.update({self.in_state: state})
                                index, state = sess.run([self.sample, self.out_state], feed)
                                #print(y_hat)
                                #print(len(y_hat[0]))
                                #prob.append(list(y_hat[0][index])[0])
                                words = words_decode(index, self.i2t)
                                if words[0] == "EOS":
                                    sentence.extend(words)
                                    break
                                elif words[0] == "0":
                                    break
                                else:
                                    sentence.extend(words)
                            print(sentence)
                            #print(prob)

                            self.saver.save(sess, "checkpoints/rnn/result", iteration)
                        iteration += 1
                    print("---------------------------------------------------------------")
                    print("Epoch {}. \n LOSS {}. Time{}".format(epoch, epoch_loss, time.time() - epoch_time))
                print("Total time: {}.".format(time.time()-start))

            else:
                sentence = [area]
                state = None
                for _ in range(LEN_GENERATED):
                    batch = [words_encode([sentence[-1]], t2i=self.t2i)]
                    feed = {self.seq:batch}
                    if state is not None:
                        feed.update({self.in_state: state})
                    index, state = sess.run([self.sample, self.out_state], feed)
                    words = words_decode(index[-1], self.i2t)
                    if words[0] == "EOS":
                        sentence.extend(words)
                        break
                    elif words[0] == "0":
                        break
                    else:
                        sentence.extend(words)
                print(sentence)
                return sentence

    def set_area(self, area):
        self.area = area



if __name__ == "__main__":
    rnnModel = RNN()
    rnnModel.train(isTraining=True, area="machine_learning")
    #main2(isTraining=False, area="computer_science")
    #s = main2(isTraining=False, area="computer_science")
    #print(s)
