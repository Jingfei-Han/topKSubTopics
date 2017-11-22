import numpy as np
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import time
import json
import pickle

#DATA_PATH = "data/rnn_data.txt"
DATA_PATH = "data/rnn_data_sample.txt"
HIDDEN_SIZE = 200
EMBED_SIZE = 200
BATCH_SIZE = 32
NUM_STEPS = 60
SKIP_STEP = 10
TEMPRATURE = 0.7
LR = 0.001
LEN_GENERATED = 300

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
    tmp = i+1
    topic2index["EOS"] = tmp+1
    index2topic[tmp+1] = "EOS"
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

def create_rnn(seq, hidden_size=HIDDEN_SIZE):
    #cell = tf.contrib.rnn.GRUCell(hidden_size)
    cell = tf.nn.rnn_cell.GRUCell(200)
    in_state = tf.placeholder_with_default(
                cell.zero_state(tf.shape(seq)[0], tf.float32), [None, hidden_size])
    length = tf.reduce_sum(tf.reduce_max(tf.sign(seq), 2), 1) + 1
    output, out_state = tf.nn.dynamic_rnn(cell, seq, length, in_state)
    return output, in_state, out_state

def create_model(seq, temp, i2t, hidden=HIDDEN_SIZE):
    one_hot = tf.one_hot(seq, len(i2t))
    embed_matrix = tf.Variable(tf.constant(0.0, shape=[len(i2t), EMBED_SIZE]), trainable=False, name="embed_matrix")
    embedding_placehoder = tf.placeholder(dtype=tf.float32, shape=[len(i2t), EMBED_SIZE])
    embedding_init = embed_matrix.assign(embedding_placehoder)
    seq = tf.nn.embedding_lookup(embed_matrix, seq, name="embed")
    output, in_state, out_state = create_rnn(seq, hidden)
    logits = tf.contrib.layers.fully_connected(output, len(i2t), None)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, :-1], labels=one_hot[:, 1:]))

    y_hat = tf.nn.softmax(logits)
    sample = tf.argmax(y_hat, 0)
    #sample = tf.multinomial(tf.exp(logits[:, -1]/temp), 1)[:, 0]

    return loss, sample, in_state, out_state, embedding_init, embedding_placehoder

def training(t2i, i2t, seq, loss, optimizer, global_step, temp, sample, in_state, out_state, embedding_init, embedding_placeholder, embedding):
    saver = tf.train.Saver()
    start = time.time()
    with tf.Session() as sess:
        writer= tf.summary.FileWriter("graphs/rnn", sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(embedding_init, feed_dict={embedding_placeholder:embedding})

        ckpt = tf.train.get_checkpoint_state(os.path.dirname("checkpoints/rnn/checkpoints"))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        iteration = global_step.eval()
        for batch in read_batch(read_data(DATA_PATH, t2i=t2i)):
            batch_loss, _ = sess.run([loss, optimizer], {seq: batch})
            print(iteration, batch_loss)

            if (iteration + 1) % SKIP_STEP == 0:
                print("Iter {}. \n LOSS {}. Time{}".format(iteration, batch_loss, time.time() - start))
                online_inference(sess, t2i, i2t, seq, sample, temp, in_state, out_state)
                start = time.time()
                saver.save(sess, "checkpoints/rnn/result", iteration)
            iteration += 1

def online_inference(sess, t2i, i2t, seq, sample, temp, in_state, out_state, seed="machine_learning"):
    #sentence = [seed]
    sentence = ["machine_learning"]
    state = None
    for _ in range(LEN_GENERATED):
        batch = [words_encode([sentence[-1]], t2i=t2i)]
        feed = {seq:batch, temp:TEMPRATURE}

        if state is not None:
            feed.update({in_state: state})
        index, state = sess.run([sample, out_state], feed)
        print(np.max(index))
        words = words_decode(index, i2t)
        if words[0] == "EOS":
            sentence.extend(words)
            break
        elif words[0] == "0":
            break
        else:
            sentence.extend(words)
    print(sentence)

def main():
    with open("data/cs_candidate.json", "r") as f:
        candidate = json.load(f)
    with open("data/cs_candidate_emb.pkl", "rb") as f:
        embedding = pickle.load(f)
    t2i, i2t = getCandidateMap(candidate)
    t2i["0"] = 0
    i2t[0] = "0"
    seq = tf.placeholder(tf.int32, [None, None])
    temp = tf.placeholder(tf.float32)
    loss, sample, in_state, out_state, embedding_init, embedding_placeholder = create_model(seq, temp, i2t)
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
    optimizer = tf.train.GradientDescentOptimizer(LR).minimize(loss, global_step=global_step)
    make_dir("checkpoints")
    make_dir("checkpoints/rnn")
    return t2i, i2t, seq, loss, optimizer, global_step, temp, sample, in_state, out_state, embedding_init, embedding_placeholder, embedding
    #training(t2i, i2t, seq, loss, optimizer, global_step, temp, sample, in_state, out_state, embedding_init, embedding_placeholder, embedding)

if __name__ == "__main__":
    main()
