from topKSubAreas import TopKSubAreas
from globalVar import taxonomy, w2v_model
from utils import getCandidateMap

import pickle
import json
import os
import numpy as np
import copy

"""
def main():
    print("read data...")
    with open("data/rnn_data", "wb") as f:
        data_X, data_Y = pickle.load(f)
    print("read finished.")

    learning_rate = 0.001
"""

def getCandidateList(path="data/cs_candidate.json"):
    with open(path, "r") as f:
        data = json.load(f)
    assert type(data) == list
    tmp = np.asarray(data)
    np.random.shuffle(tmp) # shuffle
    data = list(tmp)

    emb = []
    emb.append(list(np.zeros(200, dtype=np.float32))) # "EOS" embedding

    for i in range(len(data)-1, -1, -1):
        cur_word= data[i]
        try:
            emb.append(list(w2v_model[cur_word]))
        except:
            del data[i]

    emb.append(list(np.zeros(200, dtype=np.float32))) # 0 is padding
    emb = emb[::-1]
    with open("data/cs_candidate_emb.pkl", "wb") as f:
        pickle.dump(emb, f)
    #update candidate
    with open("data/cs_candidate.json", "w") as f:
        json.dump(data, f, indent=4)

    return data





def generate_data(k = 10, threshold = 0.0, hasCandidate = True):
    """
    :param k: the length of output, i.e. top k sub-area
    :param threshold: only consider sub-areas whose confidence >= threshold
    :param hasCandidate: if True, we use our candidate data, otherwise use taxonomy
    :return: train_X, train_Y, test_X, test_Y
    """


    print("---------------------------------------")
    print("alpha= ", k, " threshold= ", threshold)
    print("start to generate data...")

    candidate = []

    if hasCandidate:
        candidate = getCandidateList("data/cs_candidate.json")
    else:
        for i in taxonomy.keys():
            candidate.append(i)

    t2i, i2t = getCandidateMap(candidate)

    cnt = 0
    #contextList = ["physic", "math", "chemistry", "biology", "engineering", "biochemistry", "geography",
    #               "linguistics", "philosophy", "computer_science"]
    context = "computer_science"

    topk = TopKSubAreas(area="deep_learning", context=context, k=k, threshold=threshold, method = "origin")

    output = []

    for area in candidate:
        topk.set_area(area)
        ranked_scores = topk.getResult()
        """
        #fix k
        ok = True
        if len(ranked_scores) == k:
            for i in ranked_scores:
                if i[0] not in candidate:
                    ok = False
                    break
            if ok:
                cnt += 1
                if cnt % 100 == 0:
                    print("current samples: ", cnt)

                tmpSentence = [t2i[area]]
                for oneItem in ranked_scores:
                    tmpSentence.append(t2i[oneItem[0]])
                tmpSentence.append(t2i["EOS"])

                output.append(tmpSentence)
        """
        #don't fix the k
        for i in range(len(ranked_scores)-1, -1, -1):
            if ranked_scores[i][0] not in candidate:
                del ranked_scores[i]
        cnt += 1
        if cnt % 100 == 0:
            print("current samples: ", cnt)

        tmpSentence = [area]
        for oneItem in ranked_scores:
            tmpSentence.append(oneItem[0])
        tmpSentence.append("EOS")

        output.append(tmpSentence)

    print("Total samples: ", cnt)
    print("---------------------------------------")
    print("save data...")

    with open("data/rnn_data.txt", "w") as f:
        json.dump(output, f)

    print("save finished!")

def generate_data2(k, threshold):
    a = TopKSubAreas(k=k, threshold=threshold, context="", has_similiarity=True)
    sentences = {}
    cnt = 0
    for area in taxonomy:
        a.set_area(area)
        ch, _ = a.getResult()
        if len(ch) > 0:
            sentences[area] = ch
        cnt += 1
        if cnt % 10000 == 0:
            print("Now finish: {}".format(cnt))
    print("generate finished!")
    print("save sentences")
    with open("data/every_tax_result.json", "w") as f:
        json.dump(sentences, f, indent=4)
    print("save finished!")

def preprocess_data(path="data/every_tax_result.json", count = 10):
    if not os.path.exists(path):
        generate_data2(k=15, threshold=0.1)

    print("we have origin data, so we will preprocess it.")
    with open(path, "r") as f:
        data = json.load(f) #data is a dict

    #generate vocabulary:
    print("-----------------------------------------------------")
    print("generate vocabulary...")
    vocab = set()
    for area in data.keys():
        vocab.add(area)
        for i in data[area]:
            vocab.add(i[0])
    vocab = list(vocab)
    np.random.shuffle(vocab) # vocabulary table
    print("vocabulary size is: {}".format(len(vocab)))
    emb = []
    for i in vocab:
        emb.append(list(w2v_model[i]))

    #add EOS
    vocab.append("EOS")
    emb.append(list(np.random.randn(200)))

    #save data
    with open("data/vocab_table.json", "w") as f:
        json.dump(vocab, f, indent=4)

    with open("data/vocab_emb.pkl", "wb") as f:
        pickle.dump(emb, f) #[[],[],[]]

    print("generate vocabulary table and embedding table finished!")

    print("-----------------------------------------------------")
    print("generate random samples...")

    def change_value(res, rand):
        assert len(res) == len(rand)
        tmp = copy.deepcopy(res)
        for i in range(len(tmp)):
            tmp[i][1] += rand[i]
        tmp = sorted(dict(tmp).items(), key=lambda x:x[-1], reverse=True) #[("A",0.4), ("B", 0.1)]
        return tmp

    def generate_sentence(area, res):
        tmp = [area]
        for i in res:
            tmp.append(i[0])
        tmp.append("EOS")
        return tmp

    sigma = 0.1 # normal distribution: N(0, 0.1^2)
    sentences = []
    for area in data.keys():
        res = data[area] #res is: [ ["A", 0.6], ["B", 0.4]]
        len_res = len(res)
        cnt = min(len_res, count)
        rand = np.zeros(len_res) #random value
        for _ in range(cnt):
            tmp = change_value(res, rand)
            sentences.append(generate_sentence(area, tmp))
            rand = sigma * np.random.randn(len_res)

    np.random.shuffle(sentences)
    with open("data/rnn_samples_new.json", "w") as f:
        json.dump(sentences, f, indent=4)
    print("generate random samples finished!")

    return 0




if __name__ == "__main__":
    #generate_data2(k=15, threshold=0.1)
    preprocess_data(count=10,path="data/every_tax_result.json")
