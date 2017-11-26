from topKSubAreas import TopKSubAreas
from globalVar import taxonomy, w2v_model
from utils import getCandidateMap

import pickle
import json
import os
import numpy as np

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
        ok = True
        #don't fix the k
        """
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


if __name__ == "__main__":
    generate_data(k=50, threshold=0.5)
