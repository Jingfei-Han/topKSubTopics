from topKSubAreas import TopKSubAreas
from globalVar import taxonomy, w2v_model

import pickle




def generate_data(k = 10, threshold = 0.0):
    """
    :param k: the length of output, i.e. top k sub-area
    :param threshold: only consider sub-areas whose confidence >= threshold
    :return: train_X, train_Y, test_X, test_Y
    """

    print("---------------------------------------")
    print("alpha= ", k, " threshold= ", threshold)
    print("start to generate data...")
    cnt = 0
    #contextList = ["physic", "math", "chemistry", "biology", "engineering", "biochemistry", "geography",
    #               "linguistics", "philosophy", "computer_science"]
    context = "computer_science"

    input_embedding = []
    output_embedding = []

    input_name = []
    output_name = []

    topk = TopKSubAreas(area="deep_learning", context=context, k=k, threshold=threshold, method = "origin")

    for area in taxonomy.keys():
        topk.set_area(area)
        ranked_scores = topk.getResult()

        if len(ranked_scores) == k:
            cnt += 1
            if cnt % 100 == 0:
                print("current samples: ", cnt)
            #fix the length
            input_embedding.append(list(w2v_model[area]))
            input_name.append(area)
            tmp = []
            tmp_name = []
            for oneItem in ranked_scores:
                tmp.extend(list(w2v_model[oneItem[0]]))
                tmp_name.append(oneItem[0])
            output_embedding.append(tmp)
            output_name.append(tmp_name)

    print("Total samples: ", cnt)
    print("---------------------------------------")
    print("save data...")
    with open("data/rnn_data", "wb") as f:
        pickle.dump((input_embedding, output_embedding), f)
    with open("data/rnn_data_name", "wb") as f:
        pickle.dump((input_name, output_name), f)

    print("save finished!")


if __name__ == "__main__":
    generate_data(k=10, threshold=0.2)
