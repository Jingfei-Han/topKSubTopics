from globalVar import w2v_model
import logging
import mlp
from keras.models import load_model
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

def originMethod(candidateWeight, candidateSet, k, context):
    candidateSet = [list(i) for i in candidateSet] # change set to list

    scores = {}
    for d in range(len(candidateSet) - 1): # excluding root
        tmpcats = candidateSet[d + 1]
        for c in tmpcats:
            tmp_score = 0
            for d in range(len(candidateWeight)):
                tmpcats2 = candidateSet[d]
                for c2 in tmpcats2:
                    try:
                        tmp_score += w2v_model.n_similarity([c], [c2, context]) * candidateWeight[d]
                    except Exception as e:
                        logging.info(e)
            scores[c] = tmp_score
    ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    #print(ranked_scores)
    ranked_scores = ranked_scores[:k]
    return ranked_scores

def mlpMethod(candidateSet, k):
    filename = "mlp_model.h5"
    mlp_model = mlp.MLP(filename = filename, isReadModel=False, isRun=True)
    mlp_model.train_model(emb=200, epoch=20)
    candidateSet = [list(i) for i in candidateSet]
    area = candidateSet[0][0]
    context = "computer_science"
    candidate = []
    for d in range(len(candidateSet) - 1):
        tmpcats = candidateSet[d+1]
        for c in tmpcats:
            candidate.append(c)

    candidate = list(set(candidate))

    scores = mlp_model.predict(area, candidate, context)
    ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    #print(ranked_scores)
    ranked_scores = ranked_scores[:k]
    return ranked_scores

