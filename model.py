from globalVar import w2v_model
import logging
import mlp
import rnn
from globalVar import rnn_model, expert_filter

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

"""
input: candidateWeight, candidateSet, k, context
    candidateWeight: [float, float, ...] , n = the number of layers
    candidateSet: [set, set, ...], n = the number of layers
    k: int
    context: str
output: ranked_scores
    ranked_scores = [["A", 0.9], ["B", 0.84], ["C", 0.7], ...] 
    
    before sort operation:
        scores = {"A":0.9, "B":0.8, "C":0.7}

"""

def originMethod(candidateWeight, candidateSet, k, context, isParent=False):
    candidateSet = [list(i) for i in candidateSet] # change set to list

    scores = {}
    area = candidateSet[0][0]
    for d in range(len(candidateSet) - 1): # excluding root
        tmpcats = candidateSet[d + 1]
        for c in tmpcats:
            if c != area:
                tmp_score = 0
                for d in range(len(candidateWeight)):
                    tmpcats2 = candidateSet[d]
                    for c2 in tmpcats2:
                        try:
                            # c: sub-area, c2: area, e.g. c:deep_learning; c2:machine_learning
                            tmp_score += w2v_model.n_similarity([c], [c2, context]) * candidateWeight[d]
                        except:
                            try:
                                tmp_score += w2v_model.n_similarity([c], [c2]) * candidateWeight[d]
                            except Exception as e:
                                #logging.info(e)
                                pass

                scores[c] = tmp_score
            else:
                pass
    """
    # special need , database's subarea includes xml
    if area == "database":
        scores["xml"] = 0.71
    """
    if not isParent:
        #filter using expert annotation
        if area in expert_filter:
            tmp_sm = 0.99
            for i in expert_filter[area]:
                scores[i] = tmp_sm
                tmp_sm -= 0.001

    ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    #print(ranked_scores)
    ranked_scores = ranked_scores[:k]
    return ranked_scores

def mlpMethod(candidateSet, k):
    mlp_model = mlp.MLP(isReadModel=True, isRun=True)
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

def rnnMethod(area):
    global rnn_model
    if rnn_model is None:
        rnn_model = rnn.RNN()

    ranked_scores = rnn_model.train(False, area)
    ranked_scores = ranked_scores[1:] #delete the first element
    return ranked_scores


