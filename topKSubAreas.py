from model import originMethod, mlpMethod, rnnMethod
import math
from utils import getCandidateSet, normalize_display_name, normalize_name_for_space_name

class TopKSubAreas(object):
    def __init__(self, area="machine_learning", context="computer_science",
                 k=15, kp=3, threshold = 0.0, weighting_mode=0, compute_mode=0, method="origin", has_parents=False, has_children=True):
        self.area = normalize_name_for_space_name(area.lower())
        self.context = normalize_name_for_space_name(context.lower())
        self.k = k #top k childrens
        self.kp = kp #top kp parents
        self.threshold = threshold
        self.weighting_mode = weighting_mode
        self.compute_mode = compute_mode

        self.candidateDepth = 3
        self.candidateSet = None #[set, set, set, ...]
        self.candidateWeight = None
        self.rankResult = None
        self.has_children = has_children

        self.has_parents = has_parents
        self.candidateParentSet = None
        self.rankParentResult = None

        self.method = method

    def set_k(self, k):
        self.k = k
    def set_kp(self, kp):
        self.kp = kp
    def set_area(self, area):
        self.area = area
    def set_context(self, context):
        self.context = context

    def _weight_for_depth(self, d):
        # compute the d-th layer's weight

        if self.weighting_mode == 1:
            return math.exp(4 - d)
        else: # this means 'root takes all weight'
            if d == 0:
                return 1
            else:
                return 0

    def _getCandidate(self):
        self.candidateSet = getCandidateSet(area=self.area, depth=self.candidateDepth, compute_mode=self.compute_mode)

        self.candidateWeight = []
        for d in range(len(self.candidateSet)):
            weight_for_depth_d = self._weight_for_depth(d)
            if weight_for_depth_d <= 0:
                break
            self.candidateWeight.append(weight_for_depth_d)

        if self.has_parents:
            self.candidateParentSet = getCandidateSet(area=self.area, depth=self.candidateDepth, compute_mode=3)


    def _originMethod(self, isParent = False):
        if not isParent:
            assert self.candidateWeight is not None #assure weight is not none
            ranked_scores = originMethod(self.candidateWeight, self.candidateSet, self.k, self.context)
        else:
            #parent
            ranked_scores = originMethod(self.candidateWeight, self.candidateParentSet, self.kp, self.context)
        return ranked_scores


    def _mlp(self):
        ranked_scores = mlpMethod(self.candidateSet, self.k)
        return ranked_scores

    def _rnn(self):
        ranked_score = rnnMethod(self.area)
        return ranked_score

    def _getTopK(self):
        self.rankResult = []
        #parent
        self.rankParentResult = []
        self._getCandidate()

        if self.has_children:

            if self.method == "origin":
                rank = self._originMethod()
            elif self.method == "mlp":
                rank = self._mlp()
            elif self.method == "rnn":
                rank = self._rnn()
            else:
                rank = []

            if self.method != "rnn":
                for oneItem in rank:
                    if oneItem[1] >= self.threshold:
                        displayname = normalize_display_name(oneItem[0])
                        self.rankResult.append(displayname)
                    else:
                        break
            else:
                self.rankResult = [normalize_display_name(i) for i in rank]

        if self.has_parents:
            rank = self._originMethod(isParent=True)
            for oneItem in rank:
                if oneItem[1] >= self.threshold:
                    displayname = normalize_display_name(oneItem[0])
                    self.rankParentResult.append(displayname)
                else:
                    break


    def getResult(self):
        try:
            self._getTopK()
        except:
            #if we get expetion, return empty
            self.rankResult = []
            self.rankParentResult = []
        return self.rankResult, self.rankParentResult

