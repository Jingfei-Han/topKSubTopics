from model import originMethod, mlpMethod
import math
from utils import getCandidateSet

class TopKSubAreas(object):
    def __init__(self, area="machine_learning", context="computer_science",
                 k=15, weighting_mode=0, compute_mode=0, method="origin"):
        self.area = area
        self.context = context
        self.k = k
        self.weighting_mode = weighting_mode
        self.compute_mode = compute_mode

        self.candidateDepth = 3
        self.candidateSet = None #[set, set, set, ...]
        self.candidateWeight = None
        self.rankResult = None

        self.method = method

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

    def _originMethod(self):
        assert self.candidateWeight is not None #assure weight is not none
        ranked_scores = originMethod(self.candidateWeight, self.candidateSet, self.k, self.context)
        return ranked_scores

    def _mlp(self):
        ranked_scores = mlpMethod(self.candidateSet, self.k)
        return ranked_scores

    def _getTopK(self):
        self._getCandidate()
        if self.method == "origin":
            self.rankResult = self._originMethod()
        elif self.method == "mlp":
            self.rankResult = self._mlp()
        else:
            self.randResult = []
            pass

    def getResult(self):
        self._getTopK()
        return self.rankResult

