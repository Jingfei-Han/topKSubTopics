from globalVar import taxonomy, mag
from model import originMethod, mlpMethod
import math
from collections import defaultdict

class TopKSubAreas(object):
    def __init__(self, area="deep_learning", context="computer_science",
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

        self.method = method.strip().lower()

    def _weight_for_depth(self, d):
        #d = self.candidateDepth

        if self.weighting_mode == 1:
            return math.exp(4 - d)
        else: # this means 'root takes all weight'
            if d == 0:
                return 1
            else:
                return 0

    def _subcats_not_more_than_depth(self):
        subcats = [set([self.area])]
        for i in range(self.candidateDepth):
            tmpcats = set()
            for j in subcats[-1]:
                if j in taxonomy:
                    tmpcats.update(taxonomy[j]['subcats'])
            subcats.append(tmpcats)
        return subcats

    def _get_mag_subcats(self):
        subcats = [set([self.area])]
        dic = defaultdict(set)
        for i in mag.keys():
            if mag[i]['par_name'] == self.area:
                dic[mag[i]['ch_label']].add(mag[i]['ch_name'])
        for i in dic.keys():
            subcats.append(dic[i])
        return subcats

    #subcats_not_more_than_depth
    # The function named get_mag_subcats returns subcats too
    def _getCandidateSet(self):
        if self.compute_mode == 0:
            #origin:
            self.candidateSet = self._subcats_not_more_than_depth()
        elif self.compute_mode == 1:
            #mag
            self.candidateSet = self._get_mag_subcats()
        else:
            #merge origin and mag
            subcat1 = self._get_mag_subcats()
            subcat2 = self._subcats_not_more_than_depth()
            len_1 = len(subcat1)
            len_2 = len(subcat2)
            subcats = []
            for i in range(min(len_1, len_2)):
                subcats.append((subcat1[i] | subcat2[i]))
            for i in range(min(len_1, len_2), max(len_1, len_2)):
                if len_1 > len_2:
                    # add subcat1
                    subcats.append(subcat1[i])
                else:
                    # add subcat2
                    subcats.append(subcat2[i])
            self.candidateSet = subcats

    def _getCandidateWeight(self):
        self.candidateWeight = []
        for d in range(len(self.candidateSet)):
            weight_for_depth_d = self._weight_for_depth(d)
            if weight_for_depth_d <= 0:
                break
            self.candidateWeight.append(weight_for_depth_d)

    def _originMethod(self):
        self._getCandidateWeight()
        assert self.candidateWeight is not None #assure weight is not none
        ranked_scores = originMethod(self.candidateWeight, self.candidateSet, self.k, self.context)
        return ranked_scores

    def _mlp(self):
        ranked_scores = mlpMethod(self.candidateSet, self.k)
        return ranked_scores

    def _getTopK(self):
        self._getCandidateSet()
        if self.method == "origin":
            self.rankResult = self._originMethod()
        elif self.method == "mlp":
            self.rankResult = self._mlp()
        else:
            pass

    def getResult(self):
        self._getTopK()
        return self.rankResult

