from nltk.stem import WordNetLemmatizer
lemmatize = WordNetLemmatizer().lemmatize
from globalVar import taxonomy, mag, ccs, parent_taxonomy
from collections import defaultdict, OrderedDict
import os
import copy

"""
class Node(object):
    def __init__(self, name, children=[], parents=[]):
        self.name = name

        self.setChildren(children)
        self.setParents(parents)

    def setChildren(self, childrenList):
        self.children = []
        for i in childrenList:
            self.children.append(Node(i))

    def setParents(self, parentsList):
        self.parents = []
        for i in parentsList:
            self.parents.append(Node(i))

class OrderedDefaultDict(OrderedDict, defaultdict):
    def __init__(self, default_factory=None, *args, **kwargs):
        super(OrderedDefaultDict, self).__init__(*args, **kwargs)
        self.default_factory = default_factory
"""

def normalize_name_for_space_name(name):
    # e.g.: "machine learning algorithms" --> "machine_learning_agorithm"
    tmp = "_".join(name.split())
    name = normalize_name_for_querying_vector_model(tmp)
    return name

def normalize_name_for_querying_vector_model(name):
    # e.g.: "machines_learning" --> "machine_learning"
    tmp = name.split('_')
    for i in range(len(tmp)):
        tmp[i] = lemmatize(tmp[i])
    name = '_'.join(tmp)
    return name

def normalize_display_name(name):
    #e.g. "machine_learning" --> "Machine Learning"
    tmp = name.split("_")
    return " ".join(i.title() for i in tmp)

#subcats
def subcats_not_more_than_depth(area, depth):
    subcats = [set([area])]
    for i in range(depth):
        tmpcats = set()
        for j in subcats[-1]:
            if j in taxonomy:
                tmpcats.update(taxonomy[j]['subcats'])
        if len(tmpcats) > 0:
            subcats.append(tmpcats)
        else:
            break
    return subcats

def parents_not_more_than_depth(area, depth):
    parents = [set([area])]
    for i in range(depth):
        tmpcats = set()
        for j in parents[-1]:
            if j in parent_taxonomy:
                tmpcats.update(parent_taxonomy[j])
        if len(tmpcats) > 0:
            parents.append(tmpcats)
        else:
            break
    return parents

def get_subcats(area, data):
    #just two layers
    subcats = [set([area])]
    dic = defaultdict(set)
    for i in data.keys():
        if i == area:
            subcats.append(set(data[i]))
    return subcats

#subcats_not_more_than_depth
# The function named get_mag_subcats returns subcats too
def mergeTwoSet(subcat1, subcat2):

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
    return subcats

def getCandidateSet(area, compute_mode, depth):
    # the number of the mode: 5
    if compute_mode >= 5:
        compute_mode = 0
    if compute_mode == 0:
        #mixed taxonomy, mag and css
        subcat1 = get_subcats(area, mag)
        subcat2 = subcats_not_more_than_depth(area = area, depth = depth)
        subcat3 = get_subcats(area, ccs)

        subcats = mergeTwoSet(subcat1, subcat3)
        subcats = mergeTwoSet(subcats, subcat2)
        #subcats = mergeTwoSet(subcat1, subcat2)

        candidateSet = subcats
    elif compute_mode == 1:
        #origin:
        candidateSet = subcats_not_more_than_depth(area=area, depth=depth)
    elif compute_mode == 2:
        #mag
        candidateSet = get_subcats(area, mag)
    elif compute_mode == 3:
        #compute parent candidate set
        candidateSet = parents_not_more_than_depth(area=area, depth=1)
    else:
        candidateSet = get_subcats(area, ccs)

    return candidateSet

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

def make_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def get_hierarchy(data, root, name, n, depth, showTopics, c_or_p="children"):
    if len(root[c_or_p]) == 0:
        root[c_or_p] = []
        return root
    if n == depth:
        root[c_or_p] = []
        return root
    else:
        for i in range(len(root[c_or_p])-1, -1, -1):
            name = root[c_or_p][i]
            if name not in showTopics:
                showTopics.add(name)
            else:
                del root[c_or_p][i]
                continue
            tmp = data[name]
            try:
                tmp.pop("level")
            except:
                pass
            root_tmp = copy.deepcopy(tmp)
            root[c_or_p][i] = get_hierarchy(data, root_tmp, name, n+1, depth, showTopics, c_or_p=c_or_p)
    return root

def get_midDict(area_name, context, k, kp, weighting_mode, compute_mode, method, confidence, depth_of_tree, c_or_p="children"):

    c_or_p = c_or_p.lower().strip()
    if c_or_p != "children":
        c_or_p = "parents"
    cur_index, has_parents, has_children = (0, False, True) if c_or_p == "children" else (1, True, False)

    result = get_topk(area_name, context, k, kp, weighting_mode, compute_mode, method, confidence, has_parents, has_children)
    display_name = normalize_display_name(area_name)
    """save to an OrderedDefaultDict -- children"""
    dic = defaultdict(OrderedDict)

    dic[display_name]["name"] = display_name
    dic[display_name]["level"] = 0
    dic[display_name][c_or_p] = result[cur_index]

    for depth in range(1, depth_of_tree + 1):
        for topic in list(dic):
            if dic[topic]["level"] == depth - 1:
                for subtopic in dic[topic][c_or_p]:
                    # subtopic is a display name
                    if subtopic not in dic.keys():
                        tmp = get_topk(subtopic, context, k, kp, weighting_mode, compute_mode, method, confidence, has_parents=has_parents, has_children=has_children)
                        dic[subtopic]["name"] = subtopic
                        dic[subtopic]["level"] = depth
                        dic[subtopic][c_or_p] = tmp[cur_index]

    def preprocess(c_or_p):
        #preprocess: handle the same subtopic
        #Record childset
        childRecord = [display_name]
        for curParent in childRecord:
            if curParent in dic:
                dic[curParent][c_or_p] = dic[curParent][c_or_p][::-1] # reverse, in order to delete
                childList = dic[curParent][c_or_p]
                for i in range(len(childList)-1, -1, -1):
                    curChild = childList[i]
                    if curChild in childRecord:
                        del childList[i]
                    else:
                        childRecord.append(curChild)
                dic[curParent][c_or_p] = dic[curParent][c_or_p][::-1]
            else:
                continue
        return dic

    dic = preprocess(c_or_p)

    return dic

