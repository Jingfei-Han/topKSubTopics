from nltk.stem import WordNetLemmatizer
lemmatize = WordNetLemmatizer().lemmatize
from globalVar import taxonomy, mag, ccs
from collections import defaultdict
import os

def normalize_name_for_space_name(name):
    # e.g.: "machine learning algorithms" --> "machine_learning_agorithm"
    tmp = "_".join(name.split(" "))
    name = normalize_name_for_querying_vector_model(tmp)
    return name

def normalize_name_for_querying_vector_model(name):
    # e.g.: "machines_learning" --> "machine_learning"
    tmp = name.split('_')
    for i in range(len(tmp)):
        tmp[i] = lemmatize(tmp[i])
    name = '_'.join(tmp)
    return name

#subcats
def subcats_not_more_than_depth(area, depth):
    subcats = [set([area])]
    for i in range(depth):
        tmpcats = set()
        for j in subcats[-1]:
            if j in taxonomy:
                tmpcats.update(taxonomy[j]['subcats'])
        subcats.append(tmpcats)
    return subcats

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
    # the number of the mode: 4
    if compute_mode >= 4:
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
    elif compute_mode ==2:
        #mag
        candidateSet = get_subcats(area, mag)
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