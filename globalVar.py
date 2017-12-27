import logging
from gensim.models import Word2Vec
import pickle

import os
from keras.models import load_model
import rnn
from collections import defaultdict

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

global taxonomy, w2v_model, mag, ccs
global parent_taxonomy
global mlp_model
global rnn_model

def load_vector_model(vector_model):
    logging.info('loading vector model from {}...'.format(vector_model))
    w2v_model = Word2Vec.load(vector_model)
    logging.info('model loaded.')
    return w2v_model

def load_taxonomy(taxonomy_infile):
    logging.info('loading taxonomy from {}...'.format(taxonomy_infile))
    with open(taxonomy_infile, 'rb') as f:
        taxonomy = pickle.load(f)
    logging.info('taxonomy loaded.')
    return taxonomy

def load_parent_taxonomy(parent_taxonomy_infile):
    if os.path.exists(parent_taxonomy_infile):
        logging.info('loading parent taxonomy from {}...'.format(parent_taxonomy_infile))
        with open(parent_taxonomy_infile, 'rb') as f:
            parent_taxonomy = pickle.load(f)
        logging.info('parent taxonomy loaded.')
    else:
        logging.info('start to get parent taxonomy...')
        parent_taxonomy = parentTaxonomy()
    return parent_taxonomy

def get_data_from_pickle(filename):
    with open(filename, "rb") as f:
        fos = pickle.load(f)
    return fos

def mergeAllTaxonomy():
    new_tax = dict()
    # taxonomy + mag + ccs --> new hierarchy
    for i in taxonomy.keys():
        new_tax[i] = taxonomy[i]['subcats']
        if i in mag:
            new_tax[i].update(mag[i])
        if i in ccs:
            new_tax[i].update(ccs[i])
    return new_tax

def parentTaxonomy():
    new_tax = mergeAllTaxonomy()
    parent_tax = defaultdict(set)
    for i in new_tax.keys():
        for j in new_tax[i]:
            parent_tax[j].add(i)
    with open("data/parent_taxonomy.pkl", "wb") as f:
        pickle.dump(parent_tax, f)
    print("save parent taxonomy successfully!")
    return parent_tax

def preprocessTaxonomy(taxonomy, start_area = "scientific_discipline"):
    """
    :param taxonomy:  raw taxonomy
    :param start_area: only consider topic from start_area
    :return: new_taxonomy which can replace taxonomy
    """
    new_tax = {}
    A = set() # raw set
    new_tax[start_area] = taxonomy[start_area].copy()

    #adjust
    new_tax[start_area]["subcats"].remove("vietnam_war_patrol_vessel_of_the_united_state")
    new_tax[start_area]["subcats"].add("natural_science")

    #B = A.copy() # every iteration set
    B = set()
    B.add(start_area)
    cnt = 0

    while len(B) != len(A):
        C = B - A
        B_tmp = B.copy()
        for i in C:
            try:
                if len(taxonomy[i]["subcats"]) > 100:
                    # chemistry: 72 ge
                    # delete the number of subcats greater than 40
                    B.remove(i)
                    B_tmp.remove(i)
                    continue
                new_tax[i] = taxonomy[i]
                A.add(i)
                for j in taxonomy[i]["subcats"]:
                    B.add(j) # add all subcats in B, in order to iterate it next time.
            except Exception as e:
                print("ERROR 1: {}".format(e))
        assert len(B_tmp) == len(A)
        print("Epoch {}: len_A = {}, len_B = {}".format(cnt, len(A), len(B)))
        cnt += 1

    print("delete invalid children...")
    for j in new_tax.keys():
        tmp = []
        for i in new_tax[j]["subcats"]:
            if i not  in new_tax.keys():
                tmp.append(i)
        for i in tmp:
            new_tax[j]["subcats"].remove(i)
    print("finish!")

    return new_tax


taxonomy = None
parent_taxonomy = None
w2v_model = None
mag = None
mlp_model = None
rnn_model = None
"""
taxonomy_infile = "/dev/shm/b/wiki_taxonomy_lemmatized.pkl" #pickle 2
parent_taxonomy_infile = "data/parent_taxonomy.pkl"
vector_model_infile = "/dev/shm/a/wiki_text_20161201_1to4_200d.model"
mag_infile = "/dev/shm/b/mag2.pkl"
ccs_infile = "/dev/shm/b/acm_ccs.pkl"
"""

taxonomy_infile = "/dev/shm/a/wiki_taxonomy_lemmatized.pkl" #pickle 3
parent_taxonomy_infile = "data/parent_taxonomy.pkl"
vector_model_infile = "/dev/shm/a/wiki_text_20161201_1to4_200d.model"
mag_infile = "/dev/shm/a/mag2.pkl"
ccs_infile = "/dev/shm/a/acm_ccs.pkl"
mlp_infile = "data/mlp_model.h5"

taxonomy = load_taxonomy(taxonomy_infile)
taxonomy = preprocessTaxonomy(taxonomy)
w2v_model = load_vector_model(vector_model_infile)
mag = get_data_from_pickle(mag_infile)
ccs = get_data_from_pickle(ccs_infile)
parent_taxonomy = load_parent_taxonomy(parent_taxonomy_infile)

if os.path.exists(mlp_infile):
    mlp_model = load_model(mlp_infile)

rnn_model = rnn.RNN()
