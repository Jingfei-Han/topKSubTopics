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



taxonomy = None
parent_taxonomy = None
w2v_model = None
mag = None
mlp_model = None
rnn_model = None

taxonomy_infile = "/dev/shm/a/wiki_taxonomy_lemmatized.pkl"
parent_taxonomy_infile = "data/parent_taxonomy.pkl"
vector_model_infile = "/dev/shm/a/wiki_text_20161201_1to4_200d.model"
mag_infile = "/dev/shm/a/mag2.pkl"
ccs_infile = "/dev/shm/a/acm_ccs.pkl"

mlp_infile = "data/mlp_model.h5"

taxonomy = load_taxonomy(taxonomy_infile)
w2v_model = load_vector_model(vector_model_infile)
mag = get_data_from_pickle(mag_infile)
ccs = get_data_from_pickle(ccs_infile)
parent_taxonomy = load_parent_taxonomy(parent_taxonomy_infile)

if os.path.exists(mlp_infile):
    mlp_model = load_model(mlp_infile)

rnn_model = rnn.RNN()
