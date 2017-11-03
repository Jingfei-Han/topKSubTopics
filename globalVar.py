import logging
from gensim.models import Word2Vec
import pickle

import os
from keras.models import load_model

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

global taxonomy, w2v_model, mag, ccs
global mlp_model

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

def get_data_from_pickle(filename):
    with open(filename, "rb") as f:
        fos = pickle.load(f)
    return fos


taxonomy = None
w2v_model = None
mag = None
mlp_model = None

taxonomy_infile = "/dev/shm/a/wiki_taxonomy_lemmatized.pkl"
vector_model_infile = "/dev/shm/a/wiki_text_20161201_1to4_200d.model"
mag_infile = "/dev/shm/a/mag2.pkl"
ccs_infile = "/dev/shm/a/acm_ccs.pkl"

mlp_infile = "data/mlp_model.h5"

taxonomy = load_taxonomy(taxonomy_infile)
w2v_model = load_vector_model(vector_model_infile)
mag = get_data_from_pickle(mag_infile)
ccs = get_data_from_pickle(ccs_infile)

if os.path.exists(mlp_infile):
    mlp_model = load_model(mlp_infile)
