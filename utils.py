from nltk.stem import WordNetLemmatizer
lemmatize = WordNetLemmatizer().lemmatize

def normalize_name_for_space_name(name):
    tmp = "_".join(name.split(" "))
    name = normalize_name_for_querying_vector_model(tmp)
    return name

def normalize_name_for_querying_vector_model(name):
    tmp = name.split('_')
    for i in range(len(tmp)):
        tmp[i] = lemmatize(tmp[i])
    name = '_'.join(tmp)
    return name

