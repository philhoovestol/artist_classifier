from random import shuffle
import gensim 
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
import numpy as np

n = 1
ignore = ['the', 'a', 'an']

artists = ['beyonce-knowles', '50-cent', 'eazy-e', 'casey-veggies', 'fetty-wap', 'flatbush-zombies', 'bas', 'frank-ocean', 'grandmaster-flash', 'childish-gambino', 'clipse', 'big-l', 'aloe-blacc', 'eminem', 'future', 'flobots', 'david-banner', '2-chainz', 'drake', 'big-sean', 'dr-dre', 'earl-sweatshirt', 'chance-the-rapper', 'common', 'asap-rocky']

print_correct_counts = False
print_incorrect_pairs = False
print_false_positive_rates = False

def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def featureExtractor(m, x):
    x_list = [i.lower() for i in x.split(' ')]
    result0 = []
    result = {}
    for iw in ignore:
        if iw in x_list:
            x_list.remove(iw)
    for i in range(len(x_list)-m+1):
        gram = tuple(x_list[i:i+m])
        gram_string = ' '.join(gram)
        result0.append(gram_string)
        if gram in result:
            result[gram] += 1
        else:
            result[gram] = 1
    return result0, result

def create_sets(m, original_list, use_doc_embedding_model=False, select_artists_only=True):
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []
    if select_artists_only:
        new_original_list = []
        for item in original_list:
            artist = item.split(" ")[0]
            if artist in artists:
                new_original_list.append(item)
        original_list = new_original_list
    #shuffle(original_list)
    count = 0 #use count to add every 5th example to test_examples (as opposed to train_examples)
    song_tokens = []
    if use_doc_embedding_model:
        lyrics_to_index = np.load("word_embedding_models/lyrics_to_index.npy").item()
        model = Doc2Vec.load("word_embedding_models/d2v.model")
    for example_string in original_list:
        artist = example_string.split(" ")[0]
        if artist not in artists:
            artists.append(artist)
        lyrics = (" ").join(example_string.split(" ")[1:])
        gram_tokens, song_as_dict = featureExtractor(m, lyrics)
        song_tokens.append(gram_tokens)
        if count < 7:
            if use_doc_embedding_model:
                train_x.append(model.docvecs[lyrics_to_index[lyrics]])
            else:
                train_x.append(song_as_dict)
            train_y.append(artists.index(artist))
        elif count == 7:
            if use_doc_embedding_model:
                val_x.append(model.docvecs[lyrics_to_index[lyrics]])
            else:
                val_x.append(song_as_dict)
            val_y.append(artists.index(artist))
        else:
            if use_doc_embedding_model:
                test_x.append(model.docvecs[lyrics_to_index[lyrics]])
            else:
                test_x.append(song_as_dict)
            test_y.append(artists.index(artist))
        count += 1
        if count > 9:
            count = 0

    return song_tokens, (train_x, train_y, val_x, val_y, test_x, test_y)

def getWord2VecDict():
    model = Word2Vec.load('word_embedding_models/all_songs_dataset_skip_gram_n=1_mc=1.model')
    WordVectorz=dict(zip(model.wv.index2word,model.wv.vectors))
    return WordVectorz

def songAsWord2VecAverage(model_dict, lyrics_as_dict):
    avg_this = []
    for gram, count in lyrics_as_dict.items():
        for i in range(count):
            gram_string = ' '.join(gram)
            if gram_string in model_dict:
                model_result = model_dict[gram_string]
                avg_this.append(model_result)
    if len(avg_this) == 0:
        print 'incoming 0 vector'
        return [0]*word_embedding_dim
    result = np.nanmean(avg_this, axis=0)
    return result

def x_sets_as_word_embeddings(train_x, val_x, test_x):
    print 'rewriting all x sets with word embedding model'
    model_dict = getWord2VecDict()
    new_train_x = []
    for x in train_x:
        new_train_x.append(songAsWord2VecAverage(model_dict, x))
    train_x = new_train_x

    new_val_x = []
    for x in val_x:
        new_val_x.append(songAsWord2VecAverage(model_dict, x))
    val_x = new_val_x

    new_test_x = []
    for x in test_x:
        new_test_x.append(songAsWord2VecAverage(model_dict, x))
    test_x = new_test_x

    return train_x, val_x, test_x
