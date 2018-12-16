import os,sys,inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from util import create_sets
import gensim 
from gensim.models import Word2Vec
import datasets.all_songs_full_length

all_songs_list = datasets.all_songs_full_length.my_list
song_tokens = create_sets(1, all_songs_list, select_artists_only=False)[0]
print 'creating SG model for n=1 and mc=1'
model1 = gensim.models.Word2Vec(song_tokens, sg=1, min_count=1)
print 'creating SG model for n=1 and mc=5'
model5 = gensim.models.Word2Vec(song_tokens, sg=1, min_count=5)
model1.save('all_songs_dataset_skip_gram_n=1_mc=1.model')
model5.save('all_songs_dataset_skip_gram_n=1_mc=5.model')