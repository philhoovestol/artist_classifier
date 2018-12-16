from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import datasets.all_Songs_Dataset_2
import datasets.all_songs_full_length
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm

#implementation taken from https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5 and adjusted for the scheme of our problem

all_songs_list = datasets.all_songs_full_length.my_list
songs = []
artists = ['beyonce-knowles', '50-cent', 'eazy-e', 'casey-veggies', 'fetty-wap', 'flatbush-zombies', 'bas', 'frank-ocean', 'grandmaster-flash', 'childish-gambino', 'clipse', 'big-l', 'aloe-blacc', 'eminem', 'future', 'flobots', 'david-banner', '2-chainz', 'drake', 'big-sean', 'dr-dre', 'earl-sweatshirt', 'chance-the-rapper', 'common', 'asap-rocky']

lyrics_to_index = {}

for i, raw_song in enumerate(all_songs_list):
	artist = raw_song.split(" ")[0]
	lyrics = (" ").join(raw_song.split(" ")[1:])
	songs.append(lyrics)
	if artist in artists:
		lyrics_to_index[lyrics] = i

tokenizer = TreebankWordTokenizer()
print 'tagging data'
tagged_data = [TaggedDocument(words=tokenizer.tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(songs)]

max_epochs = 100
alpha = 0.025

model = Doc2Vec(alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
print 'building model from data'

model.build_vocab(tagged_data)

print 'training model'
for epoch in tqdm(range(max_epochs)):
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")

np.save('lyrics_to_index.npy', lyrics_to_index)
print('lyrics_to_index saved')