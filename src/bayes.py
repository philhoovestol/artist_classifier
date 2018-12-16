from util import *
from collections import defaultdict
from random import choice
import datasets.newDataset_25_NewArtists_1
from pprint import pprint

#for laplace smoothing; only applies to gram probs not artist probs
pseudocount = 0.05

#include probabilty of track being by artist independent of lyrics in probabilistic inference
include_artist_prob = False

#where probabilities will be stored
gram_counts = []
artist_counts = [0]*len(artists)

artist_correct = [0] * len(artists)
artist_totals = [0] * len(artists)
artist_false_positives = [0] * len(artists)

train_x, train_y, val_x, val_y, test_x, test_y = create_sets(n, datasets.newDataset_25_NewArtists_1.my_list)[1]

def tune_counts():
	print 'tuning counts'

	#initialize counts

	gram_counts = []
	artist_counts = [0]*len(artists)

	for i in range(len(artists)):
		gram_counts.append(defaultdict(int))

	#increment counts

	for i in range(len(train_x)):
		artist_count = gram_counts[train_y[i]]
		artist_counts[train_y[i]] += 1
		for gram in train_x[i]:
			value = train_x[i][gram]
			artist_count[gram] += value


	#laplace smooth

	grams = set()
	for x in train_x + val_x + test_x:
		for gram in x:
			grams.add(gram)
	for gram_count in gram_counts:
		for gram in grams:
			gram_count[gram] += pseudocount


	#normalize

	new_gram_counts = []
	for gram_count in gram_counts:
		new_gram_count = {}
		total = 0
		for gram in gram_count:
			total += gram_count[gram]
		for gram in gram_count:
			new_gram_count[gram] = gram_count[gram] * 1.0 / total
		new_gram_counts.append(new_gram_count)
	gram_counts = new_gram_counts

	total = sum(artist_counts)
	new_artist_counts = []
	for artist_count in artist_counts:
		new_artist_counts.append(1.0 * artist_count / total)
	artist_counts = new_artist_counts

	return artist_counts, gram_counts

#returns accuracy rate predicting y from x
def evaluatePredictor(x, y, print_inaccurate_pairs=False, update_correct_counts=False, update_false_positive_counts=False):
    randomized_prediction_count = 0
    incorrect_pairs = defaultdict(int)
    errors = 0
    for i in range(len(x)):
		x_grams = x[i]
		max_artist_index = -1
		max_artist_prob = float('-inf')
		for artist_index in range(len(artists)):
			prob =  1.0
			if include_artist_prob:
				prob *= artist_counts[artist_index]
			artist_gram_count = gram_counts[artist_index]
			for gram in x_grams:
				gram_prob = artist_gram_count[gram]
				prob *= gram_prob
			if prob > max_artist_prob:
				max_artist_prob = prob
				max_artist_index = artist_index
		assert max_artist_index != -1
		if max_artist_prob == 0:
			randomized_prediction_count += 1
			max_artist_index = choice(range(len(artists)))
		if y[i] != max_artist_index:
			if update_false_positive_counts:
				artist_false_positives[max_artist_index] += 1
			errors += 1
			pair = [artists[max_artist_index], artists[y[i]]]
			pair.sort()
			incorrect_pairs[tuple(pair)] += 1
		else:
			if update_correct_counts:
				artist_correct[max_artist_index] += 1
		if update_correct_counts:
			artist_totals[y[i]] += 1
    if print_inaccurate_pairs:
        d_view = [ (v,k) for k,v in incorrect_pairs.iteritems() ]
        d_view.sort(reverse=True) # natively sort tuples by first element
        print 'pairs of artists and number of times they were mixed up with each other'
        pprint(d_view)
    #print str(randomized_prediction_count) + ' predictions were made at random'
    return (1.0*errors/len(x))
		#else:
			#print 'correctly predicted artist '+str(artists[max_artist_index])+' with probability '+str(max_artist_prob)

artist_counts, gram_counts = tune_counts()
print 'training error: '+str(evaluatePredictor(train_x, train_y))
print 'dev error: '+str(evaluatePredictor(val_x, val_y))
print 'test error: '+str(evaluatePredictor(test_x, test_y, print_incorrect_pairs, print_correct_counts, print_false_positive_rates))

if print_correct_counts:
	print 'artists in order of decreasing recall rate:'
	sort_this = []
	for i in range(len(artists)):
		sort_this.append((100.0*artist_correct[i]/artist_totals[i],artists[i]))
	sort_this.sort(reverse=True)
	for ranked_artist in sort_this:
		print ranked_artist[1]+': '+str(ranked_artist[0])+'%'

if print_false_positive_rates:
	print 'artists in order of decreasing number of false positives (where that artist was output of the inaccurate prediction):'
	sort_this = []
	for i in range(len(artists)):
		sort_this.append((artist_false_positives[i],artists[i]))
	sort_this.sort(reverse=True)
	for ranked_artist in sort_this:
		print ranked_artist[1]+': '+str(ranked_artist[0])


