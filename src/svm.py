from util import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import sklearn.feature_extraction
from matplotlib import pyplot as plt
from random import shuffle
from collections import defaultdict
import gensim 
from gensim.models import Word2Vec
import math
from tqdm import tqdm
import datasets.all_songs_full_length

numIters = 20
eta = 0.01
lamb = 0.4
weights_all = []

artist_correct = [0] * len(artists)
artist_totals = [0] * len(artists)
artist_false_positives = [0] * len(artists)

use_word_embedding_model = True
use_doc_embedding_model = False #overrides use_word_embedding_model
word_embedding_dim = 100


if use_word_embedding_model or use_doc_embedding_model:
    weights_all = [[0]*word_embedding_dim for _ in range(len(artists))]
else:
    weights_all = [defaultdict(int) for _ in range(len(artists))]

def learnPredictor(weights, x, y, featureExtractor, eta, lamb):
    for j in range(len(x)):
        phi = x[j]
        y_i = y[j]
        dp = 0
        if use_word_embedding_model or use_doc_embedding_model:
            for i in range(word_embedding_dim):
                dp += phi[i] * weights[i]
        else:
            dp = dotProduct(phi, weights)
        margin = dp*y_i
        if margin < 1:
            if use_word_embedding_model or use_doc_embedding_model:
                for i in range(word_embedding_dim):
                    gradient = -phi[i] * y_i
                    weights[i] -= eta * (gradient + lamb * weights[i])
            else:
                for f in phi:
                    gradient = -phi[f] * y_i
                    weights[f] -= eta * (gradient + lamb * weights[f])

#returns predicted artist
def predict_artist(phi):
    highest_score_index = 0
    highest_score = float('-inf')
    for i, weights in enumerate(weights_all):
        score = 0
        if use_word_embedding_model or use_doc_embedding_model:
            for j in range(word_embedding_dim):
                score += phi[j] * weights[j]
        else:
            score = dotProduct(phi, weights)
        if score > highest_score:
            highest_score_index = i
            highest_score = score
    #print 'highest score: '+str(highest_score)+ ' for artist '+str(artists[highest_score_index])
    return highest_score_index

def evaluatePredictors(x, y, print_incorrect_pairs=False, update_correct_counts=False, update_false_positive_counts=False):
    incorrect_pairs = defaultdict(int)
    error = 0
    for i in range(len(x)):
        predicted_y = predict_artist(x[i])
        true_y = y[i]
        if predicted_y != true_y:
            if print_incorrect_pairs:
                pair = [artists[predicted_y], artists[true_y]]
                pair.sort()
                incorrect_pairs[tuple(pair)] += 1
            #print 'prediction was incorrect'
            error += 1
            if update_false_positive_counts:
                artist_false_positives[predicted_y] += 1
        else:
            if update_correct_counts:
                artist_correct[predicted_y] += 1
        if update_correct_counts:
            artist_totals[true_y] += 1
        #else:
            #print 'prediction was correct'
    if print_incorrect_pairs: 
        d_view = [ (v,k) for k,v in incorrect_pairs.iteritems() ]
        d_view.sort(reverse=True) # natively sort tuples by first element
        print 'pairs of artists and number of times they were mixed up with each other'
        for v,k in d_view:
            print "%s: %d" % (k,v)
    return 1.0 * error / len(x)

train_errors = []
val_errors = []
test_errors = []

train_x, train_y, val_x, val_y, test_x, test_y = create_sets(n, datasets.all_songs_full_length.my_list, use_doc_embedding_model)[1]

if use_word_embedding_model and not use_doc_embedding_model: 
    train_x, val_x, test_x = x_sets_as_word_embeddings(train_x, val_x, test_x)

print 'running stochastic gradient descent'
for j in tqdm(range(numIters)):
    for i in range(len(artists)):
        #print 'training for artist '+str(artists[i])

        #edit y for particular artist (+1 if by artist i, -1 if not)
        new_train_y = []
        new_train_x = []
        for k, y_i in enumerate(train_y):
            if y_i == i:
                new_train_y.append(1)
            else:
                new_train_y.append(-1)

        learnPredictor(weights_all[i], train_x, new_train_y, featureExtractor, eta, lamb)
    train_error = evaluatePredictors(train_x, train_y)
    train_errors.append(train_error)
    val_error = evaluatePredictors(val_x, val_y)
    val_errors.append(val_error)
    test_error = evaluatePredictors(test_x, test_y)
    test_errors.append(test_error)

    #shuffle training data
    shuffled_indices = range(len(train_x))
    shuffle(shuffled_indices)
    shuffled_train_x = []
    shuffled_train_y = []
    for i in shuffled_indices:
        shuffled_train_x.append(train_x[i])
        shuffled_train_y.append(train_y[i])
    train_x = shuffled_train_x
    train_y = shuffled_train_y
    

test_error = evaluatePredictors(test_x, test_y, print_incorrect_pairs, print_correct_counts, print_false_positive_rates)
print 'training error: '+str(train_error)
print 'val error: '+str(val_error)
print 'test error: '+str(test_error)
x = range(numIters)
plt.plot(x, train_errors, label='Training Error')
plt.plot(x, val_errors, label='Dev Error')
plt.plot(x, test_errors, label='Test Error')
plt.xticks(range(1, numIters+1))
plt.xlabel('iteration')
plt.ylabel('error rate')
plt.title('Error Rate over Iterations for SVM with N='+str(n))
plt.legend()
plt.show()

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