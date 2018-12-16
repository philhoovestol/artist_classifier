This project contains 3 main models implementing artificial intelligence techniques to predict a song's artist given only its lyrics. Here we explain how they are structured and how one can run each implementation:

1. SVM - custom implementation of an SVM which optimizes using stochastic gradient descent, hinge loss, and regularizataion. Run by executing the script svm.py. Uses functions and variables declared in util.py. Any hyperparameters can be altered at their declaration at the top of svm.py.
	-If you want to run using word (Word2Vec) or document (Doc2Vec) embeddings* as features, set the corresponding boolean use_word_embeddings/use_doc_embeddings in util.py as True (use_doc_embeddings will override use_word_embeddings)

2. Naive Bayes Model - Run by executing bayes.py. Also uses functions and variables declared in util.py. adjust pseudocount (lambda) used for Laplace Smoothing with its variable at the top of bayes.py.

-Both models use word based N-gram feature extraction. The value of N can be adjusted through the variable n at the top of util.py
-for both of the models mentioned above, the following booleans at the top of util.py can be set to True to print certain values after classification and evaluation: 
	-print_correct_counts to print out a list of the artists in descending order of recall rate.
	-print_incorrect_pairs to print out a list of pairs of artists in descending order of the number of times the classifier confused one for the other
	-print_false_positive_rates to print out a list of the artists in descending order of their false positive rate

3. LSTM - run by executing lstm.py
