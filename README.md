# explore-enron-emails
Deep learning and doc2vec exploration of enron email dataset: https://www.cs.cmu.edu/~enron/

Expects directory `./maildir`to hold unzipped dataset from above link.

Tensorboard logs, neural network model/weights/loss/accuracy files from training stored under `./logs`.

**To run:**

1) download dataset: https://www.cs.cmu.edu/~enron/

2) *python train_doc2vec.py* (with `should_create_data = True` on first run)

3) *python inference_doc2vec.py* (for sanity check that doc2vec operates correctly)

4) *python train_nn.py* (wiht `should_aggregate_data = True` on first run)

*Future Work:*
1) We attempt to only use `sent` mail from users.  Add more emails besides these.
2) Due to (1), we ignore two users.  Add these two users in.
3) Since number of emails varies by user, try weighting each class (user) by their email usage.
4) Try bayes
5) Try SVM
6) Try decision trees
7) Cross-validate.  Similar to link (4) below.
8) Try forming weights matrix by doc2vec as weights of embedding layer in neural network classifier.  Similar in theory to link (2) below.
9) Visualize weights.
10) Play with hyperparameters of doc2vec model.  Goes along with (7) above.
11) Play with hyperparameters of neural network classifier.  Goes along with (7) above.
12) Try kaggle's version of the dataset (may be cleansed/more uniform)
13) Try approaches covered under (5) of `Auxillary` resources below
14) Try approaches covered under (6) of `Auxillary` resources below

*Primary resources*:
1) http://linanqiu.github.io/2015/10/07/word2vec-sentiment/
2) https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
3) https://stackoverflow.com/questions/48842866/gensim-models-doc2vec-has-no-attribute-labeledsentence
4) https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
5) https://stackoverflow.com/questions/46197493/using-gensim-doc2vec-with-keras-conv1d-valueerror

Auxillary Resources:
1) https://ahmedbesbes.com/sentiment-analysis-on-twitter-using-word2vec-and-keras.html
2) https://www.kaggle.com/zichen/explore-enron/data
3) https://en.wikipedia.org/wiki/Word2vec#cite_note-doc2vec_java-11
4) https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb
5) https://medium.com/@williamkoehrsen/machine-learning-with-python-on-the-enron-dataset-8d71015be26d
6) https://medium.com/@klintcho/doc2vec-tutorial-using-gensim-ab3ac03d3a1
