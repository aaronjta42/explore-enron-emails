# explore-enron-emails
Deep learning and doc2vec exploration of enron email dataset: https://www.cs.cmu.edu/~enron/

Expects directory `./maildir`to hold unzipped dataset from above link.

**To run:**

1) download dataset: https://www.cs.cmu.edu/~enron/

2) *python train_doc2vec.py* (with `should_create_data = True` on first run)

3) *python inference_doc2vec.py* (for sanity check that doc2vec operates correctly)

4) *python train_nn.py* (wiht `should_aggregate_data = True` on first run)

Tensorboard logs, neural network model/weights/loss/accuracy files from training stored under `./logs`.



*Useful resources*:
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

