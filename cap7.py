import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalAvgPool1D

import glob
import os

from random import shuffle

def pre_process(file_path):
    positive_path = os.path.join(file_path, "pos")
    negative_path = os.path.join(file_path, "neg")
    pos_label = 1
    neg_label = 0
    dataset = []
    for filename in glob.glob(os.path.join(positive_path, "*.txt")):
        with open(filename, "r") as f:
            dataset.append((pos_label, f.read()))

    for filename in glob.glob(os.path.join(negative_path, "*.txt")):
        with open(filename, "r") as f:
            dataset.append((neg_label, f.read()))

    shuffle(dataset)
    return dataset

file_path = "./aclImdb_v1/aclImdb/test"
dataset = pre_process(file_path)
print(dataset[0])

from nltk.tokenize import TreebankWordTokenizer
from gensim.models.keyedvectors import KeyedVectors
from nlpia.loaders import get_data
# word_vectors = get_data("w2v", limit=200000)
word_vectors = KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin.gz", binary=True, limit=200000)


def tokenize_and_vectorize(dataset):
    tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    expected = []
    for sample in dataset:
        tokens = tokenizer.tokenize(sample[1])
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                pass
        vectorized_data.append(sample_vecs)
    return vectorized_data

def collect_expected(dataset):
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected

vectorized_data = tokenize_and_vectorize(dataset)
expected = collect_expected(dataset)

split_point = int(len(vectorized_data) * .8)
x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]

maxlen = 400
batch_size = 32
embedding_dims = 300
filters = 250
kernel = 3
hidden_dims = 250
epochs = 2

def pad_trunc(data, maxlen):
    new_data = []
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0)

    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data


x_train = pad_trunc(x_train, maxlen)
x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)

x_test = pad_trunc(x_test, maxlen)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)

print("build model...")
model = Sequential()
model.add(Conv1D(
    filters,
    kernel,
    padding="valid",
    activation="relu",
    strides=1,
    input_shape=(maxlen, embedding_dims)
))

model.add(GlobalAvgPool1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.add(Dense(num_classes))
model.add(Activation("sigmoid"))
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
model_structure = model.to_json()
with open("cnn_model", "w") as json_file:
    json_file.write(model_structure)
model.save_weights("cann_wight.h5")


from keras.models import model_from_json
with open("cnn_model.json", "r") as f:
    json_string = f.read()
model = model_from_json(json_string)
model.load_weights("cnn_wrights.h5")

smaple = "fuck you man"
vec_list = tokenize_and_vectorize([(1, smaple)])
test_vec_list = pad_trunc(vec_list, maxlen)
test_vec = np.reshape(test_vec_list, (len(test_vec_list), maxlen, embedding_dims))
model.prefict(test_vec)
model.predict_classes(test_vec)
