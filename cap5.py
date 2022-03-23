import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizer_v2.gradient_descent import SGD

x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
y_train = np.array([[0], [1], [1], [0]])
model = Sequential()
num = 10
model.add(Dense(num, input_dim=2))
model.add(Activation("tanh"))
model.add(Dense(1))
model.add(Activation("sigmoid"))
print(model.summary())
sgd = SGD(learning_rate=0.1)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
print(model.predict(x_train))
model.fit(x_train, y_train, epochs=1000)
print(model.predict(x_train))

model_structure = model.to_json()
with open("basic_model.json", "w") as f:
    f.write(model_structure)
model.save_weights("basic_weights.h5")