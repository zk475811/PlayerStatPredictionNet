from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# player, home, age, last_game_score
# could use last game deviation from average like +5 or -2
num_features = 4
# player number 1 is Kyrie Irving and 2 is James Harden
# 0 is away, 1 is home
x_train = np.array([[1, 0, 25, 50],
		    [1, 0, 25, 29],
		    [1, 1, 25, 33],
		    [1, 0, 25, 33],
		    [2, 0, 28, 54],
		    [2, 0, 28, 46],
		    [2, 1, 28, 42]])

# fantasy points
y_train = np.array([[29], [33], [33], [42], [46], [42], [35]])

# need to add more data for testing and validation this just temporary to make sure net learns
x_test = x_train
y_test = y_train

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=num_features))
model.add(Dense(1, activation='relu'))

# mean square error regression
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=1, epochs=100)
score = model.evaluate(x_test, y_test, batch_size=1)

print()
# Kyrie Irving away and home game prediction
next_game_home = np.array([[1, 0, 25, 53]])
next_game_away = np.array([[1, 1, 25, 33]])
print(model.predict(next_game_home, batch_size=1))
print(model.predict(next_game_away, batch_size=1))
# James Harden test predictions
print(model.predict(np.array([[2, 0, 28, 35]]), batch_size=1))
print(model.predict(np.array([[2, 0, 28, 45]]), batch_size=1))
