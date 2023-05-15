import keras


# keras создание модели
def create_model(input_shape):
    model = keras.Sequential()
    # model.add(keras.layers.GlobalAveragePooling1D())
    # посмотреть на результаты и попробовать поменять на 16 или 64 например
    model.add(keras.layers.Dense(32, activation="relu", input_shape=input_shape))
    model.add(keras.layers.Dense(32, activation="tanh"))
    model.add(keras.layers.Dense(2, activation="sigmoid"))
    model.summary()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return (model)


__all__ = ['create_model']
