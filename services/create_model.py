import keras


# keras создание модели
def create_model(input_shape_word, input_shape_text):
    input_word = keras.Input(shape=input_shape_word)
    input_text = keras.Input(shape=input_shape_text)

    x_word = keras.layers.Dense(32, activation="relu")(input_word)
    x_word = keras.layers.Dense(16, activation="tanh")(x_word)

    x_text = keras.layers.Dense(32, activation="relu")(input_text)
    x_text = keras.layers.Dense(16, activation="tanh")(x_text)

    merged = keras.layers.concatenate([x_word, x_text])

    output = keras.layers.Dense(1, activation="sigmoid")(merged)

    model = keras.Model(inputs=[input_word, input_text], outputs=output)
    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
    # model = keras.Sequential()
    # # model.add(keras.layers.GlobalAveragePooling1D())
    # # посмотреть на результаты и попробовать поменять на 16 или 64 например
    # model.add(keras.layers.Dense(32, activation="relu", input_shape=input_shape))
    # model.add(keras.layers.Dense(32, activation="tanh"))
    # model.add(keras.layers.Dense(2, activation="sigmoid"))
    # model.summary()
    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # return (model)


__all__ = ['create_model']
