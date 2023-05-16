# keras тренировка модели
def train_model(model, tt_data, val_size=.3, epochs=1, batch_size=16):
    vals = int(len(tt_data[0]) * val_size)
    training_data = tt_data[0]
    training_labels = tt_data[1]
    testing_data = tt_data[2]
    testing_labels = tt_data[3]

    x_val = training_data[:vals]
    x_train = training_data[vals:]

    y_val = training_labels[:vals]
    y_train = training_labels[vals:]

    fit_model = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                          verbose=1, shuffle=True)
    model_results = model.evaluate(testing_data, testing_labels)
    return fit_model, model_results


__all__ = ['train_model']
