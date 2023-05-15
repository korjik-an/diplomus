import numpy as np


# подготовка тренировочных данных
def create_training(word_vectors_array, text_vectors_array, labels, train_size=0.95, maxlen=768):
    # Combine word and text vectors
    combined_array = np.concatenate((word_vectors_array, text_vectors_array), axis=1)

    # Делим на обучающую и тестовую выборки
    split_index = int(len(combined_array) * 0.95)
    x_train = combined_array[:split_index]
    y_train = labels[:split_index]
    x_test = combined_array[split_index:]
    y_test = labels[split_index:]

    # x_train = pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post', dtype='float32')
    # x_test = pad_sequences(x_test, maxlen=maxlen, padding='post', truncating='post', dtype='float32')

    return x_train, y_train, x_test, y_test



__all__ = ['create_training']
