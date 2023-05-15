import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from services.clean_text import clean_text
from services.create_model import create_model
from services.create_training import create_training
from services.get_text_vector import get_text_vector
from services.get_word_vectors import get_word_vectors
from services.train_model import train_model


def main():
    print('разметка датасета')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folders = [os.path.join(current_dir, 'data', 'human_texts'), os.path.join(current_dir, 'data', 'ai_texts')]
    labels = [0, 1]
    texts = []

    for folder, label in zip(data_folders, labels):
        files = os.listdir(folder)
        for file in files:
            with open(os.path.join(folder, file), 'r') as f:
                text = f.read(1200)  # read only the first 1200 characters of the text file
                texts.append({'filename': file, 'text': text, 'label': label})

    global df
    df = pd.DataFrame(texts)
    df.to_csv('dataset.csv', index=False)

    print('создание массивов для полученных векторов')
    text_vectors = []
    word_vectors_list = []

    print('считывание текстов для обработки')
    for text in df['text']:
        # очистка текста
        text = clean_text(text)

        # получение векторов слов текста
        word_vectors = get_word_vectors(str(text))
        word_vectors = np.array(word_vectors)

        # получение вектора текста
        text_vector = get_text_vector(str(text))

        # добавление векторов в список
        text_vectors.append(text_vector)
        word_vectors_list.append(word_vectors)

    print("Перебор закончился")
    # print("word_vectors_list:", word_vectors_list)
    # print("text_vectors:", text_vectors)
    # преобразование списков в массивы numpy
    max_length = max(len(vectors) for vectors in word_vectors_list)

    # Выполнить паддинг или обрезку векторов до максимальной длины
    word_vectors_list = [np.pad(vectors, ((0, max_length - len(vectors)), (0, 0)), mode='constant') if len(
        vectors) < max_length else vectors[:max_length] for vectors in word_vectors_list]

    # Преобразовать список в массив numpy
    word_vectors_array = np.array(word_vectors_list)
    word_vectors_array = word_vectors_array.reshape(-1, 1)
    text_vectors_array = np.array(text_vectors)
    print("Обработка BERT закончена")
    # создание модели
    input_shape = (768,)
    model = create_model(input_shape)
    print("Модель создана")

    # обучение модели
    print(text_vectors_array)
    text_vectors_array = pad_sequences(text_vectors_array.tolist(), maxlen=768, padding='post', truncating='post',
                                       dtype='float32')
    labels_array = df['label'].values

    # создание кортежа tt_data
    tt_data = create_training(word_vectors_array, text_vectors_array, labels_array, maxlen=768, train_size=0.95)

    # код для обучения модели
    fit_model, model_results = train_model(model, tt_data, val_size=.3, epochs=10, batch_size=16)

    print("Обучение произведено")

    # оценка результатов на тестовых данных
    test_loss, test_acc, test_precision, test_recall, test_f1score = model.evaluate(model_results.testing_data,
                                                                                    model_results.testing_labels)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}")
    print(f"Test precision: {test_precision}")
    print(f"Test recall: {test_recall}")
    print(f"Test F1 score: {test_f1score}")

    # проверки
    # print(df.head())
    # print(text_vectors[0])
    # print(word_vectors[0])


if __name__ == "__main__":
    main()
