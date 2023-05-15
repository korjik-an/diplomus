from tensorflow import keras
from tensorflow.keras import layers
from transformers import AutoTokenizer, TFAutoModel
import numpy as np
import nltk
nltk.download('punkt')

# Загрузка предварительно обученной модели DeepPavlov/rubert-base-cased
model_name = 'DeepPavlov/rubert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
rubert_model = TFAutoModel.from_pretrained(model_name, from_pt=True)

# Пример функции для создания векторов слов с использованием модели DeepPavlov/rubert-base-cased
def get_word_vectors(words):
    input_ids = []
    attention_masks = []

    for word in words:
        encoded = tokenizer.encode_plus(
            word,
            add_special_tokens=True,
            max_length=32,
            padding='longest',
            return_attention_mask=True
        )

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = np.array(input_ids)
    attention_masks = np.array(attention_masks)

    inputs = [input_ids, attention_masks]
    outputs = rubert_model(inputs)
    word_vectors = outputs.last_hidden_state[:, 0, :]

    return word_vectors

# Пример функции для создания вектора текста с использованием модели DeepPavlov/rubert-base-cased
def get_text_vector(text):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='longest',
        return_attention_mask=True
    )

    input_ids = np.array(encoded['input_ids']).reshape((1, -1))
    attention_mask = np.array(encoded['attention_mask']).reshape((1, -1))

    inputs = [input_ids, attention_mask]
    outputs = rubert_model(inputs)
    text_vector = outputs.last_hidden_state[:, 0, :]

    return text_vector

# Пример массива текстов
texts = ["Это пример текста 1.", "Это пример текста 2.", "Это пример текста 3."]

# Токенизация и получение векторов слов для каждого текста
word_vectors = []
for text in texts:
    tokens = nltk.word_tokenize(text)
    vectors = get_word_vectors(tokens)
    word_vectors.append(vectors)

word_vectors = np.concatenate(word_vectors, axis=0)

# Получение векторов текстов
text_vectors = []
for text in texts:
    vector = get_text_vector(text)
    text_vectors.append(vector)

text_vectors = np.concatenate(text_vectors, axis=0)

# Пример меток классов для каждого текста
labels = [0, 1, 0]

# Создаем модель на основе Keras
text_input = layers.Input(shape=(text_vectors.shape[1],))

text_branch = layers.Dense(64, activation='relu')(text_input)
text_branch = layers.Dropout(0.5)(text_branch)


output = layers.Dense(1, activation='sigmoid')(text_branch)

model = keras.Model(inputs=text_input, outputs=output)

# Компилируем модель
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Тренируем модель
model.fit(text_vectors, np.array(labels), epochs=10, batch_size=16)