import os

import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report

from services.clean_text import clean_text
from services.get_text_vector import get_text_vector

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
# def get_text_vector(text):
#     encoded = tokenizer.encode_plus(
#         text,
#         add_special_tokens=True,
#         max_length=128,
#         padding='longest',
#         return_attention_mask=True
#     )
#
#     input_ids = np.array(encoded['input_ids']).reshape((1, -1))
#     attention_mask = np.array(encoded['attention_mask']).reshape((1, -1))
#
#     inputs = [input_ids, attention_mask]
#     outputs = rubert_model(inputs)
#     text_vector = outputs.last_hidden_state[:, 0, :]
#
#     return text_vector


print('разметка датасета')
current_dir = os.path.dirname(os.path.abspath(__file__))
data_folders = [os.path.join(current_dir, 'data', 'human_texts'), os.path.join(current_dir, 'data', 'ai_texts')]
labels = [0, 1]
texts = []

for folder, label in zip(data_folders, labels):
    files = os.listdir(folder)
    for i, file in enumerate(files):
        if i >= 500:  # берем только первые 10 текстов
            break
        with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
            text = f.read(1200)  # read only the first 1200 characters of the text file
            texts.append({'filename': file, 'text': text, 'label': label})

global df
df = pd.DataFrame(texts)
df.to_csv('dataset.csv', index=False)
# Пример массива текстов
# texts = ["Это пример текста 1.", "Это пример текста 2.", "Это пример текста 3."]

# Токенизация и получение векторов слов для каждого текста


# word_vectors_list = []
#
# for text in df['text']:
#     text = clean_text(text)
#     tokens = nltk.word_tokenize(text)
#     vectors = get_word_vectors(tokens)
#     word_vectors_list.append(vectors)
#
# word_vectors_list = np.concatenate(word_vectors_list, axis=0)

# Получение векторов текстов
text_vectors = []
for text in df['text']:
    text = clean_text(text)
    text_vector = get_text_vector(str(text))
    text_vectors.append(text_vector)

#text_vectors = np.concatenate(text_vectors, axis=0)
#print(text_vectors.shape)
#text_vectors = text_vectors.reshape((-1, 1))
#print(text_vectors.shape)

#labels_array = np.array(df['label'].values)
#print(labels_array.shape)

text_vectors = np.concatenate(text_vectors, axis=0)
text_vectors = text_vectors.reshape((-1, 1))

labels_array = np.array(df['label'].values)

train_ratio = 0.95
train_size = int(train_ratio * len(labels_array))

combined = list(zip(text_vectors, labels_array))
np.random.shuffle(combined)
text_vectors_shuffled, labels_array_shuffled = zip(*combined)

# Преобразуем обратно в массивы numpy
text_vectors_shuffled = np.array(text_vectors_shuffled)
labels_array_shuffled = np.array(labels_array_shuffled)

train_text_vectors = text_vectors_shuffled[:train_size]
train_labels = labels_array_shuffled[:train_size]
train_labels = keras.utils.to_categorical(train_labels, num_classes=2)

test_text_vectors = text_vectors_shuffled[train_size:]
test_labels = labels_array_shuffled[train_size:]
test_labels = keras.utils.to_categorical(test_labels, num_classes=2)

test_text_vectors = test_text_vectors.reshape((-1, 1))
print(test_labels.shape)
print(test_text_vectors.shape)

#train_ratio = 0.8
#train_size = int(train_ratio * len(text_vectors))

#train_text_vectors = text_vectors[:train_size]
#train_labels = labels_array[:train_size]

#test_text_vectors = text_vectors[train_size:]
#test_labels = labels_array[train_size:]

val_size = .3
vals = int(len(train_text_vectors) * val_size)

x_val = train_text_vectors[:vals]
x_train = train_text_vectors[vals:]

y_val = train_labels[:vals]
y_train = train_labels[vals:]


#text_vectors = text_vectors.reshape((-1, text_vectors.shape[1]))
#text_vectors = text_vectors.reshape((1, -1))
#print(text_vectors.shape)

# Создаем модель на основе Keras
text_input = layers.Input(shape=(text_vectors.shape[1],))

text_branch = layers.Dense(32, activation='relu')(text_input)
text_branch = layers.Dense(16, activation='tanh')(text_branch)
text_branch = layers.Dropout(0.5)(text_branch)

output = layers.Dense(2, activation='softmax')(text_branch)

model = keras.Model(inputs=text_input, outputs=output)

# Компилируем модель
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', ])

# Тренируем модель
model_res = model.fit(train_text_vectors, np.array(train_labels), epochs=10, batch_size=16)
print(model_res.history)

# Оценка модели на тестовых данных
predictions = model.predict(test_text_vectors)
predictions_rounded = predictions.round()

loss, accuracy = model.evaluate(test_text_vectors, test_labels)
print("Loss:", loss)
print("Accuracy:", accuracy)

if len(test_labels) == len(predictions_rounded):
    precision = precision_score(test_labels, predictions_rounded, average='weighted')
    recall = recall_score(test_labels, predictions_rounded, average='weighted')
    f1 = f1_score(test_labels, predictions_rounded, average='weighted')
    accuracy = accuracy_score(test_labels, predictions_rounded)
else:
    print("Error: Mismatch in array sizes.")
    print(test_labels.shape)
    print(test_text_vectors.shape)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Accuracy:", accuracy)



# проверки
# print(df.head())
# print(text_vectors[0])
# print(word_vectors[0])
