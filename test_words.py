import os
import torch
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
#from transformers import AutoTokenizer, TFAutoModel
from transformers import BertModel, BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from services.clean_text import clean_text
from services.get_word_vectors import get_word_vectors

nltk.download('punkt')

# загрузка предварительно обученной модели DeepPavlov/rubert-base-cased
#model_name = 'DeepPavlov/rubert-base-cased'
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#rubert_model = TFAutoModel.from_pretrained(model_name, from_pt=True)


# пример функции для создания векторов слов с использованием модели DeepPavlov/rubert-base-cased
#def get_word_vectors(words):
 #   input_ids = []
  #  attention_masks = []

   # for word in words:
    #    encoded = tokenizer.encode_plus(
    #        word,
    #        add_special_tokens=True,
    #        max_length=32,
    #        padding='longest',
    #        return_attention_mask=True
    #   )

    #    input_ids.append(encoded['input_ids'])
    #    attention_masks.append(encoded['attention_mask'])

    #input_ids = np.array(input_ids, dtype=object)
   # attention_masks = np.array(attention_masks, dtype=object)

    #input_ids = np.array(input_ids)
    #attention_masks = np.array(attention_masks)

   # inputs = [input_ids, attention_masks]
   # inputs = tf.convert_to_tensor(inputs)
   # outputs = rubert_model(inputs)
   # word_vectors = outputs.last_hidden_state[:, 0, :]

   # return word_vectors



def get_word_vectors(text):
    model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    # применение токенизатора Bert к тексту
    tokens = tokenizer.tokenize(text)
    # добавление CLS и SEP токенов в начало и конец вектора токенов
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    # получение идентификатора для каждого токена
    token_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    # получение векторов BERT для входных токенов
    outputs = model(token_ids)
    last_hidden_states = outputs[0].squeeze(0)
    # нормализация векторов слов
    # word_vectors = [vectors / np.linalg.norm(vectors) for vectors in last_hidden_states]
    word_vectors = [vectors.detach().numpy() / np.linalg.norm(vectors.detach().numpy(), ord=2) for vectors in
                    last_hidden_states]
    # text_vector =
    return word_vectors

# пример функции для создания вектора текста с использованием модели  DeepPavlov/rubert-base-cased
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
        if i >= 500:
            break
        with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
            text = f.read(1200)  # read only the first 1200 characters of the text file
            texts.append({'filename': file, 'text': text, 'label': label})

global df
df = pd.DataFrame(texts)
df.to_csv('dataset.csv', index=False)
# пример массива текстов

# токенизация и получение векторов слов для каждого текста
word_vectors_list = []

for text in df['text']:
     text = clean_text(text)
     #tokens = nltk.word_tokenize(text)
     vectors = get_word_vectors(text)
     word_vectors_list.append(vectors)

word_vectors = np.concatenate(word_vectors_list, axis=0)
print(word_vectors.shape)

word_vectors = word_vectors.reshape((word_vectors.shape[0], word_vectors.shape[1]))
print(word_vectors.shape)

# получение векторов текстов

#for text in df['text']:
#    text = clean_text(text)
#   word_vector = get_words_vectors(str(text))
#  word_vectors.append(word_vector)

#text_vectors = np.concatenate(text_vectors, axis=0)
#print(text_vectors.shape)
#text_vectors = text_vectors.reshape((-1, 1))
#print(text_vectors.shape)

#labels_array = np.array(df['label'].values)
#print(labels_array.shape)

#text_vectors = np.concatenate(text_vectors, axis=0)
#text_vectors = text_vectors.reshape((-1, 1))

labels_array = np.array(df['label'].values)
print(labels_array.shape)

train_ratio = 0.8
train_size = int(train_ratio * len(labels_array))

combined = list(zip(word_vectors, labels_array))
np.random.shuffle(combined)
word_vectors_shuffled, labels_array_shuffled = zip(*combined)

# преобразуем обратно в массивы numpy
word_vectors_shuffled = np.array(word_vectors_shuffled)
labels_array_shuffled = np.array(labels_array_shuffled)

train_word_vectors = word_vectors_shuffled[:train_size]
train_labels = labels_array_shuffled[:train_size]
train_labels = keras.utils.to_categorical(train_labels, num_classes=2)

test_word_vectors = word_vectors_shuffled[train_size:]
test_labels = labels_array_shuffled[train_size:]
test_labels = keras.utils.to_categorical(test_labels, num_classes=2)

test_word_vectors = test_word_vectors.reshape((-1, 768))
print(test_labels.shape)
print(test_word_vectors.shape)

#train_ratio = 0.8
#train_size = int(train_ratio * len(text_vectors))

#train_text_vectors = text_vectors[:train_size]
#train_labels = labels_array[:train_size]

#test_text_vectors = text_vectors[train_size:]
#test_labels = labels_array[train_size:]

val_size = .3
vals = int(len(train_word_vectors) * val_size)

x_val = train_word_vectors[:vals]
x_train = train_word_vectors[vals:]

y_val = train_labels[:vals]
y_train = train_labels[vals:]


#text_vectors = text_vectors.reshape((-1, text_vectors.shape[1]))
#text_vectors = text_vectors.reshape((1, -1))
#print(text_vectors.shape)

# создаем модель на основе Keras
word_input = layers.Input(shape=(word_vectors.shape[1],))

word_branch = layers.Dense(16, activation='relu')(word_input)
word_branch = layers.Dropout(0.5)(word_branch)
word_branch = layers.Dense(16, activation='tanh')(word_branch)
word_branch = layers.Dropout(0.5)(word_branch)

output = layers.Dense(2, activation='sigmoid')(word_branch)

model = keras.Model(inputs=word_input, outputs=output)

# компилируем модель
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', ])

# тренируем модель
#early_stopping = EarlyStopping(patience=2, monitor='val_loss', mode='min')
#model_res = model.fit(train_word_vectors, train_labels, epochs=100, batch_size=16, validation_data=(test_word_vectors, test_labels), callbacks=[early_stopping])
model_res = model.fit(train_word_vectors, np.array(train_labels), epochs=6, batch_size=16)
print(model_res.history)

# оценка модели на тестолвых данных
predictions = model.predict(test_word_vectors)
predictions_rounded = predictions.round()


loss, accuracy = model.evaluate(test_word_vectors, test_labels)
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
    print(test_word_vectors.shape)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Accuracy:", accuracy)

