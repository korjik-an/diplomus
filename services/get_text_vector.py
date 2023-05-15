from transformers import BertTokenizer, BertModel
import torch
import numpy as np


# получение вектора всего текста
def get_text_vector(text):
    # загрузка предобученной модели BERT
    model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

    # применение токенизатора Bert к тексту
    text_token = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        last_hidden_states = model(**text_token)[0]

        # нормализация вектора текста
        text_vector = last_hidden_states.mean(dim=1).squeeze().numpy()
        text_vector /= np.linalg.norm(text_vector)
    return text_vector


__all__ = ['get_text_vector']
