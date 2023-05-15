import numpy as np
import torch
from transformers import BertModel, BertTokenizer


# получение векторов слов
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


__all__ = ['get_word_vectors']
