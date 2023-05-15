from transformers import BertTokenizer


# токенизация слов
def tokenize_text(text):
    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    # токенизация по словам
    tokens = tokenizer.tokenize(text)
    # добавление CLS и SEP токенов в начало и конец вектора токенов
    tokens = tokens
    # получение идентификатора для каждого токена
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return token_ids


__all__ = ['tokenize_text']
