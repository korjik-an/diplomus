import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# очищение текста
def clean_text(text):
    # приведение текста к нижнему регистру
    text = text.lower()

    # удаление знаков пунктуации
    text = re.sub(r'[^\w\s]', '', text)

    # удаление цифр из текста
    text = re.sub(r'\d+', '', text)

    # удаление стоп-слов
    stop_words = set(stopwords.words('russian'))
    words = word_tokenize(text, language='russian')
    words = [word for word in words if word not in stop_words]

    # объединение слов в очищенный текст
    text = ' '.join(words)
    return text


__all__ = ['clean_text']
