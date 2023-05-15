def classify_text(text, model, threshold=0.55):
    # преобразуем текст в вектор
    text_vector = preprocess_text(text)
    # получаем вероятности принадлежности к классам
    probabilities = model.predict(np.array([text_vector]))[0]
    # определяем класс, если вероятность больше порогового значения
    if probabilities[0] > threshold:
        return "человек"
    elif probabilities[1] > threshold:
        return "ИИ"
    else:
        return "нельзя однозначно определить"

text = "Это простой текст, написанный человеком"
model = create_model()
# загрузить веса модели, сохраненные на этапе обучения
model.load_weights('model_weights.h5')
result = classify_text(text, model)
print(result)
