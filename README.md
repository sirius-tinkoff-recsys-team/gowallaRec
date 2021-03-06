# Рекомендации на примере датасета Gowalla

EDA можно посмотреть в ноутбуке [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SMwCJsS2YFxeANKAdXlwDlxL_IHNt4Hl)

## Реализованы модели:

- [iALS](https://implicit.readthedocs.io/en/latest/als.html) из библиотеки implicit, основан на матричной фактиризации
- TopNModel рекомендует самые популярные локации среди всех пользователей на тренировочном периоде
- TopNPersonalized рекомендует самые популярные локации для каждого пользователя на тренировочном периоде
- TopNNearestModel рекомендует самые близкие к последней локации пользователя локации
- [LightGCN](https://arxiv.org/abs/2002.02126) с bpr лоссом, графовая сеть, которая ходит
- [Catboost](https://catboost.ai) обучается с Logloss и ранжирует кандидатов из моделей первого уровня

## Подготовка данных:
`prepare_gowalla_dataset.py` скачивает нужные данные и подготавливает разбиение трейн/тест/валидация

`prepare_catboost_dataset.py` готовт кандидатов для обучения catboost, возвращает csv файлы вида `(userId, itemId)`

## Обучение:
`train.py` обучает модель, указанную в `config.yaml`

`fit_catboost.py` обучает Catboost и сохраняет модель `catboost.cbm`