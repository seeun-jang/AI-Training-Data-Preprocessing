import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. IMDB 데이터셋
# =========================

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = \
    imdb.load_data(num_words=10000)


print(
    "훈련 데이터: {}, 훈련 레이블: {}".format(
        len(train_data),
        len(train_labels)
    )
)

print(train_data[0])

print(
    len(train_data[0]),
    len(train_data[1])
)


# =========================
# 2. 단어 사전
# =========================

word_index = imdb.get_word_index()

word_index = {
    k: (v + 3)
    for k, v in word_index.items()
}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3


reverse_word_index = dict(
    [
        (value, key)
        for (key, value)
        in word_index.items()
    ]
)


def decode_review(text):
    return ' '.join(
        [
            reverse_word_index.get(i, '?')
            for i in text
        ]
    )


print(decode_review(train_data[0]))


# =========================
# 3. 길이 256으로 맞추기
# =========================

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index["<PAD>"],
    padding="post",
    maxlen=256
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index["<PAD>"],
    padding="post",
    maxlen=256
)


print(
    len(train_data[0]),
    len(train_data[1])
)

print(train_data[0])


# =========================
# 4. ★ 원-핫 벡터 변환
# =========================

train_labels = keras.utils.to_categorical(
    train_labels,
    num_classes=2
)

test_labels = keras.utils.to_categorical(
    test_labels,
    num_classes=2
)


print("원-핫 변환된 라벨:")
print(train_labels[:5])


# =========================
# 5. 모델 구성
# =========================

vocab_size = 10000


model = keras.Sequential()

model.add(
    keras.layers.Embedding(
        vocab_size,
        16,
        input_shape=(None,)
    )
)

model.add(
    keras.layers.GlobalAveragePooling1D()
)

model.add(
    keras.layers.Dense(
        16,
        activation='relu'
    )
)

model.add(
    keras.layers.Dense(
        2,
        activation='softmax'
    )
)


model.summary()


# =========================
# 6. 모델 컴파일
# =========================

model.compile(
    optimizer='adam',

    # ★ 원-핫 벡터이므로 categorical_crossentropy
    loss='categorical_crossentropy',

    metrics=['accuracy']
)


# =========================
# 7. 검증 / 학습 데이터 분리
# =========================

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# =========================
# 8. 학습
# =========================

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)


# =========================
# 9. 테스트 평가
# =========================

results = model.evaluate(
    test_data,
    test_labels,
    verbose=2
)

print(results)


# =========================
# 10. 학습 결과
# =========================

history_dict = history.history

print(history_dict.keys())


acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(
    1,
    len(acc) + 1
)


# =========================
# 11. Loss 그래프
# =========================

plt.plot(
    epochs,
    loss,
    'bo',
    label='Training loss'
)

plt.plot(
    epochs,
    val_loss,
    'b',
    label='Validation loss'
)

plt.title(
    'Training and validation loss'
)

plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()

plt.show()


# =========================
# 12. Accuracy 그래프
# =========================

plt.clf()


plt.plot(
    epochs,
    acc,
    'bo',
    label='Training acc'
)

plt.plot(
    epochs,
    val_acc,
    'b',
    label='Validation acc'
)

plt.title(
    'Training and validation accuracy'
)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend()

plt.show()
