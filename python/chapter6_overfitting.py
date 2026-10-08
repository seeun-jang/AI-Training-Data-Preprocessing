import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

NUM_WORDS = 1000


# =========================
# 1. IMDB 데이터
# =========================

(train_data, train_labels), (test_data, test_labels) = \
    keras.datasets.imdb.load_data(num_words=NUM_WORDS)

train_labels = train_labels.reshape(-1, 1)
test_labels = test_labels.reshape(-1, 1)


# =========================
# 2. Multi-Hot Encoding
# =========================

def multi_hot_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))

    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0

    return results


train_data = multi_hot_sequences(train_data, NUM_WORDS)
test_data = multi_hot_sequences(test_data, NUM_WORDS)


# =========================
# 3. 기본 모델 생성 함수
# =========================

def make_model(units):
    model = keras.Sequential([
        keras.layers.Dense(
            units,
            activation='relu',
            input_shape=(NUM_WORDS,)
        ),
        keras.layers.Dense(
            units,
            activation='relu'
        ),
        keras.layers.Dense(
            1,
            activation='sigmoid'
        )
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy']
    )

    return model


baseline_model = make_model(16)
smaller_model = make_model(4)
bigger_model = make_model(512)


# =========================
# 4. L2 규제 적용 모델
# =========================

l2_model = keras.Sequential([
    keras.layers.Dense(
        16,
        kernel_regularizer=keras.regularizers.l2(0.001),
        activation='relu',
        input_shape=(NUM_WORDS,)
    ),

    keras.layers.Dense(
        16,
        kernel_regularizer=keras.regularizers.l2(0.001),
        activation='relu'
    ),

    keras.layers.Dense(
        1,
        activation='sigmoid'
    )
])

l2_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy']
)


# =========================
# 5. Dropout 적용 모델
# =========================

dropout_model = keras.Sequential([
    keras.layers.Dense(
        16,
        activation='relu',
        input_shape=(NUM_WORDS,)
    ),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(
        16,
        activation='relu'
    ),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(
        1,
        activation='sigmoid'
    )
])

dropout_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy']
)


# =========================
# 6. 모델 학습 - 전부 Epoch 100
# =========================

baseline_history = baseline_model.fit(
    train_data,
    train_labels,
    epochs=100,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2
)

smaller_history = smaller_model.fit(
    train_data,
    train_labels,
    epochs=100,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2
)

bigger_history = bigger_model.fit(
    train_data,
    train_labels,
    epochs=100,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2
)

l2_model_history = l2_model.fit(
    train_data,
    train_labels,
    epochs=100,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2
)

dropout_history = dropout_model.fit(
    train_data,
    train_labels,
    epochs=100,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2
)


# =========================
# 7. 모델 평가 그래프
# =========================

def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(12, 7))

    for name, history in histories:
        val = plt.plot(
            history.epoch,
            history.history['val_' + key],
            '--',
            label=name.title() + ' Val'
        )

        plt.plot(
            history.epoch,
            history.history[key],
            color=val[0].get_color(),
            label=name.title() + ' Train'
        )

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)
    plt.show()


plot_history([
    ('baseline', baseline_history),
    ('smaller', smaller_history),
    ('bigger', bigger_history),
    ('l2', l2_model_history),
    ('dropout', dropout_history)
])
