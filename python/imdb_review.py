import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import unicodedata

# 1. 학습 환경 및 하이퍼파라미터
keras.utils.set_random_seed(42)
VOCAB_SIZE = 10000
MAX_LEN = 256
EPOCHS = 40
BATCH_SIZE = 512

# 2. IMDB 영화 리뷰 데이터셋
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)

# 3. 데이터 전처리 및 원-핫 인코딩
x_train = keras.utils.pad_sequences(x_train, maxlen=MAX_LEN, padding='post')
x_test = keras.utils.pad_sequences(x_test, maxlen=MAX_LEN, padding='post')
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

print("부정: [1, 0] / 긍정: [0, 1]")
print("원-핫 인코딩 결과:\n", y_train[:5])
print("훈련 데이터:", x_train.shape)
print("테스트 데이터:", x_test.shape)

# 4. 훈련 및 검증 데이터 분리
x_val, x_fit = x_train[:7500], x_train[7500:]
y_val, y_fit = y_train[:7500], y_train[7500:]

# 5. 영화 리뷰 감성 분류 모델
model = keras.Sequential([
    keras.Input(shape=(MAX_LEN,)),
    keras.layers.Embedding(VOCAB_SIZE, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.0005)),
    keras.layers.Dense(2, activation='softmax')
])

# 6. 모델 구조 및 컴파일
model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. 최적 모델 저장
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_imdb.keras', monitor='val_loss',
    mode='min', save_best_only=True
)

# 8. 모델 학습
history = model.fit(
    x_fit, y_fit,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint],
    verbose=2
)

# 9. 최적 모델 평가
best_model = keras.models.load_model('best_imdb.keras')
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)

h = history.history
best_epoch = np.argmin(h['val_loss'])

print("\n" + "=" * 55)
print("       IMDB 영화 리뷰 분류 모델 학습 결과")
print("=" * 55)
print(f"총 학습 횟수          : {EPOCHS} Epoch")
print(f"최적 학습 시점        : {best_epoch + 1} Epoch")
print(f"최적 모델 검증 정확도 : {h['val_accuracy'][best_epoch] * 100:.2f}%")
print(f"최고 검증 정확도      : {max(h['val_accuracy']) * 100:.2f}%")
print(f"최종 테스트 정확도    : {test_acc * 100:.2f}%")
print(f"최종 테스트 손실      : {test_loss:.4f}")
print("=" * 55)

# 10. 영화 리뷰 감성 예측 결과
SAMPLE_COUNT = 3
predictions = best_model.predict(x_test[:SAMPLE_COUNT], verbose=0)

def align_text(value, width):
    s = str(value)
    size = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)
    return s + ' ' * max(0, width - size)

def print_row(values):
    widths = [6, 10, 10, 13, 13, 8]
    print("| " + " | ".join(align_text(v, w) for v, w in zip(values, widths)) + " |")

print("\n===== 영화 리뷰 감성 예측 결과 =====")
print_row(["리뷰", "실제 감성", "예측 감성", "부정 확률", "긍정 확률", "판정"])
print("-" * 90)

for i, probs in enumerate(predictions):
    actual = np.argmax(y_test[i])
    predicted = np.argmax(probs)
    print_row([
        i + 1,
        "긍정" if actual == 1 else "부정",
        "긍정" if predicted == 1 else "부정",
        f"{probs[0] * 100:.2f}%",
        f"{probs[1] * 100:.2f}%",
        "정답" if actual == predicted else "오답"
    ])

print("-" * 90)

# 11. Loss 및 Accuracy 그래프
plt.close('all')
epochs = range(1, len(h['loss']) + 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Loss 그래프
axes[0].plot(epochs, h['loss'], 'bo', markersize=5, label='Training loss')
axes[0].plot(epochs, h['val_loss'], 'b-', linewidth=1.5, label='Validation loss')
axes[0].set_title('Training and validation loss')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].set_xlim(0, EPOCHS)
axes[0].legend()

# Accuracy 그래프
axes[1].plot(epochs, h['accuracy'], 'bo', markersize=5, label='Training acc')
axes[1].plot(epochs, h['val_accuracy'], 'b-', linewidth=1.5, label='Validation acc')
axes[1].set_title('Training and validation accuracy')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].set_xlim(0, EPOCHS)
axes[1].legend()

# 12. Xming 그래프 출력
plt.tight_layout()
fig.canvas.draw()
plt.show(block=True)