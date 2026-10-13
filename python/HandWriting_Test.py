# 실습과제 1: MNIST 손글씨 훈련 결과 시각화
#  (1) 훈련/평가의 Epoch별 Loss 그래프
#  (2) 잘못 구분한 이미지 표현
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
 
# 1. 데이터 준비
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
y_train_oh = keras.utils.to_categorical(y_train, 10)
y_test_oh = keras.utils.to_categorical(y_test, 10)
 
# 2. 모델 구성
model = keras.Sequential([
    keras.Input(shape=(784,)),
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()
 
# 3. 학습 (평가 데이터를 validation_data로 넣어 Epoch마다 평가 Loss 기록)
EPOCHS = 10
history = model.fit(x_train, y_train_oh,
                    epochs=EPOCHS, batch_size=128,
                    validation_data=(x_test, y_test_oh))
 
test_loss, test_acc = model.evaluate(x_test, y_test_oh, verbose=0)
print(f"테스트 loss: {test_loss:.4f}, 테스트 정확도: {test_acc:.4f}")
 
# 4. 훈련/평가 Epoch별 Loss 시각화
epochs = range(1, EPOCHS + 1)
plt.figure(figsize=(7, 4))
plt.plot(epochs, history.history["loss"], "bo-", label="Training loss")
plt.plot(epochs, history.history["val_loss"], "r^-", label="Test loss")
plt.title("MNIST - Training / Test Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("mnist_loss.png", dpi=120)
plt.show()
 
# 5. 잘못 구분한 이미지 표현
pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
wrong_idx = np.where(pred != y_test)[0]
print(f"잘못 구분한 이미지 수: {len(wrong_idx)} / {len(y_test)}")
 
n_show = min(25, len(wrong_idx))
plt.figure(figsize=(10, 10))
for i in range(n_show):
    idx = wrong_idx[i]
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
    plt.title(f"true:{y_test[idx]} / pred:{pred[idx]}", color="red", fontsize=10)
    plt.axis("off")
plt.suptitle("MNIST - Misclassified Images")
plt.tight_layout()
plt.savefig("mnist_wrong.png", dpi=120)
plt.show()
 