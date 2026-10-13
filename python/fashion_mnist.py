
# 실습과제 2: Fashion MNIST - CNN 구현으로 분류
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 1. 데이터 준비 (CNN 입력: 28x28x1)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

y_train_oh = keras.utils.to_categorical(y_train, 10)
y_test_oh = keras.utils.to_categorical(y_test, 10)

# 2. CNN 모델 구성
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 3. 학습
EPOCHS = 10

history = model.fit(
    x_train,
    y_train_oh,
    epochs=EPOCHS,
    batch_size=128,
    validation_split=0.2
)

# 4. 평가
test_loss, test_acc = model.evaluate(
    x_test,
    y_test_oh,
    verbose=0
)

print(f"테스트 loss: {test_loss:.4f}, 테스트 정확도: {test_acc:.4f}")

# 5. 학습 곡선 (Loss / Accuracy)
epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, history.history["loss"], "bo-", label="Training loss")
plt.plot(epochs, history.history["val_loss"], "r^-", label="Validation loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs, history.history["accuracy"], "bo-", label="Training acc")
plt.plot(epochs, history.history["val_accuracy"], "r^-", label="Validation acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fashion_curves.png", dpi=120)

# plt.show() 제거 - 마지막에 두 창 동시에 출력

# 6. 예측 결과 예시 (맞으면 파랑, 틀리면 빨강)
pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

plt.figure(figsize=(12, 6))

for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")

    color = "blue" if pred[i] == y_test[i] else "red"

    plt.title(
        f"{class_names[pred[i]]}\n(true: {class_names[y_test[i]]})",
        color=color,
        fontsize=9
    )

    plt.axis("off")

plt.suptitle(f"Fashion MNIST CNN - test acc {test_acc:.4f}")
plt.tight_layout()
plt.savefig("fashion_pred.png", dpi=120)

# 두 그래프 창 동시에 출력
plt.show()
