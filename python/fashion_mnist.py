# TensorFlow and tf.keras
import tensorflow as tf

# helper libraries
import numpy as np
from matplotlib import pyplot as plt


# ========================================
# 1. Fashion MNIST 데이터 불러오기
# ========================================

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = \
    fashion_mnist.load_data()


class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]


# ========================================
# 2. 데이터 확인
# ========================================

print(train_images.shape)
print(len(train_labels))
print(train_labels)

print(test_images.shape)
print(len(test_labels))
print(test_labels)


# ========================================
# 3. 원본 이미지 확인
# ========================================

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# ========================================
# 4. 0 ~ 1 사이 값으로 정규화
# ========================================

train_images = train_images / 255.0
test_images = test_images / 255.0


# ========================================
# 5. 정규화된 이미지 확인
# ========================================

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# ========================================
# 6. 학습 이미지 25개 확인
# ========================================

plt.figure(figsize=(10, 10))

for i in range(25):

    plt.subplot(5, 5, i + 1)

    plt.xticks([])
    plt.yticks([])

    plt.grid(False)

    plt.imshow(
        train_images[i],
        cmap=plt.cm.binary
    )

    plt.xlabel(
        class_names[train_labels[i]]
    )

plt.show()


# ========================================
# 7. 신경망 모델 생성
# ========================================

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)
])


# ========================================
# 8. 모델 컴파일
# ========================================

model.compile(
    optimizer='adam',

    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),

    metrics=['accuracy']
)


# ========================================
# 9. 모델 학습
# ========================================

model.fit(
    train_images,
    train_labels,
    epochs=10
)


# ========================================
# 10. 테스트 데이터 평가
# ========================================

test_loss, test_acc = model.evaluate(
    test_images,
    test_labels,
    verbose=2
)

print('\nTest accuracy:', test_acc)


# ========================================
# 11. Softmax를 이용한 확률 모델 생성
# ========================================

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])


# ========================================
# 12. 테스트 이미지 예측
# ========================================

predictions = probability_model.predict(
    test_images
)


print(predictions[0])


# ========================================
# 13. 첫 번째 이미지 확률 합 확인
# ========================================

sum = 0

for pred in predictions[0]:
    sum += pred

print(sum)


# ========================================
# 14. 첫 번째 이미지 예측 번호
# ========================================

print(
    np.argmax(predictions[0])
)


# ========================================
# 15. 첫 번째 이미지 예측 이름
# ========================================

print(
    class_names[
        np.argmax(predictions[0])
    ]
)


# ========================================
# 16. 예측 이미지 출력 함수
# ========================================

def plot_image(
        i,
        predictions_array,
        true_label,
        img):

    true_label = true_label[i]
    img = img[i]

    plt.grid(False)

    plt.xticks([])
    plt.yticks([])

    plt.imshow(
        img,
        cmap=plt.cm.binary
    )

    predicted_label = np.argmax(
        predictions_array
    )

    # 예측이 맞으면 파란색
    if predicted_label == true_label:
        color = 'blue'

    # 예측이 틀리면 빨간색
    else:
        color = 'red'

    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            class_names[predicted_label],
            100 * np.max(predictions_array),
            class_names[true_label]
        ),
        color=color
    )


# ========================================
# 17. 예측 확률 막대그래프 함수
# ========================================

def plot_value_array(
        i,
        predictions_array,
        true_label):

    true_label = true_label[i]

    plt.grid(False)

    plt.xticks(range(10))
    plt.yticks([])

    thisplot = plt.bar(
        range(10),
        predictions_array,
        color="#777777"
    )

    plt.ylim([0, 1])

    predicted_label = np.argmax(
        predictions_array
    )

    # 예측한 값
    thisplot[predicted_label].set_color(
        'red'
    )

    # 실제 정답
    thisplot[true_label].set_color(
        'blue'
    )


# ========================================
# 18. 하나의 이미지 예측 확인
# ========================================

i = 12

plt.figure(
    figsize=(6, 3)
)

plt.subplot(
    1,
    2,
    1
)

plot_image(
    i,
    predictions[i],
    test_labels,
    test_images
)

plt.subplot(
    1,
    2,
    2
)

plot_value_array(
    i,
    predictions[i],
    test_labels
)

plt.show()


# ========================================
# 19. 여러 이미지 예측 결과 확인
# ========================================

num_rows = 5
num_cols = 3

num_images = (
    num_rows * num_cols
)

plt.figure(
    figsize=(
        2 * 2 * num_cols,
        2 * num_rows
    )
)


for i in range(num_images):

    # 이미지
    plt.subplot(
        num_rows,
        2 * num_cols,
        2 * i + 1
    )

    plot_image(
        i,
        predictions[i],
        test_labels,
        test_images
    )

    # 확률 막대그래프
    plt.subplot(
        num_rows,
        2 * num_cols,
        2 * i + 2
    )

    plot_value_array(
        i,
        predictions[i],
        test_labels
    )


plt.tight_layout()
plt.show()