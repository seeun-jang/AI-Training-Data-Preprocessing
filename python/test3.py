import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense


# =========================
# 1. Training 데이터 불러오기
# =========================

all_files = []

for i in range(0, 10):
    path_dir = './images/training/{0}'.format(i)

    file_list = os.listdir(path_dir)
    file_list.sort()

    all_files.append(file_list)


x_train_datas = []
y_train_datas = []


for num in range(0, 10):

    for numbers in all_files[num]:

        img_path = './images/training/{0}/{1}'.format(num, numbers)

        print("load : " + img_path)

        img = Image.open(img_path)

        # 픽셀값 0~255 → -1~1
        imgarr = np.array(img).astype(np.float32)
        imgarr = imgarr / 127.5 - 1.0

        # 28x28 → 784x1
        x_train_datas.append(
            np.reshape(imgarr, (784, 1))
        )

        # One-Hot Encoding
        y_tmp = np.zeros(shape=(10))
        y_tmp[num] = 1

        y_train_datas.append(y_tmp)


print(len(x_train_datas))
print(len(y_train_datas))


# =========================
# 2. Testing 데이터 불러오기
# =========================

eval_files = []

for i in range(0, 10):
    path_dir = './images/testing/{0}'.format(i)

    file_list = os.listdir(path_dir)
    file_list.sort()

    eval_files.append(file_list)


x_test_datas = []
y_test_datas = []


for num in range(0, 10):

    for numbers in eval_files[num]:

        img_path = './images/testing/{0}/{1}'.format(num, numbers)

        print("load : " + img_path)

        img = Image.open(img_path)

        # 픽셀값 0~255 → -1~1
        imgarr = np.array(img).astype(np.float32)
        imgarr = imgarr / 127.5 - 1.0

        # 28x28 → 784x1
        x_test_datas.append(
            np.reshape(imgarr, (784, 1))
        )

        # One-Hot Encoding
        y_tmp = np.zeros(shape=(10))
        y_tmp[num] = 1

        y_test_datas.append(y_tmp)


print(len(x_test_datas))
print(len(y_test_datas))


# =========================
# 3. NumPy 배열 형태 변경
# =========================

x_train_datas = np.reshape(
    x_train_datas,
    (-1, 784)
)

y_train_datas = np.reshape(
    y_train_datas,
    (-1, 10)
)

x_test_datas = np.reshape(
    x_test_datas,
    (-1, 784)
)

y_test_datas = np.reshape(
    y_test_datas,
    (-1, 10)
)


# =========================
# 4. 신경망 모델 생성
# =========================

input = tf.keras.Input(
    shape=(784,),
    name="Input"
)

hidden1 = Dense(
    512,
    activation="relu",
    name="Hidden1"
)(input)

hidden2 = Dense(
    256,
    activation="relu",
    name="Hidden2"
)(hidden1)

output = Dense(
    10,
    activation="softmax",
    name="Output"
)(hidden2)


model = tf.keras.Model(
    inputs=[input],
    outputs=[output]
)


# =========================
# 5. Optimizer 설정
# =========================

opt = keras.optimizers.Adam(
    learning_rate=0.001
)


model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)


model.summary()


# =========================
# 6. 학습
# =========================

model.fit(
    x_train_datas,
    y_train_datas,
    epochs=10,
    shuffle=True
)


# =========================
# 7. 테스트
# =========================

test_loss, test_acc = model.evaluate(
    x_test_datas,
    y_test_datas
)


print("테스트 손실:", test_loss)
print("테스트 정확도:", test_acc)
