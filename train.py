import os
# [중요] GPU를 물리적으로 숨겨서 CUDA 에러를 원천 차단합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras import layers, Sequential
import argparse
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# 텐서플로가 CPU만 사용하도록 명시적 설정
tf.config.set_visible_devices([], 'GPU')

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True, help="path to csv Data")
ap.add_argument("-o", "--save", type=str, required=True, help="path to save .h5 model")

args = vars(ap.parse_args())
path_csv = args["dataset"]
path_to_save = args["save"]

# 1. 데이터 로드
if not os.path.exists(path_csv):
    print(f"[Error] 파일을 찾을 수 없습니다: {path_csv}")
    exit()

df = pd.read_csv(path_csv)

# X(좌표), Y(라벨) 분리
x = df.drop(columns=['Pose_Class'])
y = df['Pose_Class']

# 라벨 숫자로 인코딩 및 원핫 인코딩
y_factorized, y_categories = y.factorize()
class_number = len(y_categories)
y = keras.utils.to_categorical(y_factorized)

# 데이터 타입 변환 (안정성 확보)
x = x.astype('float32')

# 학습/검증 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(f'[INFO] 데이터 로드 완료: {len(df)} 샘플 / {class_number} 클래스')

# 2. 모델 정의 (CPU 환경에 적합한 가벼운 구조)
model = Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2), # 과적합 방지
    layers.Dense(128, activation='relu'),
    layers.Dense(class_number, activation="softmax")
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# 3. 콜백 설정
checkpoint = keras.callbacks.ModelCheckpoint(
    path_to_save, 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    mode='max'
)
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15)

# 4. 학습 시작
print('[INFO] CPU 모드로 학습을 시작합니다...')
history = model.fit(
    x_train, y_train,
    epochs=100, # CPU는 안정적이니 에포크를 조금 더 늘려도 됩니다.
    batch_size=64,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint, earlystopping]
)

# 5. 결과 시각화 및 저장
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.savefig('metrics.png')
print('[INFO] 학습 완료 및 metrics.png 저장됨')
