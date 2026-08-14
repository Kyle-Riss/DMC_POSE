import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 올바른 패치 위치
import keras.src.layers.reshaping.flatten as flatten_module
original_compute = flatten_module.Flatten.compute_output_spec

def patched_compute(self, inputs):
    if isinstance(inputs, list):
        inputs = inputs[0]
    return original_compute(self, inputs)

flatten_module.Flatten.compute_output_spec = patched_compute

import tensorflow as tf
import subprocess
import onnxruntime as ort
import numpy as np

model = tf.keras.models.load_model(
    "z_image_classification_model_vgg16_1.h5",
    compile=False
)
print("모델 로드 완료")
print("입력 shape:", model.input_shape)
print("출력 shape:", model.output_shape)

model.export("vgg16_saved_model")
print("SavedModel 저장 완료")

result = subprocess.run(
    [
        "python", "-m", "tf2onnx.convert",
        "--saved-model", "vgg16_saved_model",
        "--output", "vgg16.onnx",
        "--opset", "13"
    ],
    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    capture_output=True,
    text=True
)

print(result.stdout)
if result.returncode != 0:
    print("에러 발생:")
    print(result.stderr)
else:
    print("ONNX 변환 완료: vgg16.onnx")

sess = ort.InferenceSession("vgg16.onnx")
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
print("\n[ONNX 검증]")
print("입력 이름:", input_name)
print("입력 shape:", input_shape)

shape = [1 if str(d).startswith("unk") else d for d in input_shape]
dummy = np.random.randn(*shape).astype(np.float32)
output = sess.run(None, {input_name: dummy})
print("출력 shape:", output[0].shape)
print("검증 완료!")
