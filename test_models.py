"""
모델 로딩 테스트
YOLO와 Keras 모델이 제대로 로드되는지 확인
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import logging
logging.basicConfig(level=logging.INFO)

print("\n" + "="*60)
print("🔧 모델 로딩 테스트")
print("="*60 + "\n")

# 작업 디렉토리 확인
cwd = os.getcwd()
print(f"📁 작업 디렉토리: {cwd}")

# 모델 파일 확인
models = {
    'YOLO Seg': 'yolo11n-bed-seg.pt',
    'YOLO Pose': 'yolo11m-pose.pt',
    'Keras 6-class': 'my_model_six_check.keras'
}

print("\n✓ 모델 파일 확인:")
for name, path in models.items():
    exists = os.path.exists(path)
    size = os.path.getsize(path) / (1024*1024) if exists else 0
    status = "✅" if exists else "❌"
    print(f"  {status} {name:20s} {path:25s} ({size:.1f} MB)")

# YOLO 모델 로드 테스트
print("\n⏳ YOLO 모델 로딩 중...")
try:
    from ultralytics import YOLO
    seg_model = YOLO('yolo11n-bed-seg.pt')
    print("  ✅ YOLO Seg 로드 완료")
except Exception as e:
    print(f"  ❌ YOLO Seg 로드 실패: {e}")

try:
    pose_model = YOLO('yolo11m-pose.pt')
    print("  ✅ YOLO Pose 로드 완료")
except Exception as e:
    print(f"  ❌ YOLO Pose 로드 실패: {e}")

# Keras 모델 로드 테스트
print("\n⏳ Keras 모델 로딩 중...")
try:
    import tensorflow as tf
    import keras
    tf.config.set_visible_devices([], 'GPU')
    
    keras_clf = keras.models.load_model('my_model_six_check.keras')
    print("  ✅ Keras 6-class 로드 완료")
    print(f"     - Input shape: {keras_clf.input_shape}")
    print(f"     - Output shape: {keras_clf.output_shape}")
except Exception as e:
    print(f"  ❌ Keras 로드 실패: {e}")

print("\n" + "="*60)
print("✨ 모든 모델 로드 테스트 완료!")
print("="*60 + "\n")
