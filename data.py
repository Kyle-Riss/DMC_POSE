import os
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# 1. 모델 설정 (YOLO11 Medium Pose 모델 사용)
# 처음 실행 시 자동으로 .pt 파일을 다운로드합니다.
model = YOLO('yolo11m-pose.pt') 

# 2. 경로 설정
base_path = '/home/dmc/pose/data'
folders = [f'p{i:02d}' for i in range(1, 13)] # p01 ~ p12

data_list = []

print("[INFO] YOLO11m을 이용한 관절 데이터 추출을 시작합니다...")

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    if not os.path.exists(folder_path):
        print(f"[Warning] 폴더를 찾을 수 없습니다: {folder_path}")
        continue
    
    img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in tqdm(img_files, desc=f"처리 중: {folder}"):
        img_path = os.path.join(folder_path, img_name)
        
        # YOLO11 추론 (사람 검출 + 포즈 추정을 한 번에 수행)
        # verbose=False로 설정하여 터미널 로그를 깔끔하게 유지합니다.
        results = model.predict(img_path, device='cuda:0', verbose=False)

        # 결과가 있고, 사람이 최소 한 명 이상 검출되었을 때
        if len(results) > 0 and len(results[0].keypoints) > 0:
            # 첫 번째 사람(가장 확신도가 높은 사람)의 키포인트 추출
            # .xyn은 이미지 크기에 맞춰 정규화된(0~1) 좌표입니다. 
            # 기존 코드처럼 픽셀 좌표를 원하시면 .xy를 사용하세요.
            kpts_obj = results[0].keypoints[0] # 첫 번째 사람
            
            # (17, 2) 형태의 x, y 좌표 추출 후 1차원 배열로 평탄화 (34개)
            kpts = kpts_obj.xy[0].cpu().numpy().flatten()
            
            # 만약 관절이 17개가 다 안 잡혔을 경우를 대비해 크기 체크
            if len(kpts) == 34:
                # [라벨명, x1, y1, x2, y2...] 형태로 리스트 생성
                data_list.append(np.concatenate([[folder], kpts]))

# 3. 데이터프레임 생성 및 저장
columns = ['Pose_Class'] + [f'kpt_{i}' for i in range(34)]
df = pd.DataFrame(data_list, columns=columns)

output_path = '/home/dmc/pose/pose_dataset.csv'
df.to_csv(output_path, index=False)

print(f"\n--- 추출 완료 ---")
print(f"총 샘플 수: {len(df)}")
print(f"저장 경로: {output_path}")
