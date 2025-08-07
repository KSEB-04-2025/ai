import os
import argparse
from PIL import Image

# 🟡 명령줄 인자 받기
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="원본 이미지 폴더 경로")
parser.add_argument("--output_dir", default="output_images", help="결과 저장 폴더 경로")
args = parser.parse_args()

# 📁 폴더 만들기
os.makedirs(args.output_dir, exist_ok=True)

# 이미지 확장자 목록
img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
counter = 1

# 이미지 파일 처리
for fname in sorted(os.listdir(args.input_dir)):
    if not any(fname.lower().endswith(ext) for ext in img_exts):
        continue

    path = os.path.join(args.input_dir, fname)
    img = Image.open(path)
    w, h = img.size
    mid = w // 2

    # 좌우 반 잘라서 저장
    left_img = img.crop((0, 0, mid, h))
    right_img = img.crop((mid, 0, w, h))

    left_name = f"scratch{counter:03}.jpg"
    counter += 1
    right_name = f"scratch{counter:03}.jpg"
    counter += 1

    left_img.save(os.path.join(args.output_dir, left_name))
    right_img.save(os.path.join(args.output_dir, right_name))

print("✅ 좌우 반 잘라 저장 완료!")
