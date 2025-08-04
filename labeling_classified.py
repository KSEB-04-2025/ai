"""
scratch 라벨용 bounding-box 검출 및 YOLOv8 라벨(txt) 자동 생성
"""

import cv2
import numpy as np
from pathlib import Path

# ───────── 1) 스크래치 엣지 기반 bounding-box 검출 ─────────
def detect_scratch_bboxes(
    img_bgr,
    low_thr: int = 50,
    high_thr: int = 150,
    min_length: int = 20,
    dilate_iter: int = 1
):
    """
    Canny 엣지 → 팽창 → 외곽선 검출 → boundingRect → 필터링
    - low_thr/high_thr: Canny 임계값
    - min_length: w 또는 h 중 하나가 이 이상일 때만 box 생성
    - dilate_iter: 엣지 팽창 횟수
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_thr, high_thr)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=dilate_iter)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if max(w, h) >= min_length:
            bboxes.append((x, y, w, h))
    return bboxes


# ───────── 2) YOLOv8 라벨(txt) 저장 ─────────
def save_bboxes_to_yolo_txt(
    img_path: str,
    bboxes: list[tuple[int,int,int,int]],
    class_id: int = 0,
    out_dir: str = "labels"
):
    """
    bboxes: list of (x, y, w, h) in 픽셀 단위
    YOLO 포맷: class x_center y_center width height (모두 normalized)
    """
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    Path(out_dir).mkdir(exist_ok=True)
    lines = []
    for (x, y, bw, bh) in bboxes:
        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

    txt_path = Path(out_dir) / (Path(img_path).stem + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ───────── 3) 단일 이미지 처리 ─────────
def process_image(
    img_path: str,
    out_vis_dir: str = "output",
    out_label_dir: str = "labels",
    # scratch 검출 파라미터
    low_thr: int = 80,
    high_thr: int = 200,
    min_length: int = 20,
    dilate_iter: int = 1
):
    img = cv2.imread(str(img_path))
    vis = img.copy()

    # 1) scratch bounding-box 검출
    bboxes = detect_scratch_bboxes(
        img,
        low_thr=low_thr,
        high_thr=high_thr,
        min_length=min_length,
        dilate_iter=dilate_iter
    )

    # 2) bounding-box 시각화 (녹색)
    for x, y, bw, bh in bboxes:
        tl = (x, y)
        br = (x + bw, y + bh)
        cv2.rectangle(vis, tl, br, (0, 255, 0), 2)

    # 3) 시각화 이미지 저장
    Path(out_vis_dir).mkdir(exist_ok=True)
    vis_path = Path(out_vis_dir) / Path(img_path).name
    cv2.imwrite(str(vis_path), vis)

    # 4) YOLOv8 라벨(txt) 저장
    save_bboxes_to_yolo_txt(
        img_path,
        bboxes,
        class_id=0,
        out_dir=out_label_dir
    )


# ───────── 4) 배치 처리 ─────────
if __name__ == "__main__":
    src_dir = "labeling_test_images"   # 테스트 이미지 폴더
    for img_path in Path(src_dir).glob("*.[jp][pn]g"):
        process_image(str(img_path))
    print("처리 완료! 'output/' (시각화)와 'labels/' (txt) 폴더를 확인하세요.")
