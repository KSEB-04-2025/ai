import argparse
from pathlib import Path
from ultralytics import YOLO


def predict_images(model_path: str, image_dir: str, out_dir: str = "runs/predict_results", name: str = "test_predictions", conf: float = 0.25, imgsz: int = 896, device: str = "cpu"):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
    
    if not Path(image_dir).exists():
        raise FileNotFoundError(f"❌ 이미지 또는 폴더를 찾을 수 없습니다: {image_dir}")
    
    model = YOLO(model_path)
    
    print(f"✅ 모델 로드 완료: {model_path}")
    print(f"📁 이미지 경로: {image_dir}")
    print(f"📦 결과 저장: {out_dir}/{name}")
    print(f"🚀 디바이스: {device}")

    results = model.predict(
        source=image_dir,
        save=True,
        conf=conf,
        imgsz=imgsz,
        device=device,  
        project=out_dir,
        name=name,
        exist_ok=True
    )
    
    print("🎯 예측 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Test Image Prediction")

    parser.add_argument("--model", type=str, required=True, help="Trained YOLO model path (.pt)")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to image or directory of images to predict")
    parser.add_argument("--out_dir", type=str, default="runs/predict_results", help="Output directory root")
    parser.add_argument("--name", type=str, default="test_predictions", help="Subfolder name for results")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=896, help="Image size (default: 896)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or 'cuda' or '0' (default: cpu)")

    args = parser.parse_args()

    predict_images(
        model_path=args.model,
        image_dir=args.img_dir,
        out_dir=args.out_dir,
        name=args.name,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device
    )
