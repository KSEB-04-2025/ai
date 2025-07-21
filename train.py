from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')  

    model.train(
        data='dataset/defect.yaml',
        epochs=30,                
        imgsz=1024,               
        batch=8,                  
        name='defect_model_s',    
        project='runs/train',
        device=0,
        lr0=0.0005,
        optimizer='Adam',
        warmup_epochs=3,
        close_mosaic=10,
        pretrained=True
    )

if __name__ == '__main__':
    main()
