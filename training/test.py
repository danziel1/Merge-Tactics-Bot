from ultralytics import YOLO


model = YOLO("./runs/detect/train/weights/best.pt")

results = model.predict(source="./test.png", conf=0.5)

results[0].show() 

# Access detections from the first image
boxes = results[0].boxes
for i, box in enumerate(boxes):
    xyxy = box.xyxy[0].tolist()        # [x1, y1, x2, y2]
    cls_id = int(box.cls[0])           # Class index
    conf = float(box.conf[0])          # Confidence score
    label = results[0].names[cls_id]   # Class name from model

    xMid = (xyxy[0] + xyxy[2])/2
    yMid = (xyxy[1] + xyxy[3])/2

    print(f"Detection {i+1}:")
    print(f"  Class: {label} (ID: {cls_id})")
    print(f"  Confidence: {conf:.2f}")
    print(f"  Coordinates: x1={xyxy[0]:.1f}, y1={xyxy[1]:.1f}, x2={xyxy[2]:.1f}, y2={xyxy[3]:.1f}")
    print()


    #   Coordinates: x1=436.7, y1=1357.9, x2=526.3, y2=1473.5
    # Coordinates: x1=365.4, y1=1224.4, x2=437.4, y2=1302.2
# more than 1375 is on bench