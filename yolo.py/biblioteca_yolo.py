from ultralytics import YOLO

# Carrega o cérebro da yolo treinada com as fotos carregadas no roboflow!
model = YOLO("runs/detect/train-2/weights/best.pt")


results = model.predict(source="0", show=True)