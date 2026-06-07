from ultralytics import YOLO

# 1. Carrega o cérebro da IA (versão nano, super leve)
model = YOLO("yolo26n.pt")

# 2. Liga a webcam e mostra o resultado ao vivo na tela
results = model.predict(source="0", show=True)