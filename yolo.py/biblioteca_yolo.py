from ultralytics import YOLO

# Carregamos o modelo (cérebro)
modelo = YOLO("yolov8n.pt") 


resultados = modelo.predict(source="0",show=True)

