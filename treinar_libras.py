from ultralytics import YOLO
from roboflow import Roboflow

# 1. Conecta ao Roboflow e baixa o banco de dados que criamos
rf = Roboflow(api_key="7Uwr7I1FNyJlnMSfTRTr") 
project = rf.workspace("drubio-imoveis").project("alfabeto_libras_webcam")
version = project.version(1)
dataset = version.download("yolov8")

# 2. Carrega a arquitetura YOLO para iniciar o aprendizado
model = YOLO("yolo11n.pt") 

# 3. Executa o treinamento por 50 épocas
model.train(data=f"{dataset.location}/data.yaml", epochs=50, imgsz=640)




                