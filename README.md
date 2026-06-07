# Biblioteca Yolov8 para detecção de Objetos - Visão Computacional

Trata-se de um projeto de estudo de bibliotecas para a matéria de Linguagem de Programação II.

Atualmente, o projeto utiliza a arquitetura **YOLOv8** com pesos pré-treinados para realizar a detecção de objetos genéricos através da webcam, garantindo a integridade do código dentro de um ambiente virtual isolado (`venv`).

---

## Código de Validação (`biblioteca_yolo.py`)

O script principal carrega o modelo nano (leve) e ativa a captura de vídeo:

```python
from ultralytics import YOLO

# 1. Carrega o modelo genérico de fábrica (versão nano, super leve)
modelo = YOLO("yolov8n.pt") 

# 2. Liga a webcam e realiza a detecção ao vivo na tela
resultados = modelo.predict(source="0", show=True)

```

---

## Implementações Futuras: Reconhecimento de LIBRAS

O objetivo final deste projeto é aplicar **Transfer Learning** para criar um modelo customizado voltado à acessibilidade. As próximas etapas do desenvolvimento incluem:

* **Ampliação do Dataset:** Coleta de um lote robusto de imagens (de 30 a 50 fotos por caractere) capturadas diretamente pela webcam do notebook para evitar distorções de resolução.
* **Rotulagem via Roboflow:** Utilização da plataforma Roboflow para o poligonamento (*bounding boxes*) e tagueamento das classes das letras.
* **Treinamento Customizado:** Execução do treinamento por múltiplas épocas para gerar os pesos específicos (`best.pt`) focados no **reconhecimento do alfabeto manual de LIBRAS** (ex: saudações como "BOM DIA").

---

## Como Rodar o Projeto

1. Ative o seu ambiente virtual:
```bash
.\venv\Scripts\activate

```


2. Instale a biblioteca necessária:
```bash
pip install ultralytics

```


3. Execute o script:
```bash
python biblioteca_yolo.py

```



---



