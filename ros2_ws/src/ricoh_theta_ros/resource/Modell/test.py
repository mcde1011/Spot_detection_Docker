from ultralytics import YOLO

# Charger le modèle
model = YOLO('best.pt')

# Tester sur l'image
results = model("extincteur.jpeg", save=True)

