import cv2
from ultralytics import YOLO
import numpy as np

# Charger le modèle
model = YOLO('C:\\Users\\EEIA\\Desktop\\Morpion\\best.pt')  

# Ouvrir la caméra
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Impossible de lire la frame")
        break

    # Détection d'objets
    results = model(frame)
    annotated_frame = results[0].plot()
    
    # Stocker les positions des classes détectées
    positions = {"black": [], "white": []}
    
    for r in results[0].boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])  # Coordonnées de la boîte
        label = results[0].names[int(r.cls[0])]  # Nom de la classe
        
        if label in positions:
            positions[label].append(((x1 + x2) // 2, (y1 + y2) // 2))  # Centre du rectangle

    # Vérifier les conditions de victoire
    def check_victory(points):
        if len(points) < 3:
            return None
        
        # Vérifier les alignements
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                for k in range(j + 1, len(points)):
                    p1, p2, p3 = points[i], points[j], points[k]
                    
                    # Vérification des alignements (horizontaux, verticaux, diagonaux)
                    if abs(p1[0] - p2[0]) < 20 and abs(p2[0] - p3[0]) < 20:  # Vertical
                        return [p1, p2, p3]
                    elif abs(p1[1] - p2[1]) < 20 and abs(p2[1] - p3[1]) < 20:  # Horizontal
                        return [p1, p2, p3]
                    elif abs((p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-6) - (p3[1] - p2[1]) / (p3[0] - p2[0] + 1e-6)) < 0.1:  # Diagonal
                        return [p1, p2, p3]
        return None
    
    for label in ["black", "white"]:
        winning_points = check_victory(positions[label])
        if winning_points:
            cv2.line(annotated_frame, winning_points[0], winning_points[2], (0, 255, 0), 5)
            cv2.putText(annotated_frame, "VICTOIRE!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            break

    # Affichage de la détection
    cv2.imshow('YOLO Morpion', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
