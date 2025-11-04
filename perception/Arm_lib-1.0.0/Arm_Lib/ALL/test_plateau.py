import time
import cv2
from ultralytics import YOLO

# Initialisation du plateau (3x3)
board = [0] * 9  # 0 = vide, 1 = humain "X" (white), 2 = IA "O" (black)

# Initialisation du modèle YOLO et de la caméra
model = YOLO("/home/pi/Desktop/Arm_lib-1.0.0/Arm_Lib/best.pt")
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra")
    exit()

# Fonction pour afficher le plateau
def print_board():
    symbols = {0: ".", 1: "X", 2: "O"}
    for i in range(0, 9, 3):
        print(f"{symbols[board[i]]} {symbols[board[i+1]]} {symbols[board[i+2]]}")
    print()

# Fonction pour mettre à jour le plateau à partir de la caméra
def update_board_from_camera():
    global board
    
    # Capturer plusieurs frames pour vider le buffer
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de capturer une nouvelle frame")
            return False
    
    # Détection avec YOLO
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO Morpion', annotated_frame)
    cv2.waitKey(100)  # Temps pour rafraîchir l’affichage
    
    # Réinitialiser le plateau pour refléter uniquement ce que la caméra voit
    new_board = [0] * 9
    human_count = 0
    ai_count = 0
    
    frame_height, frame_width = frame.shape[:2]
    grid_width = frame_width / 3
    grid_height = frame_height / 3
    
    for detection in results[0].boxes:
        confidence = detection.conf.cpu().numpy()[0]
        if confidence < 0.7:  # Seuil de confiance
            continue
        x, y = detection.xyxy[0][:2].cpu().numpy()
        label = model.names[int(detection.cls.cpu().numpy()[0])]
        col = int(x // grid_width)
        row = int(y // grid_height)
        pos = row * 3 + col
        if 0 <= pos < 9:
            if label == "white" and human_count < 3:
                new_board[pos] = 1  # "X" pour humain
                human_count += 1
            elif label == "black" and ai_count < 3:
                new_board[pos] = 2  # "O" pour IA
                ai_count += 1
    
    print(f"Pions détectés - Humain: {human_count}, IA: {ai_count}")
    print(f"Nouvel état détecté : {new_board}")
    board = new_board
    return True

# Boucle principale pour tester la détection
def test_detection():
    print("Début du test de détection !")
    print_board()
    
    while True:
        print("Posez un pion sur le plateau, puis appuyez sur Entrée pour valider.")
        input()  # Attendre la validation de l'utilisateur
        
        if not update_board_from_camera():
            print("Erreur caméra, réessayez.")
            continue
        
        print("État du plateau après détection :")
        print_board()
        
        # Option pour quitter
        print("Appuyez sur 'q' puis Entrée pour quitter, ou juste Entrée pour continuer.")
        if input().lower() == 'q':
            break

# Lancer le test
if __name__ == "__main__":
    try:
        test_detection()
    finally:
        cap.release()
        cv2.destroyAllWindows()