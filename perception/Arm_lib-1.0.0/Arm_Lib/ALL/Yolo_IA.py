import random
import time
import cv2
from ultralytics import YOLO
from Arm_Lib import Arm_Device

# Initialisation des variables globales
board = [0] * 9  # 0 = vide, 1 = humain "X", 2 = IA "O"
human_pieces = 3  # Pions restants à poser pour l'humain
ai_pieces = 3     # Pions restants à poser pour l'IA

# Combinaisons gagnantes (lignes, colonnes, diagonales)
winning_combinations = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Lignes
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Colonnes
    [0, 4, 8], [2, 4, 6]              # Diagonales
]

# Initialisation du bras et de la caméra
arm = Arm_Device()
model = YOLO("/home/pi/Desktop/Arm_lib-1.0.0/Arm_Lib/best.pt")
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra")
    exit()

# Angles prédéfinis pour chaque position (à calibrer selon votre plateau physique)
positions_angles = {
    "pile": (90, 100, 0, 0, 90, 145, 500),  # Position de la pile (à calibrer)
    0: (74, 73, 0, 51, 90, 145, 500),   # Case 0
    1: (89, 73, 0, 48, 90, 145, 500),   # Case 1
    2: (103, 73, 0, 51, 90, 145, 500),  # Case 2
    3: (79, 25, 76, 33, 89, 145, 500),  # Case 3
    4: (100, 25, 76, 33, 89, 145, 500), # Case 4
    5: (89, 25, 76, 33, 89, 145, 500),  # Case 5
    6: (76, 39, 58, 25, 89, 145, 500),  # Case 6
    7: (102, 39, 58, 25, 89, 145, 500), # Case 7
    8: (89, 39, 58, 23, 89, 145, 500)   # Case 8
}

# Afficher le plateau
def print_board():
    symbols = {0: ".", 1: "X", 2: "O"}
    for i in range(0, 9, 3):
        print(f"{symbols[board[i]]} {symbols[board[i+1]]} {symbols[board[i+2]]}")
    print()

# Vérifier si quelqu’un a gagné
def check_winner(player):
    for combo in winning_combinations:
        if all(board[pos] == player for pos in combo):
            return True
    return False

# Vérifier les positions adjacentes pour la phase de déplacement
def get_adjacent(pos):
    adj = []
    if pos % 3 > 0: adj.append(pos - 1)      # Gauche
    if pos % 3 < 2: adj.append(pos + 1)      # Droite
    if pos >= 3: adj.append(pos - 3)         # Haut
    if pos <= 5: adj.append(pos + 3)         # Bas
    if pos % 3 > 0 and pos >= 3: adj.append(pos - 4)  # Diagonale haut-gauche
    if pos % 3 < 2 and pos <= 5: adj.append(pos + 4)  # Diagonale bas-droite
    if pos % 3 < 2 and pos >= 3: adj.append(pos - 2)  # Diagonale haut-droite
    if pos % 3 > 0 and pos <= 5: adj.append(pos + 2)  # Diagonale bas-gauche
    return [p for p in adj if board[p] == 0]

# Heuristique pour évaluer un état non terminal
def evaluate_board():
    ai_score = 0
    human_score = 0
    for combo in winning_combinations:
        ai_count = sum(1 for pos in combo if board[pos] == 2)
        human_count = sum(1 for pos in combo if board[pos] == 1)
        empty_count = sum(1 for pos in combo if board[pos] == 0)
        if ai_count == 3: return 100
        if human_count == 3: return -100
        if ai_count == 2 and empty_count == 1: ai_score += 10
        if human_count == 2 and empty_count == 1: human_score += 10
        if ai_count == 1 and empty_count == 2: ai_score += 1
        if human_count == 1 and empty_count == 2: human_score += 1
    return ai_score - human_score

# Mise à jour du plateau à partir de la caméra
def update_board_from_camera():
    global board
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de capturer une nouvelle frame")
        return False
    
    # Effectuer une détection sur la nouvelle frame
    results = model(frame)                       
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO Morpion', annotated_frame)
    cv2.waitKey(1)  # Forcer la mise à jour de la fenêtre
    
    # Réinitialiser le plateau
    new_board = [0] * 9
    human_count = 0
    ai_count = 0
    
    # Traiter les détections
    for detection in results[0].boxes:
        x, y = detection.xyxy[0][:2].cpu().numpy()
        label = int(detection.cls.cpu().numpy())
        print(f"Detection at x:{x}, y:{y}, label:{label}")
        col = int(x // (frame.shape[1] / 3))
        row = int(y // (frame.shape[0] / 3))
        pos = row * 3 + col
        if 0 <= pos < 9:
            if label == 1 and human_count < 3:  # "X" pour humain
                new_board[pos] = 1
                human_count += 1
            elif label == 2 and ai_count < 3:   # "O" pour IA
                new_board[pos] = 2
                ai_count += 1
    
    print(f"Pions détectés - Humain: {human_count}, IA: {ai_count}")
    board = new_board
    return True

# Déplacer le bras robotique
def move_arm(old_pos, new_pos):
    if old_pos not in positions_angles or new_pos not in positions_angles:
        print(f"Erreur : Position invalide ({old_pos} ou {new_pos})")
        return
    arm.Arm_serial_servo_write6(*positions_angles[old_pos])
    time.sleep(1)
    arm.Arm_serial_servo_write(6, 175, 500)  # Fermer la pince
    time.sleep(0.5)
    arm.Arm_serial_servo_write6(*positions_angles[new_pos])
    time.sleep(1)
    arm.Arm_serial_servo_write(6, 145, 500)  # Ouvrir la pince
    time.sleep(0.5)
    arm.Arm_serial_servo_write6(86, 118, 0, 0, 90, 114, 500)

# Minimax avec Alpha-Bêta pour la phase de pose
def minimax_pose(board, depth, is_maximizing, human_pieces, ai_pieces, alpha, beta):
    if check_winner(2): return 10 - depth
    if check_winner(1): return -10 + depth
    if human_pieces == 0 and ai_pieces == 0: return 0

    if is_maximizing:
        best_score = -float("inf")
        for pos in range(9):
            if board[pos] == 0:
                board[pos] = 2
                score = minimax_pose(board, depth + 1, False, human_pieces, ai_pieces - 1, alpha, beta)
                board[pos] = 0
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)
                if beta <= alpha: break
        return best_score
    else:
        best_score = float("inf")
        for pos in range(9):
            if board[pos] == 0:
                board[pos] = 1
                score = minimax_pose(board, depth + 1, True, human_pieces - 1, ai_pieces, alpha, beta)
                board[pos] = 0
                best_score = min(score, best_score)
                beta = min(beta, best_score)
                if beta <= alpha: break
        return best_score

# Minimax avec Alpha-Bêta pour la phase de déplacement
def minimax_move(board, depth, max_depth, is_maximizing, alpha, beta):
    if check_winner(2): return 10 - depth
    if check_winner(1): return -10 + depth
    if depth >= max_depth: return evaluate_board()

    if is_maximizing:
        best_score = -float("inf")
        ai_positions = [i for i in range(9) if board[i] == 2]
        for old_pos in ai_positions:
            adj = get_adjacent(old_pos)
            for new_pos in adj:
                board[old_pos] = 0
                board[new_pos] = 2
                score = minimax_move(board, depth + 1, max_depth, False, alpha, beta)
                board[old_pos] = 2
                board[new_pos] = 0
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)
                if beta <= alpha: break
        return best_score
    else:
        best_score = float("inf")
        human_positions = [i for i in range(9) if board[i] == 1]
        for old_pos in human_positions:
            adj = get_adjacent(old_pos)
            for new_pos in adj:
                board[old_pos] = 0
                board[new_pos] = 1
                score = minimax_move(board, depth + 1, max_depth, True, alpha, beta)
                board[old_pos] = 1
                board[new_pos] = 0
                best_score = min(score, best_score)
                beta = min(beta, best_score)
                if beta <= alpha: break
        return best_score

# Tour de l’humain (phase de pose)
def human_turn_pose():
    global human_pieces
    print("Posez votre pion manuellement sur le plateau, puis appuyez sur Entrée.")
    input()
    if not update_board_from_camera():
        print("Erreur caméra, veuillez réessayer.")
        return
    human_pieces -= 1
    print("Plateau mis à jour :")
    print_board()

# Tour de l’IA (phase de pose)
def ai_turn_pose():
    global ai_pieces
    if not update_board_from_camera():
        print("Erreur caméra, tour sauté")
        return
    best_score = -float("inf")
    best_pos = None
    for pos in range(9):
        if board[pos] == 0:
            board[pos] = 2
            score = minimax_pose(board, 0, False, human_pieces, ai_pieces - 1, -float("inf"), float("inf"))
            board[pos] = 0
            if score > best_score:
                best_score = score
                best_pos = pos
    if best_pos is not None:
        move_arm("pile", best_pos)
        board[best_pos] = 2
        ai_pieces -= 1
        print(f"L’IA pose un pion en {best_pos}")
    else:
        print("Aucune position valide trouvée pour poser un pion !")

# Tour de l’humain (phase de déplacement)
def human_turn_move():
    print("Déplacez votre pion manuellement sur le plateau, puis appuyez sur Entrée.")
    input()
    if not update_board_from_camera():
        print("Erreur caméra, veuillez réessayer.")
        return
    print("Plateau mis à jour :")
    print_board()

# Tour de l’IA (phase de déplacement)
def ai_turn_move():
    if not update_board_from_camera():
        print("Erreur caméra, tour sauté")
        return
    best_score = -float("inf")
    best_move = None
    max_depth = 4
    ai_positions = [i for i in range(9) if board[i] == 2]
    for old_pos in ai_positions:
        adj = get_adjacent(old_pos)
        for new_pos in adj:
            board[old_pos] = 0
            board[new_pos] = 2
            score = minimax_move(board, 0, max_depth, False, -float("inf"), float("inf"))
            board[old_pos] = 2
            board[new_pos] = 0
            if score > best_score:
                best_score = score
                best_move = (old_pos, new_pos)
    if best_move is None:
        print("L’IA ne peut pas bouger ! Match nul ou erreur.")
        return
    old_pos, new_pos = best_move
    move_arm(old_pos, new_pos)
    board[old_pos] = 0
    board[new_pos] = 2
    print(f"L’IA déplace un pion de {old_pos} à {new_pos}")

# Boucle principale du jeu
def play_game():
    print("Début du jeu physique avec le Dofbot !")
    print_board()
    
    # Phase de pose
    while human_pieces > 0 or ai_pieces > 0:
        if human_pieces > 0:
            human_turn_pose()
            if check_winner(1):
                print("Vous avez gagné !")
                return
        if ai_pieces > 0:
            ai_turn_pose()
            print_board()
            if check_winner(2):
                print("L’IA a gagné !")
                return
    
    # Phase de déplacement
    print("Phase de déplacement commencée !")
    while True:
        human_turn_move()
        if check_winner(1):
            print("Vous avez gagné !")
            return
        ai_turn_move()
        print_board()
        if check_winner(2):
            print("L’IA a gagné !")
            return

# Lancer le jeu
if __name__ == "__main__":
    try:
        arm.Arm_serial_servo_write6(86, 118, 0, 0, 90, 114, 500)
        play_game()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        arm.Arm_serial_servo_write6(86, 118, 0, 0, 90, 114, 500)