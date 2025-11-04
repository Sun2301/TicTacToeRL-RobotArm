import random
import time
import cv2
from ultralytics import YOLO
from Arm_Lib import Arm_Device

# Initialisation des variables globales
board = [0] * 9  # 0 = vide, 1 = humain "X" (white), 2 = IA "O" (black)
human_pieces = 3
ai_pieces = 3

winning_combinations = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
]

arm = Arm_Device()
model = YOLO("/home/pi/Desktop/Arm_lib-1.0.0/Arm_Lib/best.pt")
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra")
    exit()

positions_angles = {
    "pile": (122, 74, 15, 20, 87, 135, 500),
    0: (99, 32, 52, 58, 90, 135, 500),
    1: (89, 32, 52, 58, 90, 135, 500),
    2: (79, 32, 52, 58, 90, 135, 500),
    3: (102, 46, 48, 31, 89, 135, 500),
    4: (89, 43, 48, 31, 89, 135, 500),
    5: (77, 46, 48, 31, 89, 135, 500),
    6: (105, 54, 44, 15, 88, 135, 500),
    7: (89, 56, 44, 10, 88, 135, 500),
    8: (73, 55, 44, 13, 88, 135, 500)
}

def print_board():
    symbols = {0: ".", 1: "X", 2: "O"}
    for i in range(0, 9, 3):
        print(f"{symbols[board[i]]} {symbols[board[i+1]]} {symbols[board[i+2]]}")
    print()

def check_winner(player):
    for combo in winning_combinations:
        if all(board[pos] == player for pos in combo):
            return True
    return False

def get_adjacent(pos):
    adj = []
    if pos % 3 > 0: adj.append(pos - 1)
    if pos % 3 < 2: adj.append(pos + 1)
    if pos >= 3: adj.append(pos - 3)
    if pos <= 5: adj.append(pos + 3)
    if pos % 3 > 0 and pos >= 3: adj.append(pos - 4)
    if pos % 3 < 2 and pos <= 5: adj.append(pos + 4)
    if pos % 3 < 2 and pos >= 3: adj.append(pos - 2)
    if pos % 3 > 0 and pos <= 5: adj.append(pos + 2)
    return [p for p in adj if board[p] == 0]

def evaluate_board():
    ai_score = 0
    human_score = 0
    for combo in winning_combinations:
        ai_count = sum(1 for pos in combo if board[pos] == 2)
        human_count = sum(1 for pos in combo if board[pos] == 1)
        empty_count = sum(1 for pos in combo if board[pos] == 0)
        #if ai_count == 3: return 100
        #if human_count == 3: return -100
        if ai_count == 2 and empty_count == 1: ai_score += 10
        if human_count == 2 and empty_count == 1: human_score += 10
        #if ai_count == 1 and empty_count == 2: ai_score += 1
        #if human_count == 1 and empty_count == 2: human_score += 1
    return ai_score - human_score

def update_board_from_camera(wait_for_stability=False):
    global board
    if wait_for_stability:
        time.sleep(10)  # Délai pour le bras
    
    # Capturer plusieurs frames pour vider le buffer
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de capturer une nouvelle frame")
            return False
    
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO Morpion', annotated_frame)
    cv2.waitKey(100)
    
    # Réinitialiser le plateau à partir de la détection
    new_board = [0] * 9
    human_count = 0
    ai_count = 0
    
    frame_height, frame_width = frame.shape[:2]
    grid_width = frame_width / 3
    grid_height = frame_height / 3
    
    for detection in results[0].boxes:
        confidence = detection.conf.cpu().numpy()[0]
        if confidence < 0.7:
            continue
        x, y = detection.xyxy[0][:2].cpu().numpy()
        label = model.names[int(detection.cls.cpu().numpy()[0])]
        col = int(x // grid_width)
        row = int(y // grid_height)
        pos = row * 3 + col
        if 0 <= pos < 9:
            if label == "white" and human_count < 3:
                new_board[pos] = 1
                human_count += 1
            elif label == "black" and ai_count < 3:
                new_board[pos] = 2
                ai_count += 1
    
    print(f"Pions détectés - Humain: {human_count}, IA: {ai_count}")
    print(f"Nouvel état détecté : {new_board}")
    board = new_board
    return True

def move_arm(old_pos, new_pos):
    if old_pos not in positions_angles or new_pos not in positions_angles:
        print(f"Erreur : Position invalide ({old_pos} ou {new_pos})")
        return
    arm.Arm_serial_servo_write6(*positions_angles[old_pos])
    time.sleep(2)
    arm.Arm_serial_servo_write(6, 175, 500)
    time.sleep(0.5)
    arm.Arm_serial_servo_write6(88, 110, 10, -2, 90, 175, 500)
    time.sleep(0.5)
    new_angles = list(positions_angles[new_pos])
    new_angles[5] = 175
    arm.Arm_serial_servo_write6(*new_angles)
    time.sleep(2)
    arm.Arm_serial_servo_write(6, 135, 500)
    time.sleep(0.5)
    arm.Arm_serial_servo_write6(88, 110, 10, -2, 90, 135, 500)

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

def human_turn_pose():
    global human_pieces
    print("Posez votre pion manuellement sur le plateau, puis appuyez sur Entrée.")
    input()
    if not update_board_from_camera():
        print("Erreur caméra, veuillez réessayer.")
        return False
    human_pieces -= 1
    print("Plateau mis à jour après votre tour :")
    print_board()
    return True

def ai_turn_pose():
    global ai_pieces
    print("Tour de l’IA (pose) :")
    ai_count = sum(1 for x in board if x == 2)
    if ai_count >= 3:
        print("L’IA a déjà posé 3 pions, passage à la phase de déplacement.")
        return False
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
        print(f"L’IA va poser un pion en {best_pos}")
        move_arm("pile", best_pos)
        board[best_pos] = 2
        ai_pieces -= 1
        if not update_board_from_camera(wait_for_stability=True):
            print("Erreur caméra après le mouvement")
            return False
        print(f"Plateau après le tour IA :")
        print_board()
        return True
    else:
        print("Aucune position valide trouvée pour poser un pion !")
        return False

def human_turn_move():
    print("Déplacez votre pion manuellement sur le plateau, puis appuyez sur Entrée.")
    input()
    if not update_board_from_camera():
        print("Erreur caméra, veuillez réessayer.")
        return False
    print("Plateau mis à jour après votre tour :")
    print_board()
    return True

def ai_turn_move():
    global board
    print("Tour de l’IA (déplacement) :")
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
        return False
    old_pos, new_pos = best_move
    print(f"L’IA va déplacer un pion de {old_pos} à {new_pos}")
    move_arm(old_pos, new_pos)
    board[old_pos] = 0
    board[new_pos] = 2
    if not update_board_from_camera(wait_for_stability=True):
        print("Erreur caméra après le mouvement")
        return False
    print(f"Plateau après le tour IA :")
    print_board()
    return True

def play_game():
    global human_pieces, ai_pieces
    print("Début du jeu physique avec le Dofbot !")
    print_board()
    
    while human_pieces > 0 or ai_pieces > 0:
        if human_pieces > 0:
            if not human_turn_pose():
                continue
            if check_winner(1):
                print("Vous avez gagné !")
                return
        if ai_pieces > 0:
            if not ai_turn_pose():
                continue
            if check_winner(2):
                print("L’IA a gagné !")
                return
            print("À vous de jouer ! Attendez que le plateau soit stable avant de poser.")
            input("Appuyez sur Entrée quand vous êtes prêt...")
    
    print("Phase de déplacement commencée !")
    while True:
        if not human_turn_move():
            continue
        if check_winner(1):
            print("Vous avez gagné !")
            return
        if not ai_turn_move():
            continue
        if check_winner(2):
            print("L’IA a gagné !")
            return
        print("À vous de jouer ! Attendez que le plateau soit stable avant de déplacer.")
        input("Appuyez sur Entrée quand vous êtes prêt...")

if __name__ == "__main__":
    try:
        arm.Arm_serial_servo_write6(88, 110, 10, -2, 90, 114, 500)
        play_game()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        arm.Arm_serial_servo_write6(88, 110, 10, -2, 90, 114, 500)
