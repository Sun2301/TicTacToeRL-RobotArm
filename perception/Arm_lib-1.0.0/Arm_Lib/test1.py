import random as rd
import time
import cv2
import numpy as np
import pandas as pd
import hashlib
import os
from ultralytics import YOLO
from Arm_Lib import Arm_Device

# Initialisation des variables globales
board = [0] * 9  # 0 = vide, 1 = humain "X" (white), 2 = IA "O" (black)
human_pieces = 3
ai_pieces = 3
b_time = 3

time_1 = 500
time_2 = 1000
time_sleep = 0.5

winning_combinations = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
]

arm = Arm_Device()
model = YOLO("/home/pi/Desktop/Arm_lib-1.0.0/Arm_Lib/best_bras.pt")
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra")
    exit()

"""positions_angles = {
    "pile": (122, 74, 15, 22, 87, 85, 500),
    0: (99, 32, 52, 58, 90, 135, 500),
    1: (89, 32, 52, 58, 90, 135, 500),
    2: (79, 32, 52, 58, 90, 135, 500),
    3: (102, 46, 48, 31, 89, 135, 500),
    4: (89, 43, 48, 31, 89, 135, 500),
    5: (77, 46, 48, 31, 89, 135, 500),
    6: (105, 54, 44, 13, 88, 135, 500),
    7: (89, 56, 44, 10, 88, 135, 500),
    8: (73, 55, 44, 13, 88, 135, 500)
}"""

positions_angles = {
    "pile": (115, 74, 12, 29, 87, 85, 500),
    0: (97, 20, 75, 47, 90, 135, 500),
    1: (88, 22, 74, 45, 90,135 , 500),
    2: (78, 20, 70, 52, 90, 135, 500),
    3: (99, 32, 71, 23, 90, 135, 500),
    4: (88, 32, 71, 22, 90, 135, 500),
    5: (77, 32, 71, 24, 90, 135, 500),
    6: (102, 36, 73, 1, 90, 135, 500),
    7: (87, 40, 66, 4, 90, 135, 500),
    8: (74, 35, 74, 1, 90, 135, 500)
}

def print_board():
    symbols = {0: ".", -1: "X", 1: "O"}
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
    forbidden_diagonals = {(1, 3), (3, 1), (1, 5), (5, 1), (3, 7), (7, 3), (7, 5), (5, 7)}
    if pos % 3 > 0: adj.append(pos - 1)
    if pos % 3 < 2: adj.append(pos + 1)
    if pos >= 3: adj.append(pos - 3)
    if pos <= 5: adj.append(pos + 3)
    # Diagonales directes (vérifier si elles sont interdites)
    if pos % 3 > 0 and pos >= 3:
        target = pos - 4  # Haut-gauche
        if (pos, target) not in forbidden_diagonals:
            adj.append(target)
    if pos % 3 < 2 and pos >= 3:
        target = pos - 2  # Haut-droite
        if (pos, target) not in forbidden_diagonals:
            adj.append(target)
    if pos % 3 > 0 and pos <= 5:
        target = pos + 2  # Bas-gauche
        if (pos, target) not in forbidden_diagonals:
            adj.append(target)
    if pos % 3 < 2 and pos <= 5:
        target = pos + 4  # Bas-droite
        if (pos, target) not in forbidden_diagonals:
            adj.append(target)
    #return [p for p in adj if board[p] == 0]
    return [p for p in adj if 0 <= p < 9 and board[p] == 0]


def update_board_from_camera(wait_for_stability=False):
    global board
    if wait_for_stability:
        time.sleep(1)
        
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de capturer une nouvelle frame")
            return False
    
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO Morpion', annotated_frame)
    cv2.waitKey(80)
    
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
                new_board[pos] = -1
                human_count += 1
            elif label == "black" and ai_count < 3:
                new_board[pos] = 1
                ai_count += 1
    
    print(f"Pions détectés - Humain: {human_count}, IA: {ai_count}")
    print(f"Nouvel état détecté : {new_board}")
    board = new_board
    return True


def move_arm(old_pos, new_pos):
    if old_pos not in positions_angles or new_pos not in positions_angles:
        print(f"Erreur : Position invalide ({old_pos} ou {new_pos})")
        return
    arm.Arm_serial_servo_write6(89, 60, 52, 35, 89, 175, 500)
    time.sleep(1)
    arm.Arm_serial_servo_write6(*positions_angles[old_pos])
    time.sleep(2)
    arm.Arm_serial_servo_write(6, 175, 500)
    time.sleep(0.5)
    arm.Arm_serial_servo_write6(89, 60, 52, 35, 89, 175, 500)
    time.sleep(0.5)
    arm.Arm_serial_servo_write6(90, 75, 55, -14, 89, 175, 500) #
    time.sleep(0.5)
    #arm.Arm_serial_servo_write6(*positions_angles[new_pos])
    new_angles = list(positions_angles[new_pos])  # Convertir en liste pour modifier
    new_angles[5] = 175  # Modifier l'angle du servomoteur 6 (fermé)
    # Déplacer le bras vers new_pos avec la nouvelle valeur
    arm.Arm_serial_servo_write6(89, 60, 52, 35, 89, 175, 500)
    time.sleep(1)
    arm.Arm_serial_servo_write6(*new_angles)
    time.sleep(2)
    arm.Arm_serial_servo_write(6, 135, 500)
    time.sleep(0.5)
    arm.Arm_serial_servo_write6(89, 60, 52, 35, 89, 135, 500)
    time.sleep(0.5)
    arm.Arm_serial_servo_write6(90, 75, 55, -14, 89, 175, 500) #

# Mouvements possibles
def get_possible_moves(state, player, is_pose_phase=True):
    possible_moves = []
    if is_pose_phase:
        for pos in range(9):
            if state[pos] == 0:
                possible_moves.append(pos)
    else:
        player_positions = [i for i in range(9) if state[i] == player]
        for old_pos in player_positions:
            adj = get_adjacent(old_pos, state)
            for new_pos in adj:
                if state[old_pos] == player and state[new_pos] == 0:
                    possible_moves.append((old_pos, new_pos))
    return possible_moves


def choose_action(state, player, is_pose_phase):
    try:
        q_values = q_table[tuple(state)]
        print("Q_values",q_values)
        max_value = max(q_values.values())
        print(max_value)
        best_actions = [k for k, v in q_values.items() if v == max_value]
        print(best_actions)
        return rd.choice(best_actions)
        #return next(best_actions)
    except Exception as e:
        print("execpt:", e)
        return rd.choice(get_possible_moves(state, 1, is_pose_phase))
    
"""    
def choose_action(state, player, is_pose_phase):
    try:
        return next((k for k, v in q_table[tuple(state)].items() if v == max(q_table[tuple(state)].values())), None)
        #return rd.choice((k for k, v in q_table[tuple(state)].items() if v == max(q_table[tuple(state)].values())), None)

    except:
        print("execpt")
        return rd.choice(get_possible_moves(state, 1, is_pose_phase))"""
    

def load_q_table(file_path="/home/pi/Desktop/Arm_lib-1.0.0/Arm_Lib/qtf.npy"): #q_table_file.npy v3.1/home/pi/Desktop/Arm_lib-1.0.0/Arm_Lib/q_table_final.npy
    global q_table
    try:
        q_table = np.load(file_path, allow_pickle =True).item()
        print(f"Q-Table chargée avec {len(q_table)} états.")
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} n'existe pas.")
        exit()
    except Exception as e:
        print(f"Erreur lors du chargement de la Q-Table : {e}")
        exit()
        
def auto_detect(min_conf = 5):
    global board
    print("En attente du joueur humain")
    init_board = board.copy()
    conf_count = 0

    while True:
        cam = update_board_from_camera()
        # Comparaison de l'ancien et du nouveau plateau
        if not cam:
            conf_count = 0
            continue
        if board != init_board :
            conf_count += 1
            print("Parfait")
        else :
            conf_count = 0
            
        if conf_count  >= min_conf:
            print("Changement")            
            print_board()
            return True
        time.sleep(0.5)

def human_turn_pose():
    global human_pieces
    auto_detect()
    #input()
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
    ai_count = sum(1 for x in board if x == 1)
    if ai_count >= 3:
        print("L’IA a déjà posé 3 pions, passage à la phase de déplacement.")
        return False
    state = board.copy()
    print(state)
    #state = [-i for i in state]
    move = choose_action(state, 1, True)
    if move is None:
        print("Aucune position valide trouvée pour poser un pion !")
        return False
    print(f"L’IA va poser un pion en {move}")
    move_arm("pile", move)
    board[move] = 1
    ai_pieces -= 1
    if not update_board_from_camera(wait_for_stability=True):
        print("Erreur caméra après le mouvement")
        return False
    print(f"Plateau après le tour IA :")
    print_board()
    arm.Arm_Buzzer_On(b_time)
    time.sleep(1)
    return True
        
def human_turn_move():
    auto_detect()
    #input()
    if not update_board_from_camera():
        print("Erreur caméra, veuillez réessayer.")
        return False
    print("Plateau mis à jour après votre tour :")
    print_board()
    return True

def ai_turn_move():
    global board
    print("Tour de l’IA (déplacement) :")
    state = board.copy()
    move = choose_action(state, 1, False)
    if move is None:
        print("L’IA ne peut pas bouger ! Match nul ou erreur.")
        return False
    old_pos, new_pos = move
    print(f"L’IA va déplacer un pion de {old_pos} à {new_pos}")
    move_arm(old_pos, new_pos)
    board[old_pos] = 0
    board[new_pos] = 1
    if not update_board_from_camera(wait_for_stability=True):
        print("Erreur caméra après le mouvement")
        return False
    print(f"Plateau après le tour IA :")
    print_board()
    arm.Arm_Buzzer_On(b_time)
    time.sleep(1)
    return True

def dance():
    # Make serov return to the center
    arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 500)
    time.sleep(1)
    
    #while True:
    arm.Arm_serial_servo_write(2, 180-120, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(3, 120, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(4, 60, time_1)
    time.sleep(time_sleep)

    arm.Arm_serial_servo_write(2, 180-135, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(3, 135, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(4, 45, time_1)
    time.sleep(time_sleep)

    arm.Arm_serial_servo_write(2, 180-120, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(3, 120, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(4, 60, time_1)
    time.sleep(time_sleep)

    arm.Arm_serial_servo_write(2, 90, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(3, 90, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(4, 90, time_1)
    time.sleep(time_sleep)

    arm.Arm_serial_servo_write(2, 180-80, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(3, 80, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(4, 80, time_1)
    time.sleep(time_sleep)



    arm.Arm_serial_servo_write(2, 180-60, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(3, 60, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(4, 60, time_1)
    time.sleep(time_sleep)

    arm.Arm_serial_servo_write(2, 180-45, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(3, 45, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(4, 45, time_1)
    time.sleep(time_sleep)

    arm.Arm_serial_servo_write(2, 90, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(3, 90, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(4, 90, time_1)
    time.sleep(.001)
    time.sleep(time_sleep)



    arm.Arm_serial_servo_write(4, 20, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(6, 150, time_1)
    time.sleep(.001)
    time.sleep(time_sleep)

    arm.Arm_serial_servo_write(4, 90, time_1)
    time.sleep(.001)
    arm.Arm_serial_servo_write(6, 90, time_1)
    time.sleep(time_sleep)
    
        #break

def play_game():
    global human_pieces, ai_pieces
    update_board_from_camera(wait_for_stability=True)
    load_q_table(file_path="/home/pi/Desktop/Arm_lib-1.0.0/Arm_Lib/qtf.npy") #q_table_file.npy  v3.1
    print_board()
    
    while human_pieces > 0 or ai_pieces > 0:
        # IA commance la partie
        if ai_pieces > 0:
            if not ai_turn_pose():
                continue
            if check_winner(1):
                dance()
                print("L’IA a gagné !")
                return
            
        # Humain commance la partie   
        if human_pieces > 0:
            if not human_turn_pose():
                continue
            if check_winner(-1):
                print("Vous avez gagné !")
                for i in range(3):
                    arm.Arm_Buzzer_On(b_time)
                    time.sleep(1)
                return
        
    
    print("Phase de déplacement commencée !")
    move_count = 0                                                       
    max_moves = 50
    while move_count < max_moves:
        if not ai_turn_move():
            print("Match nul (aucun mouvement possible) !")
            return
        if check_winner(1):
            dance()
            print("L’IA a gagné !")
            return
        #
        if not human_turn_move():
            continue
        if check_winner(-1):
            print("Vous avez gagné !")
            for i in range(3):
                arm.Arm_Buzzer_On(b_time)
                time.sleep(1)
            return
        #
        move_count += 1
    print("Match nul (limite de coups atteinte) !")

global state    

if __name__ == "__main__":
    try:
        arm.Arm_serial_servo_write6(90, 75, 55, -14, 89, 135, 500) #   88, 110, 10, -2, 90, 135, 500
        play_game()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        arm.Arm_serial_servo_write6(90, 75, 55, -14, 89, 135, 500)

                        