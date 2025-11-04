import random as rd
import time
import cv2
import numpy as np
import pandas as pd
import hashlib
import os
from ultralytics import YOLO
from Arm_Lib import Arm_Device
from copy import deepcopy

# Initialisation des variables globales
board = [0] * 9  # 0 = vide, -1 = humain "X" (white), 1 = IA "O" (black)
human_pieces = 3
ai_pieces = 3
epsilon = 0.1  # Pour l'exploration pendant le jeu
alpha = 0.1    # Taux d'apprentissage
gamma = 0.9    # Facteur d'actualisation

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
model = YOLO("/home/pi/Desktop/Arm_lib-1.0.0/Arm_Lib/best.pt")
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra")
    exit()

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

# Dictionnaire des contraintes de déplacement (similaire à CONSTRAINT_DICT)
CONSTRAINT_DICT = {
    0: [1, 3, 4],
    1: [0, 2, 4],
    2: [1, 4, 5],
    3: [0, 4, 6],
    4: [0, 1, 2, 3, 5, 6, 7, 8],
    5: [2, 4, 8],
    6: [3, 4, 7],
    7: [4, 6, 8],
    8: [4, 5, 7],
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

def get_adjacent(pos, state):
    adj = CONSTRAINT_DICT[pos]
    return [p for p in adj if 0 <= p < 9 and state[p] == 0]


def update_board_from_camera(wait_for_stability=False):
    global board
    if wait_for_stability:
        time.sleep(3)
        
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de capturer une nouvelle frame")
            return False
    
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO Morpion', annotated_frame)
    cv2.waitKey(80)
    
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
    
def get_possible_moves(state, player, is_pose_phase=True):
    possible_moves = []
    state = np.array(state)
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

def get_new_state(state, action, is_current_player_agent=True, agent_symbol=1, human_symbol=-1):
    """
    Retourne le nouvel état du plateau selon l'action et le joueur
    """
    player_symbol = agent_symbol if is_current_player_agent else human_symbol
    new_state = deepcopy(np.array(state))

    if isinstance(action, int):
        new_state[action] = player_symbol
    elif isinstance(action, tuple):
        new_state[action[0]] = 0
        new_state[action[1]] = player_symbol
    else:
        return None
    return new_state

def update_qtable(q_table, state, action, reward, new_state, alpha=0.1, gamma=0.9):
    """
    Met à jour la Q-table en utilisant la différence temporelle et l'équation de Bellman
    """
    state = state_to_tuple(state)
    new_state = state_to_tuple(new_state)
    
    # Ajouter l'état s'il n'existe pas
    if state not in q_table:
        q_table[state] = {action: 0 for action in get_possible_moves(state, 1, is_pose_phase=(sum(1 for x in state if x == 1) < 3))}
    
    # Ajouter l'action si elle n'existe pas
    if action not in q_table[state]:
        q_table[state][action] = 0
    
    # Ajouter le nouvel état s'il n'existe pas
    if new_state not in q_table:
        q_table[new_state] = {action: 0 for action in get_possible_moves(new_state, 1, is_pose_phase=(sum(1 for x in new_state if x == 1) < 3))}
    
    q_value_max = max(q_table[new_state].values(), default=0)
    q_table[state][action] += alpha * (reward + gamma * q_value_max - q_table[state][action])

def choose_action(state, player, is_pose_phase, q_table, epsilon=0.1):
    """
    Choisit une action avec epsilon-greedy, met à jour la Q-table si nécessaire
    """
    actions = get_possible_moves(state, player, is_pose_phase)
    state_key = state_to_tuple(state)
    
    # Ajouter l'état à la Q-table s'il n'existe pas
    if state_key not in q_table:
        q_table[state_key] = {action: 0 for action in actions}
    
    if rd.uniform(0, 1) < epsilon:
        return rd.choice(actions)
    else:
        if not q_table[state_key]:  # Si aucune action n'a de valeur
            return rd.choice(actions)
        return max(q_table[state_key], key=q_table[state_key].get)

def load_q_table(file_path="/home/pi/Desktop/Arm_lib-1.0.0/Arm_Lib/q_table_add.npy"):
    global q_table
    try:
        q_table = np.load(file_path, allow_pickle=True).item()
        print(f"Q-Table chargée avec {len(q_table)} états.")
    except FileNotFoundError:
        print(f"Q-Table non trouvée, initialisation vide.")
        q_table = {}
    except Exception as e:
        print(f"Erreur lors du chargement de la Q-Table : {e}")
        q_table = {}

def save_q_table(file_path="/home/pi/Desktop/Arm_lib-1.0.0/Arm_Lib/q_table_save.npy"):
    """
    Sauvegarde la Q-table mise à jour
    """
    try:
        np.save(file_path, q_table)
        print(f"Q-Table sauvegardée avec {len(q_table)} états.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de la Q-Table : {e}")

def auto_detect(min_conf=5):
    global board
    print("En attente du joueur humain")
    init_board = board.copy()
    conf_count = 0

    while True:
        cam = update_board_from_camera()
        if not cam:
            conf_count = 0
            continue
        if board != init_board:
            conf_count += 1
            print("Parfait")
        else:
            conf_count = 0
            
        if conf_count >= min_conf:
            print("Changement")            
            print_board()
            return True
        time.sleep(0.5)

def human_turn_pose():
    global human_pieces
    auto_detect()
    if not update_board_from_camera():
        print("Erreur caméra, veuillez réessayer.")
        return False
    human_pieces -= 1
    print("Plateau mis à jour après votre tour :")
    print_board()
    return True

def ai_turn_pose(q_table):
    global ai_pieces, board
    print("Tour de l’IA (pose) :")
    ai_count = sum(1 for x in board if x == 1)
    if ai_count >= 3:
        print("L’IA a déjà posé 3 pions, passage à la phase de déplacement.")
        return False
    state = board.copy()
    old_state = deepcopy(state)
    action = choose_action(state, 1, True, q_table, epsilon)
    if action is None:
        print("Aucune position valide trouvée pour poser un pion !")
        return False
    print(f"L’IA va poser un pion en {action}")
    move_arm("pile", action)
    board[action] = 1
    ai_pieces -= 1
    if not update_board_from_camera(wait_for_stability=True):
        print("Erreur caméra après le mouvement")
        return False
    new_state = board.copy()
    
    # Calculer la récompense
    reward = -0.1  # Récompense par défaut pour un mouvement
    if check_winner(1):
        reward = 5
    elif check_winner(-1):
        reward = -4
    
    # Mettre à jour la Q-table
    update_qtable(q_table, old_state, action, reward, new_state, alpha, gamma)
    
    print(f"Plateau après le tour IA :")
    print_board()
    return True

def human_turn_move():
    global board
    auto_detect()
    if not update_board_from_camera():
        print("Erreur caméra, veuillez réessayer.")
        return False
    print("Plateau mis à jour après votre tour :")
    print_board()
    return True

def ai_turn_move(q_table):
    global board
    print("Tour de l’IA (déplacement) :")
    state = board.copy()
    old_state = deepcopy(state)
    action = choose_action(state, 1, False, q_table, epsilon)
    if action is None:
        print("L’IA ne peut pas bouger ! Match nul ou erreur.")
        return False
    old_pos, new_pos = action
    print(f"L’IA va déplacer un pion de {old_pos} à {new_pos}")
    move_arm(old_pos, new_pos)
    board[old_pos] = 0
    board[new_pos] = 1
    if not update_board_from_camera(wait_for_stability=True):
        print("Erreur caméra après le mouvement")
        return False
    new_state = board.copy()
    
    # Calculer la récompense
    reward = -0.1
    if check_winner(1):
        reward = 5
    elif check_winner(-1):
        reward = -4
    
    # Mettre à jour la Q-table
    update_qtable(q_table, old_state, action, reward, new_state, alpha, gamma)
    
    print(f"Plateau après le tour IA :")
    print_board()
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
    global human_pieces, ai_pieces, q_table
    update_board_from_camera(wait_for_stability=True)
    load_q_table(file_path="/home/pi/Desktop/Arm_lib-1.0.0/Arm_Lib/q_table_v3.1_updated.npy")
    print_board()
    
    move_count = 0
    while human_pieces > 0 or ai_pieces > 0:
        if ai_pieces > 0:
            if not ai_turn_pose(q_table):
                continue
            if check_winner(1):
                print("L’IA a gagné !")
                save_q_table()
                dance()
                return
        if human_pieces > 0:
            if not human_turn_pose():
                continue
            if check_winner(-1):
                print("Vous avez gagné !")
                save_q_table()
                return
        move_count += 1
        if move_count % 10 == 0:  # Sauvegarde périodique
            save_q_table()
    
    print("Phase de déplacement commencée !")
    move_count = 0
    max_moves = 50
    while move_count < max_moves:
        if not ai_turn_move(q_table):
            print("Match nul (aucun mouvement possible) !")
            save_q_table()
            return
        if check_winner(1):
            print("L’IA a gagné !")
            save_q_table()
            return
        if not human_turn_move():
            continue
        if check_winner(-1):
            print("Vous avez gagné !")
            save_q_table()
            return
        move_count += 1
        if move_count % 10 == 0:  # Sauvegarde périodique
            save_q_table()
    print("Match nul (limite de coups atteinte) !")
    save_q_table()

if __name__ == "__main__":
    try:
        arm.Arm_serial_servo_write6(90, 75, 55, -14, 89, 135, 500) #   88, 110, 10, -2, 90, 135, 500
        play_game()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        arm.Arm_serial_servo_write6(90, 75, 55, -14, 89, 135, 500)


