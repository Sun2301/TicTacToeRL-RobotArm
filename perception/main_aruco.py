import random as rd
import time
import cv2
import numpy as np
import pandas as pd
import hashlib
import os
from ultralytics import YOLO
from Arm_Lib import Arm_Device

Q_TABLE = ['q_table_k.npy', 'qtablee.npy', 'qtf.npy']

# PUT YOUR Q_TABLE HERE
file_path = "/home/pi/Desktop/Arm_lib-1.0.0/Arm_Lib/q_table_k.npy"

# Initialisation des variables globales
board = [0] * 9  # 0 = vide, -1 = humain "X" (white), 1 = IA "O" (black)
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

positions_angles = {
    "pile": (115, 74, 12, 29, 87, 85, 500),
    0: (97, 20, 75, 47, 90, 135, 500),
    1: (88, 22, 74, 45, 90, 135, 500),
    2: (78, 20, 70, 52, 90, 135, 500),
    3: (99, 32, 71, 23, 90, 135, 500),
    4: (88, 32, 71, 22, 90, 135, 500),
    5: (77, 32, 71, 24, 90, 135, 500),
    6: (102, 36, 73, 1, 90, 135, 500),
    7: (87, 40, 66, 4, 90, 135, 500),
    8: (74, 35, 74, 1, 90, 135, 500)
}

# =============================================================================
# ARUCO + HOMOGRAPHIE
# =============================================================================

ARUCO_DICT   = cv2.aruco.DICT_4X4_50
MARKER_IDS   = [0, 1, 2, 3]

aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Coordonnées normalisées des 4 coins du plateau
# Marqueur 0 --> (0,0), Marqueur 1 --> (1,0), Marqueur 2 --> (0,1), Marqueur 3 --> (1,1)
NORMALIZED_CORNERS = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
], dtype=np.float32)

# Centres des 9 cases dans l'espace normalisé
CASE_CENTERS = np.array([
    [1/6, 1/6], [3/6, 1/6], [5/6, 1/6],
    [1/6, 3/6], [3/6, 3/6], [5/6, 3/6],
    [1/6, 5/6], [3/6, 5/6], [5/6, 5/6],
], dtype=np.float32)


def get_aruco_pixels(frame):
    """Détecte les 4 marqueurs ArUco et retourne leurs centres en pixels."""
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return None
    centers = {}
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in MARKER_IDS:
            center = corners[i][0].mean(axis=0)
            centers[int(marker_id)] = center
    if not all(m in centers for m in MARKER_IDS):
        return None
    return np.array([centers[m] for m in MARKER_IDS], dtype=np.float32)


def compute_H1(aruco_pixels):
    """Calcule H1 : pixels image --> coordonnées normalisées plateau."""
    H, _ = cv2.findHomography(aruco_pixels, NORMALIZED_CORNERS)
    return H


def pixel_to_normalized(u, v, H):
    """Convertit un pixel image en coordonnées normalisées via H1."""
    pt = np.array([[[u, v]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    return result[0][0]


def find_case(normalized_coord):
    """Trouve la case la plus proche parmi les 9 cases."""
    distances = np.linalg.norm(CASE_CENTERS - normalized_coord, axis=1)
    return int(np.argmin(distances))


# =============================================================================
# FONCTIONS DU JEU
# =============================================================================

def print_board():
    symbols = {0: ".", -1: "X", 1: "O"}
    for i in range(0, 9, 3):
        print(symbols[board[i]] + " " + symbols[board[i+1]] + " " + symbols[board[i+2]])
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
    if pos % 3 > 0 and pos >= 3:
        target = pos - 4
        if (pos, target) not in forbidden_diagonals:
            adj.append(target)
    if pos % 3 < 2 and pos >= 3:
        target = pos - 2
        if (pos, target) not in forbidden_diagonals:
            adj.append(target)
    if pos % 3 > 0 and pos <= 5:
        target = pos + 2
        if (pos, target) not in forbidden_diagonals:
            adj.append(target)
    if pos % 3 < 2 and pos <= 5:
        target = pos + 4
        if (pos, target) not in forbidden_diagonals:
            adj.append(target)
    return [p for p in adj if 0 <= p < 9 and board[p] == 0]


def update_board_from_camera(wait_for_stability=False):
    global board
    if wait_for_stability:
        time.sleep(1)

    # Vider le buffer caméra
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de capturer une nouvelle frame")
            return False

    # Détecter les marqueurs ArUco
    aruco_pixels = get_aruco_pixels(frame)
    if aruco_pixels is None:
        print("Erreur : marqueurs ArUco non détectés (moins de 4 visibles)")
        return False

    # Calculer H1
    H1 = compute_H1(aruco_pixels)

    # Lancer YOLO
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO Morpion', annotated_frame)
    cv2.waitKey(80)

    # Construire le nouveau board
    new_board = [0] * 9
    human_count = 0
    ai_count = 0

    for detection in results[0].boxes:
        confidence = detection.conf.cpu().numpy()[0]
        if confidence < 0.7:
            continue

        # Centre de la bounding box
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        u = (x1 + x2) / 2
        v = (y1 + y2) / 2

        label = model.names[int(detection.cls.cpu().numpy()[0])]

        # Convertir pixel --> coordonnées normalisées
        norm = pixel_to_normalized(u, v, H1)
        nx, ny = norm

        # Ignorer les pions hors zone du plateau
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            print("Pion hors zone ignore : (" + str(round(nx, 2)) + ", " + str(round(ny, 2)) + ")")
            continue

        # Trouver la case
        case = find_case(norm)

        # Assigner au board
        if label == "white" and human_count < 3:
            new_board[case] = -1
            human_count += 1
        elif label == "black" and ai_count < 3:
            new_board[case] = 1
            ai_count += 1

    print("Pions detectes - Humain: " + str(human_count) + ", IA: " + str(ai_count))
    print("Nouvel etat detecte : " + str(new_board))
    board = new_board
    return True


def move_arm(old_pos, new_pos):
    if old_pos not in positions_angles or new_pos not in positions_angles:
        print("Erreur : Position invalide (" + str(old_pos) + " ou " + str(new_pos) + ")")
        return
    arm.Arm_serial_servo_write6(89, 60, 52, 35, 89, 175, 500)
    time.sleep(1)
    arm.Arm_serial_servo_write6(*positions_angles[old_pos])
    time.sleep(2)
    arm.Arm_serial_servo_write(6, 175, 500)
    time.sleep(0.5)
    arm.Arm_serial_servo_write6(89, 60, 52, 35, 89, 175, 500)
    time.sleep(0.5)
    arm.Arm_serial_servo_write6(90, 75, 55, -14, 89, 175, 500)
    time.sleep(0.5)
    new_angles = list(positions_angles[new_pos])
    new_angles[5] = 175
    arm.Arm_serial_servo_write6(89, 60, 52, 35, 89, 175, 500)
    time.sleep(1)
    arm.Arm_serial_servo_write6(*new_angles)
    time.sleep(2)
    arm.Arm_serial_servo_write(6, 135, 500)
    time.sleep(0.5)
    arm.Arm_serial_servo_write6(89, 60, 52, 35, 89, 135, 500)
    time.sleep(0.5)
    arm.Arm_serial_servo_write6(90, 75, 55, -14, 89, 175, 500)


def get_possible_moves(state, player, is_pose_phase=True):
    possible_moves = []
    if is_pose_phase:
        for pos in range(9):
            if state[pos] == 0:
                possible_moves.append(pos)
    else:
        player_positions = [i for i in range(9) if state[i] == player]
        for old_pos in player_positions:
            adj = get_adjacent(old_pos)
            for new_pos in adj:
                if state[old_pos] == player and state[new_pos] == 0:
                    possible_moves.append((old_pos, new_pos))
    return possible_moves


def choose_action(state, player, is_pose_phase):
    possible_move = get_possible_moves(state, player, is_pose_phase)
    try:
        q_values = q_table[tuple(state)]
        print("Q_values", q_values)
        max_value = max(q_values.values())
        best_actions = [k for k, v in q_values.items() if v == max_value]
        best_action = rd.choice(best_actions)
        if best_action in possible_move:
            print("Good choose: ", best_action)
            return best_action
        else:
            print("Bad action ! make random")
            return rd.choice(possible_move)
    except Exception as e:
        print("execpt:", e)
        return rd.choice(get_possible_moves(state, 1, is_pose_phase))


def load_q_table(file_path=file_path):
    global q_table
    try:
        q_table = np.load(file_path, allow_pickle=True).item()
        print("Q-Table chargee avec " + str(len(q_table)) + " etats.")
    except FileNotFoundError:
        print("Erreur : Le fichier " + file_path + " n'existe pas.")
        exit()
    except Exception as e:
        print("Erreur lors du chargement de la Q-Table : " + str(e))
        exit()


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
        print("Erreur camera, veuillez reessayer.")
        return False
    human_pieces -= 1
    print("Plateau mis a jour apres votre tour :")
    print_board()
    return True


def ai_turn_pose():
    global ai_pieces
    print("Tour de l'IA (pose) :")
    ai_count = sum(1 for x in board if x == 1)
    if ai_count >= 3:
        print("L'IA a deja pose 3 pions, passage a la phase de deplacement.")
        return False
    state = board.copy()
    print(state)
    move = choose_action(state, 1, True)
    if move is None:
        print("Aucune position valide trouvee pour poser un pion !")
        return False
    print("L'IA va poser un pion en " + str(move))
    move_arm("pile", move)
    board[move] = 1
    ai_pieces -= 1
    if not update_board_from_camera(wait_for_stability=True):
        print("Erreur camera apres le mouvement")
        return False
    print("Plateau apres le tour IA :")
    print_board()
    arm.Arm_Buzzer_On(b_time)
    time.sleep(1)
    return True


def human_turn_move():
    auto_detect()
    if not update_board_from_camera():
        print("Erreur camera, veuillez reessayer.")
        return False
    print("Plateau mis a jour apres votre tour :")
    print_board()
    return True


def ai_turn_move():
    global board
    print("Tour de l'IA (deplacement) :")
    state = board.copy()
    move = choose_action(state, 1, False)
    if move is None:
        print("L'IA ne peut pas bouger ! Match nul ou erreur.")
        return False
    old_pos, new_pos = move
    print("L'IA va deplacer un pion de " + str(old_pos) + " a " + str(new_pos))
    move_arm(old_pos, new_pos)
    board[old_pos] = 0
    board[new_pos] = 1
    if not update_board_from_camera(wait_for_stability=True):
        print("Erreur camera apres le mouvement")
        return False
    print("Plateau apres le tour IA :")
    print_board()
    arm.Arm_Buzzer_On(b_time)
    time.sleep(1)
    return True


def dance():
    arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 500)
    time.sleep(1)
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


def play_game():
    global human_pieces, ai_pieces, file_path
    update_board_from_camera(wait_for_stability=True)
    load_q_table(file_path)
    print_board()

    while human_pieces > 0 or ai_pieces > 0:
        if ai_pieces > 0:
            if not ai_turn_pose():
                continue
            if check_winner(1):
                dance()
                print("L'IA a gagne !")
                return
        if human_pieces > 0:
            if not human_turn_pose():
                continue
            if check_winner(-1):
                print("Vous avez gagne !")
                for i in range(3):
                    arm.Arm_Buzzer_On(b_time)
                    time.sleep(1)
                return

    print("Phase de deplacement commencee !")
    move_count = 0
    max_moves = 50
    while move_count < max_moves:
        if not ai_turn_move():
            print("Match nul (aucun mouvement possible) !")
            return
        if check_winner(1):
            dance()
            print("L'IA a gagne !")
            return
        if not human_turn_move():
            continue
        if check_winner(-1):
            print("Vous avez gagne !")
            for i in range(3):
                arm.Arm_Buzzer_On(b_time)
                time.sleep(1)
            return
        move_count += 1
    print("Match nul (limite de coups atteinte) !")


global state

if __name__ == "__main__":
    try:
        arm.Arm_serial_servo_write6(90, 75, 55, -14, 89, 135, 500)
        play_game()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        arm.Arm_serial_servo_write6(90, 75, 55, -14, 89, 135, 500)
