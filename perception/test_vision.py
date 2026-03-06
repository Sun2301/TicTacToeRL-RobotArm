#!/usr/bin/env python3
# test_vision.py
# Test isolé : ArUco + Homographie + YOLO → board
# Pas de bras, pas de Q-table. Juste la vision.

import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
CAMERA_INDEX = 1
MODEL_PATH   = "/home/pi/Desktop/Arm_lib-1.0.0/Arm_Lib/best_bras.pt"

ARUCO_DICT   = cv2.aruco.DICT_4X4_50
MARKER_IDS   = [0, 1, 2, 3]

aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

NORMALIZED_CORNERS = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
], dtype=np.float32)

CASE_CENTERS = np.array([
    [1/6, 1/6], [3/6, 1/6], [5/6, 1/6],
    [1/6, 3/6], [3/6, 3/6], [5/6, 3/6],
    [1/6, 5/6], [3/6, 5/6], [5/6, 5/6],
], dtype=np.float32)

# --- Chargement modèle ---
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra")
    exit()

print("Appuie sur 'q' pour quitter.")
print("-" * 50)


def get_aruco_pixels(frame):
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
    H, _ = cv2.findHomography(aruco_pixels, NORMALIZED_CORNERS)
    return H


def pixel_to_normalized(u, v, H):
    pt = np.array([[[u, v]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    return result[0][0]


def find_case(normalized_coord):
    distances = np.linalg.norm(CASE_CENTERS - normalized_coord, axis=1)
    return int(np.argmin(distances))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # --- Détection ArUco ---
    aruco_pixels = get_aruco_pixels(frame)

    if aruco_pixels is None:
        cv2.putText(display, "ArUco : marqueurs manquants", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Test Vision", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Dessiner les marqueurs détectés
    corners_all, ids_all, _ = detector.detectMarkers(frame)
    cv2.aruco.drawDetectedMarkers(display, corners_all, ids_all)
    cv2.putText(display, "ArUco : 4/4 OK", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Calcul H1 ---
    H1 = compute_H1(aruco_pixels)

    # --- YOLO ---
    results = model(frame, verbose=False)

    new_board   = [0] * 9
    human_count = 0
    ai_count    = 0

    for detection in results[0].boxes:
        confidence = detection.conf.cpu().numpy()[0]
        if confidence < 0.7:
            continue

        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        u = (x1 + x2) / 2
        v = (y1 + y2) / 2
        label = model.names[int(detection.cls.cpu().numpy()[0])]

        norm = pixel_to_normalized(u, v, H1)
        nx, ny = norm

        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            cv2.putText(display, "hors zone", (int(u), int(v)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            continue

        case = find_case(norm)

        # Afficher sur l'image
        cv2.circle(display, (int(u), int(v)), 8, (255, 0, 0), -1)
        cv2.putText(display,
                    label + " -> case " + str(case),
                    (int(u) + 10, int(v)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if label == "white" and human_count < 3:
            new_board[case] = -1
            human_count += 1
        elif label == "black" and ai_count < 3:
            new_board[case] = 1
            ai_count += 1

    # --- Afficher le board dans le terminal ---
    symbols = {0: ".", -1: "X", 1: "O"}
    print("\rBoard: " + str([symbols[x] for x in new_board]) +
          "  Humain: " + str(human_count) +
          "  IA: " + str(ai_count), end="")

    cv2.imshow("Test Vision", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nFin.")
