#import random
import time
#import cv2
#from ultralytics import YOLO
from Arm_Lib import Arm_Device


arm = Arm_Device()


"""model = YOLO("/home/pi/Desktop/Arm_lib-1.0.0/Arm_Lib/best_bras.pt")

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra")
    exit()
arm.Arm_serial_servo_write6(90, 75, 55, -14, 89, 135, 500)
time.sleep(2)
while True: 
    ret, frame = cap.read()
    results = model(frame)
    anot = results[0].plot()
    cv2.imshow('YOLO Morpion', anot)
    cv2.waitKey(80)
       
    """

#print(help(arm.Arm_serial_servo_write6))

#arm.Arm_serial_servo_write6(115, 74, 12, 29, 87,175, 500)#
#time.sleep(3)

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

move_arm("pile", 8)
