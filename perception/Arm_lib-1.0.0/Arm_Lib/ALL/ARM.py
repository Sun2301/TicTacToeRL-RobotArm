import random
import time
import cv2
from ultralytics import YOLO
from Arm_Lib import Arm_Device

arm = Arm_Device

arm.Arm_serial_servo_write6(80, 110, 10, -2, 90, 134,500)
