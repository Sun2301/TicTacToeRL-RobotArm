
import time
from Arm_Lib import Arm_Device
#Create a robotic arm object
Arm = Arm_Device()
time.sleep(.1)
time_1 = 500
time_2 = 1000
time_sleep = 0.5

#Arm.Arm_serial_servo_write6(88, 110, 10, -2, 90, 114, 500)
#time.sleep(2)
#Arm.Arm_serial_servo_write6(105, 54, 44, 15, 88, 135, 500)
#Arm.Arm_serial_servo_write6(89, 38, 52, 48, 90, 175, 500)


Arm.Arm_serial_servo_write6(89, 60, 52, 35, 89, 175, 500)
time.sleep(1)
Arm.Arm_serial_servo_write6(105, 54, 44, 13, 88, 135, 500)
time.sleep(1)
Arm.Arm_serial_servo_write(6, 175, 500)
time.sleep(0.5)
Arm.Arm_serial_servo_write6(89, 60, 52, 35, 89, 175, 500)
time.sleep(0.5)
Arm.Arm_serial_servo_write6(88, 110, 10, -2, 90, 175, 500)
time.sleep(0.5)
####Arm.Arm_serial_servo_write6(*new_angles)
Arm.Arm_serial_servo_write6(89, 60, 52, 35, 89, 175, 500)
time.sleep(1)
Arm.Arm_serial_servo_write6(89, 38, 52, 48, 90, 175, 500)
time.sleep(2)
Arm.Arm_serial_servo_write(6, 135, 500)
time.sleep(0.5)
Arm.Arm_serial_servo_write6(89, 60, 52, 35, 89, 135, 500)
time.sleep(0.5)
Arm.Arm_serial_servo_write6(88, 110, 10, -2, 90, 135, 500)