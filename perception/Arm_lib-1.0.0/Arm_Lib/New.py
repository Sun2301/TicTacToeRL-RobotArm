#!/usr/bin/env python3
#coding=utf-8
import time
from Arm_Lib import Arm_Device
# Create robot arm object
Arm = Arm_Device()

Arm.Arm_serial_servo_write6(95, 77, 51, 0, 89, 175, 500)