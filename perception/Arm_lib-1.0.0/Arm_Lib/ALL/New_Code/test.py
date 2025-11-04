import pandas as pd
import numpy as np
file_path="/home/pi/Desktop/Arm_lib-1.0.0/Arm_Lib/New_Code/q_table_file.npy"

q_table = np.load(file_path, allow_pickle =True).item()
#print(q_table)
print(q_table)
state = [0, 0, 1, 0, 0, 0, 0, 0, 0]

#print(q_table[tuple(state)])