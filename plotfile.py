import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import random

# p = [(2,2),(2,4),(2,8),(4,2),(4,4),(4,8),(8,2),(8,4),(8,8)]
# i1 = [29.88, 3.16, 6.34, 9.901]
# i2 = [32.12, 4.22, 7.62, 10.03]
# t1 = [[17.32  , 17.01 , 15.78 , 12.44 , 12.00 , 11.12 , 8.691 , 7.769 , 7.82],
#       [1.769  , 1.691 , 1.653 , 1.566 , 1.58 , 1.534 , 1.299 , 1.332 , 1.302],
#       [3.799  , 3.556 , 3.444 , 2.13 , 2.22 , 2.031 , 1.733 , 1.769 , 1.544],
#       [5.039  , 4.85 , 4.044 , 3.039 , 3.042 , 2.653 , 2.691 , 2.033 , 1.82]]

# s1 = []
# e1 = []
# s2 = []
# e2 = []
# sf1 = []
# sf2 = []

# iso1= []
# iso2= []

# t2 = [[18.721, 18.199, 16.096, 13.042, 13.559, 12.992, 10.213, 9.231, 9.648],
#       [2.396, 1.852, 1.83, 1.999, 1.877, 1.508, 2.212, 1.75, 1.517],
#       [4.596, 4.077, 4.127, 3.716, 3.525, 2.746, 2.314, 2.098, 2.184],
#       [6.516, 5.867, 5.349, 4.428, 4.345, 3.802, 4.116, 3.233, 2.871]]

# for i in range(4):
#     temp = [0,0,0,0,0,0,0,0,0]
#     for j in range(9):
#         temp[j] = i1[i] / t1[i][j]
#     s1.append(temp)
#     temp1 = [0,0,0,0,0,0,0,0,0]
#     for j in range(9):
#         temp1[j] = i2[i] / t2[i][j]
#     s2.append(temp1)
    
# for i in range(4):
#     temp = [0,0,0,0,0,0,0,0,0]
#     for j in range(9):
#         # print((p[j][0] * p[j][1]))
#         temp[j] = s1[i][j] / (p[j][0] * p[j][1])
#     e1.append(temp)
#     temp1 = [0,0,0,0,0,0,0,0,0]
#     for j in range(9):
#         temp1[j] = s2[i][j] / (p[j][0] * p[j][1])
#     e2.append(temp1)

# for i in range(4):
#     temp = [0,0,0,0,0,0,0,0,0]
#     for j in range(9):
#         # print((p[j][0] * p[j][1]))
#         S = s1[i][j]
#         P = p[j][0] * p[j][1]
#         temp[j] = ((1/S) - (1/P)) / (1 - (1/P))
#     sf1.append(temp)
#     temp1 = [0,0,0,0,0,0,0,0,0]
#     for j in range(9):
#         S = s2[i][j]
#         P = p[j][0] * p[j][1]
#         temp1[j] = ((1/S) - (1/P)) / (1 - (1/P))
#     sf2.append(temp1)

def printl(l):
    for i in range(len(l)):
        if(i == len(l)-1):
            print("%.3f"%l[i],end=" \\\\")
        else:
            print("%.3f"%l[i],end=" & ")


# for i in range(4):
#     temp = [0,0,0,0,0,0,0,0,0]
#     for j in range(9):
#         # print((p[j][0] * p[j][1]))
#         S = s1[i][j]
#         P = p[j][0] * p[j][1]
#         temp[j] = ((1/S) - (1/P)) / (1 - (1/P))
#     sf1.append(temp)
#     temp1 = [0,0,0,0,0,0,0,0,0]
#     for j in range(9):
#         S = s2[i][j]
#         P = p[j][0] * p[j][1]
#         temp1[j] = ((1/S) - (1/P)) / (1 - (1/P))
#     sf2.append(temp1)

t2 = [[0.605,0.762,1.804,1.553,6.697,5.139],[1.606,2.869,2.322,4.067,6.850,7.139],[3.034,3.533,3.643,9.125,8.086,11.659]]

print("\\hline")
for i in range(3):
    print("Test",end="")
    print(i+1,end=" & ")
    printl(t2[i])
    print("\n\\hline")

# xpoints = np.array([2,2,2,4,4,4,8,8,8])
# ypoints = np.array([2,4,8,2,4,8,2,4,8])
# p19 = [1,2,3,4,5,6,7,8,9]
xypoints = np.array(["(4,8192)","(8,8192)","(4,16384)","(8,16384)","(4,32768)","(8,32768)"])

# g = t2

# zpoints = np.array(g[0])
# wpoints = np.array(g[1])
# vpoints = np.array(g[2])
# # upoints = np.array(g[3])
# # spoints = np.array(g[4])

# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# plt.plot(xypoints,zpoints, marker = '.',label= "k = 10000")
# plt.plot(xypoints,wpoints, marker = '.',label= "k = 50000")
# plt.plot(xypoints,vpoints, marker = '.',label= "k = 100000")
# # plt.plot(xypoints,upoints, marker = '.',label= "")
# # plt.plot(xypoints,spoints, marker = '.',label= "testcase 4")
# plt.xlabel("Dimensions (block size,matrix size)")
# plt.ylabel("time (s)")
# plt.title("Time taken vs Dimensions")
# plt.legend()

# # Create the figure and 3D axis
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.plot3D(xpoints, ypoints, zpoints)
# # ax.set_xlabel('X Label')
# # ax.set_ylabel('Y Label')
# # ax.set_zlabel('Z Label')
# # ax.set_title('3D Scatter Plot')

# # Show the plot
# plt.show()