#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# time,w,x,y,z,tx,ty,tz,vx,vy,vz,gx,gy,gz,ax,ay,az
if __name__ == "__main__":
    # imu_gt_pose_txt = sys.argv[1]
    imu_gt_pose_txt = "./imu_integ/imu_raw_pose.txt"
    imu_raw_integration_txt = "./imu_integ/gt_integration_result.txt"
    imu_raw_noise_integration_txt = "./imu_integ/noise_integration_result.txt"
    print(imu_gt_pose_txt, imu_raw_integration_txt, imu_raw_noise_integration_txt)
    position_gt = np.loadtxt(imu_gt_pose_txt, usecols=(5, 5 + 1, 5 + 2))
    position_gt_integration = np.loadtxt(imu_raw_integration_txt, usecols=(5, 5 + 1, 5 + 2))
    position_gt_noise_integration = np.loadtxt(imu_raw_noise_integration_txt, usecols=(5, 5 + 1, 5 + 2))

    start_pt = np.array([position_gt[0, 0], position_gt[0, 1], position_gt[0, 2]])
    print(start_pt)
    index = len(position_gt_integration) - 1
    end_pt = np.array([position_gt_integration[index, 0], position_gt_integration[index, 1], position_gt_integration[index, 2]])
    print(end_pt)
    print(start_pt - end_pt)


    ### plot 3d
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(position_gt[:, 0], position_gt[:, 1], position_gt[:, 2], label='gt')
    ax.plot(position_gt_integration[:, 0], position_gt_integration[:, 1], position_gt_integration[:, 2],
            label='gt_integration')
    ax.plot(position_gt_noise_integration[:, 0], position_gt_noise_integration[:, 1],
            position_gt_noise_integration[:, 2], label='gt_integration_noise')
    ax.plot([position_gt[0, 0]], [position_gt[0, 1]], [position_gt[0, 2]], 'r.', label='start')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
