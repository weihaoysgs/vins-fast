//
// Created by weihao on 23-10-5.
//

#ifndef SRC_SIZE_POSE_PARAM_HPP
#define SRC_SIZE_POSE_PARAM_HPP

enum SIZE_PARAMETERIZATION
{
  SIZE_POSE = 7,
  SIZE_SPEED_BIAS = 9,
  SIZE_FEATURE = 1
};

enum StateOrder
{
  O_P = 0,
  O_R = 3,
  O_V = 6,
  O_BA = 9,
  O_BG = 12
};

#endif //SRC_SIZE_POSE_PARAM_HPP
