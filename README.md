# Vins-Modern [(Continuously updating...)](https://github.com/weihaoysgs/vins-fast) :heart_eyes:
<div align="center">

[English](README.md) | [Chinese](doc/Chinese.md)

![Author](https://img.shields.io/badge/Author-isweihao@zju.edu.cn-blue?link=https%3A%2F%2Fgithub.com%2Fweihaoysgs)
![License](https://img.shields.io/badge/License-GPLv3-green)

</div>

VINS has been completely reconstructed and rewritten using C++ object-oriented, and supports stero or stero + IMU.


## Stero Only Mode Result :sunglasses:

The results of running on the [Euroc](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) dataset `MH_05_difficult` in stero only mode. The accuracy is evaluated by the [EVO](https://github.com/MichaelGrupp/evo) tool, the RMSE is `0.40`

<div align="center">

<img src="./images/MH_05_Stero_Only.png" width = 55%>

</div>

## IMU Simulate Result :kissing_smiling_eyes:

Generate simulate IMU data through [vio-data-simulation](https://github.com/HeYijia/vio_data_simulation), verify whether the pre-integration results in the program are consistent with the normal integration results, and verify the correctness of the Jacobian matrix. You can even generate an entire VIO simulation dataset to verify your slam algorithm.

<div align="center">

<img src="./images/imu_simulate.png" width = 55%>

</div>

## Stero IMU Result :yum:

The stero IMU mode is test success, the `evo_ape` result as follows, the RMSE is `0.294500`.

```shell
   max   0.466681
  mean   0.273919
median   0.284079
   min   0.041530
  rmse   0.294500
   sse   96.357503
   std   0.108160
```

<div align="center">

<img src="./images/MH_05_Stero_Imu.png" width = 55%>

</div>

## Reference :stuck_out_tongue_winking_eye:

- [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)
- [Vio-data-simulation](https://github.com/HeYijia/vio_data_simulation)

## TODO List

- [x] Add IMU to opt
- [ ] Add deep method
- [ ] Add other sensor