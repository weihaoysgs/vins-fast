//
// Created by weihao on 23-9-29.
//

#ifndef ALGORITHM_HPP
#define ALGORITHM_HPP

#include "Eigen/Core"
#include "Eigen/Dense"

namespace common {
class Algorithm
{
public:
  template <typename Derived>
  static Eigen::Quaternion<typename Derived::Scalar> DeltaQ(const Eigen::MatrixBase<Derived> &theta)
  {
    typedef typename Derived::Scalar Scalar_t;

    Eigen::Quaternion<Scalar_t> dq;
    Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
    half_theta /= static_cast<Scalar_t>(2.0);
    dq.w() = static_cast<Scalar_t>(1.0);
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
  }

  template <typename Derived>
  static Eigen::Matrix<typename Derived::Scalar, 3, 3> SkewSymmetric(const Eigen::MatrixBase<Derived> &q)
  {
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1), q(2), typename Derived::Scalar(0), -q(0), -q(1), q(0),
        typename Derived::Scalar(0);
    return ans;
  }

  template <typename Derived>
  static Eigen::Quaternion<typename Derived::Scalar> positify(const Eigen::QuaternionBase<Derived> &q)
  {
    return q;
  }

  template <typename Derived>
  static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
  {
    Eigen::Quaternion<typename Derived::Scalar> qq = positify(q);
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
    ans.template block<3, 1>(1, 0) = qq.vec(), ans.template block<3, 3>(1, 1) =
                                                   qq.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() +
                                                   SkewSymmetric(qq.vec());
    return ans;
  }

  template <typename Derived>
  static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p)
  {
    Eigen::Quaternion<typename Derived::Scalar> pp = positify(p);
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
    ans.template block<3, 1>(1, 0) = pp.vec(), ans.template block<3, 3>(1, 1) =
                                                   pp.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() -
                                                   SkewSymmetric(pp.vec());
    return ans;
  }

  static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
  {
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
  }

  template <typename Derived>
  static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr)
  {
    typedef typename Derived::Scalar Scalar_t;

    Scalar_t y = ypr(0) / 180.0 * M_PI;
    Scalar_t p = ypr(1) / 180.0 * M_PI;
    Scalar_t r = ypr(2) / 180.0 * M_PI;

    Eigen::Matrix<Scalar_t, 3, 3> Rz;
    Rz << cos(y), -sin(y), 0, sin(y), cos(y), 0, 0, 0, 1;

    Eigen::Matrix<Scalar_t, 3, 3> Ry;
    Ry << cos(p), 0., sin(p), 0., 1., 0., -sin(p), 0., cos(p);

    Eigen::Matrix<Scalar_t, 3, 3> Rx;
    Rx << 1., 0., 0., 0., cos(r), -sin(r), 0., sin(r), cos(r);

    return Rz * Ry * Rx;
  }

  static Eigen::Matrix3d g2R(const Eigen::Vector3d &g)
  {
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    /// 经测试发现这里改为 (3,2,1) 好像也可以正常运行
    Eigen::Vector3d ng2{0, 0, 1.0};
    ng2 = ng2.normalized();
    // (R0 * ng1) = ng2
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    // std::cout << ng1.transpose() << "," << ng2.transpose() << std::endl;
    // std::cout << (R0 * ng1).transpose() << "," << (R0 * ng2).transpose() << std::endl;

    /// 得到当前姿态在世界系下的 yaw 角
    double yaw = R2ypr(R0).x();
    /// 单独计算出该 -yaw 角对应的姿态 R (因为是要往回旋转) -> R_{wc_1}
    /// 去掉 yaw 角，则要在当前的位姿下反向旋转 yaw 角度对应的旋转矩阵（反向对应的就是负数的 yaw）
    /// R0 可以看做 ypr 三轴，可以看做是对 此时坐标轴的旋转
    R0 = ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    /// 一些思考：其实在VIO系统中，YAW角本身就是不可观的，因此这里其实你是否将其再进行 -yaw 角度的旋转其实是无所谓的（如果只是研究精度的，在最后进行对齐就行了）
    /// 但是对于无人机等机器人系统来说，其是有朝向的，且对后面的导航至关重要，因此我们一般就直接设置成0在初始时刻，方便后面的一些导航算法的执行和计算。
    return R0;
  }

  template <size_t N> struct uint_
  {
  };

  template <size_t N, typename Lambda, typename IterT> void unroller(const Lambda &f, const IterT &iter, uint_<N>)
  {
    unroller(f, iter, uint_<N - 1>());
    f(iter + N);
  }

  template <typename Lambda, typename IterT> void unroller(const Lambda &f, const IterT &iter, uint_<0>) { f(iter); }

  template <typename T> static T normalizeAngle(const T &angle_degrees)
  {
    T two_pi(2.0 * 180);
    if (angle_degrees > 0)
      return angle_degrees - two_pi * std::floor((angle_degrees + T(180)) / two_pi);
    else
      return angle_degrees + two_pi * std::floor((-angle_degrees + T(180)) / two_pi);
  };
};
} // namespace common

#endif //ALGORITHM_HPP
