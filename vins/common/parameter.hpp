//
// Created by weihao on 23-8-9.
//

#ifndef PARAMETER_HPP
#define PARAMETER_HPP
#include "opencv2/opencv.hpp"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "memory"
#include "mutex"
#include "glog/logging.h"

namespace common {

static std::once_flag singleton_flag;
static std::string config_file_path_copy;

class Setting
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  static std::shared_ptr<Setting> getSingleton()
  {
    if (singleton_ == nullptr)
    {
      std::lock_guard<std::mutex> lck(singleton_mutex_);
      if (singleton_ == nullptr)
      {
        singleton_ = std::shared_ptr<Setting>(new Setting());
      }
    }
    return singleton_;
    // std::call_once(singleton_flag, [&] { singleton_ = std::shared_ptr<Setting>(new Setting()); });
    // return singleton_;
  }
  bool InitParamSetting(const std::string &config_file_path);
  const cv::FileStorage &getFile() const { return file_; }
  // access the parameter values
  template <typename T> static T Get(const std::string &key)
  {
    if (singleton_ == nullptr)
    {
      LOG(FATAL) << "singleton_ == nullptr";
    }
    if (singleton_->file_.isOpened())
    {
      return T(Setting::singleton_->file_[key]);
    }
    else
    {
      LOG(FATAL) << "File is not open success";
    }
  }
  ~Setting() = default;
  bool is_init_ = false;

private:
  Setting() = default;

private:
  static std::shared_ptr<Setting> singleton_;
  static std::mutex singleton_mutex_;
  cv::FileStorage file_;
};

inline bool Setting::InitParamSetting(const std::string &config_file_path)
{
  config_file_path_copy = config_file_path;
  singleton_->file_ = cv::FileStorage(config_file_path.c_str(), cv::FileStorage::READ);
  if (!singleton_->file_.isOpened())
  {
    LOG(FATAL) << "parameter file" << config_file_path << "does not exist.";
    singleton_->file_.release();
    return false;
  }
  is_init_ = true;
  return true;
}

} // namespace common

#endif //PARAMETER_HPP
