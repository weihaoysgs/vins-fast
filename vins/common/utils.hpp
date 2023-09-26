//
// Created by weihao on 23-9-26.
//

#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include "cv_bridge/cv_bridge.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

namespace common {

inline cv::Mat GetImageFromROSMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
  cv_bridge::CvImageConstPtr ptr;
  if (img_msg->encoding == "8UC1")
  {
    sensor_msgs::Image img;
    img.header = img_msg->header;
    img.height = img_msg->height;
    img.width = img_msg->width;
    img.is_bigendian = img_msg->is_bigendian;
    img.step = img_msg->step;
    img.data = img_msg->data;
    img.encoding = "mono8";
    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
  }
  else
    ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

  cv::Mat img = ptr->image.clone();
  return img;
}

inline void GetFileNames(std::string path, std::vector<std::string> &filenames)
{
  DIR *pDir;
  struct dirent *ptr;
  std::cout << "path = " << path << std::endl;
  if (!(pDir = opendir(path.c_str())))
  {
    std::cout << "Folder doesn't Exist!" << std::endl;
    return;
  }
  while ((ptr = readdir(pDir)) != 0)
  {
    if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
    {
      filenames.push_back(ptr->d_name);
    }
  }
  closedir(pDir);
}

inline bool FileExists(const std::string &file)
{
  struct stat file_status;
  if (stat(file.c_str(), &file_status) == 0 && (file_status.st_mode & S_IFREG))
  {
    return true;
  }
  return false;
}

inline bool PathExists(const std::string &path)
{
  struct stat file_status;
  if (stat(path.c_str(), &file_status) == 0 && (file_status.st_mode & S_IFDIR))
  {
    return true;
  }
  return false;
}

inline void ConcatenateFolderAndFileName(const std::string &folder, const std::string &file_name,
                                  std::string *path)
{
  *path = folder;
  if (path->back() != '/')
  {
    *path += '/';
  }
  *path = *path + file_name;
}

} // namespace common

#endif //UTILS_HPP
