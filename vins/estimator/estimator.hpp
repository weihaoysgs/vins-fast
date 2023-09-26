#ifndef ESTIMATOR_HPP
#define ESTIMATOR_HPP

#include <glog/logging.h>

namespace estimator {

class Estimator
{
public:
  void printHello() { LOG(INFO) << "Hello"; }
};

}; // namespace estimator

#endif // ESTIMATOR_HPP