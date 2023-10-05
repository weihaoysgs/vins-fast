//
// Created by weihao on 23-9-26.
//
#include "common/parameter.hpp"

namespace common {
std::shared_ptr<Setting> Setting::singleton_ = nullptr;
std::mutex Setting::singleton_mutex_;
} // namespace common