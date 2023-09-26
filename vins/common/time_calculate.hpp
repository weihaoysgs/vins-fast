#ifndef TIME_CALCULATE_HPP
#define TIME_CALCULATE_HPP
#include <chrono>

namespace common {

class TicToc
{
public:
  TicToc() { tStart(); }

  void tStart() { start = std::chrono::system_clock::now(); }

  double tEnd()
  {
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count() * 1000;
  }

private:
  std::chrono::time_point<std::chrono::system_clock> start, end;
};
} // namespace common

#endif // TIME_CALCULATE_HPP
