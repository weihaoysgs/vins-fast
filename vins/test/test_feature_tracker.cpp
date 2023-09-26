#include <glog/logging.h>
#include <gflags/gflags.h>

int main(int argc, char **argv)
{
  google::InitGoogleLogging("TestFeatureTracking");
  FLAGS_colorlogtostderr = true;
  FLAGS_stderrthreshold = google::INFO;
  google::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Hello";
}