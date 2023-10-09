mkdir -p catkin_ws/src
cd catkin_ws/src
git clone https://github.com/weihaoysgs/vins-fast
cd ..
catkin_make -j

./build/vins-fast/vins/test_feature_track