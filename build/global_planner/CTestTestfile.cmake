# CMake generated Testfile for 
# Source directory: /home/adeeb/carla-ros-bridge/catkin_ws/src/global_planner
# Build directory: /home/adeeb/carla-ros-bridge/catkin_ws/build/global_planner
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(_ctest_global_planner_roslaunch-check_launch "/home/adeeb/carla-ros-bridge/catkin_ws/build/global_planner/catkin_generated/env_cached.sh" "/usr/bin/python3" "/opt/ros/noetic/share/catkin/cmake/test/run_tests.py" "/home/adeeb/carla-ros-bridge/catkin_ws/build/global_planner/test_results/global_planner/roslaunch-check_launch.xml" "--return-code" "/usr/bin/cmake -E make_directory /home/adeeb/carla-ros-bridge/catkin_ws/build/global_planner/test_results/global_planner" "/opt/ros/noetic/share/roslaunch/cmake/../scripts/roslaunch-check -o \"/home/adeeb/carla-ros-bridge/catkin_ws/build/global_planner/test_results/global_planner/roslaunch-check_launch.xml\" \"/home/adeeb/carla-ros-bridge/catkin_ws/src/global_planner/launch\" ")
set_tests_properties(_ctest_global_planner_roslaunch-check_launch PROPERTIES  _BACKTRACE_TRIPLES "/opt/ros/noetic/share/catkin/cmake/test/tests.cmake;160;add_test;/opt/ros/noetic/share/roslaunch/cmake/roslaunch-extras.cmake;66;catkin_run_tests_target;/home/adeeb/carla-ros-bridge/catkin_ws/src/global_planner/CMakeLists.txt;43;roslaunch_add_file_check;/home/adeeb/carla-ros-bridge/catkin_ws/src/global_planner/CMakeLists.txt;0;")
subdirs("gtest")
