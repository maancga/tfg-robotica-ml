execute_process(COMMAND "/home/manu/catkin_ws/build/turtlebot/turtlebot_create/create_node/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/manu/catkin_ws/build/turtlebot/turtlebot_create/create_node/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
