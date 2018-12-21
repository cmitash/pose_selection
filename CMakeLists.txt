cmake_minimum_required(VERSION 2.8.3)
project(pose_selection)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Boost REQUIRED)
find_package(catkin COMPONENTS roscpp roslib pcl_ros)
find_package(OpenCV REQUIRED)
find_package(MPI REQUIRED) 

include_directories(${PROJECT_SOURCE_DIR}/include ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} 
  ${MPI_C_INCLUDE_PATH})

FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
           /usr/include
           /usr/local/include
           /sw/include
           /opt/local/include
           DOC "The directory where GL/glew.h resides")
FIND_LIBRARY( GLEW_LIBRARY
              NAMES GLEW glew
              PATHS
              /usr/lib64
              /usr/lib
              /usr/local/lib64
              /usr/local/lib
              /sw/lib
              /opt/local/lib
              DOC "The GLEW library")

IF (GLEW_INCLUDE_PATH)
SET( GLEW_FOUND 1 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ELSE (GLEW_INCLUDE_PATH)
SET( GLEW_FOUND 0 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ENDIF (GLEW_INCLUDE_PATH)

MARK_AS_ADVANCED( GLEW_FOUND )

FIND_PACKAGE(GLUT REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED QUIET)
LIST(APPEND LINK_LIBS "${OPENGL_LIBRARIES};${GLEW_LIBRARY};${GLUT_LIBRARY};${OPENGL_LIBRARIES}"
  )
LIST(APPEND LIB_DIRS  "${OPENGL_LIBRARY_DIR};${GLEW_LIBRARY_DIR}" )
LINK_DIRECTORIES(${LIB_DIRS})
LINK_LIBRARIES(${LINK_LIBS})
MESSAGE(STATUS "link dirs: ${LIB_DIRS}")
MESSAGE(STATUS "link libs: ${LINK_LIBS}")

catkin_package(
    CATKIN_DEPENDS 
      roscpp 
      roslib 
      pcl_ros
    INCLUDE_DIRS 
      include
    LIBRARIES 
      ${PROJECT_NAME}
)

add_library(${PROJECT_NAME}
  src/simulation_io.cpp
  src/camera.cpp
  src/glsl_shader.cpp
  src/model.cpp
  src/range_likelihood.cpp
  src/scene.cpp
  src/sum_reduce.cpp)

add_executable(get_best_pose src/pose_selection.cpp
                            src/simulation_io.cpp
                            src/camera.cpp
                            src/glsl_shader.cpp
                            src/model.cpp
                            src/range_likelihood.cpp
                            src/scene.cpp
                            src/sum_reduce.cpp) 

target_link_libraries (get_best_pose ${Boost_LIBRARIES} ${catkin_LIBRARIES}
                       ${VTK_IO_TARGET_LINK_LIBRARIES}
                       ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}
                       ${GLEW_LIBRARIES} ${OpenCV_LIBRARIES}  ${MPI_LIBRARIES}
                       )

target_link_libraries (${PROJECT_NAME} ${Boost_LIBRARIES} ${catkin_LIBRARIES}
                       ${VTK_IO_TARGET_LINK_LIBRARIES}
                       ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}
                       ${GLEW_LIBRARIES} ${OpenCV_LIBRARIES}  ${MPI_LIBRARIES}
                       )

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
