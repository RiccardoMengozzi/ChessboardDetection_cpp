cmake_minimum_required(VERSION 3.15)
project(ChessboardDetection_cpp)

set(CMAKE_CXX_STANDARD 17)


# Include the header files directory
include_directories(src/include)

# Add the source files to the project
add_executable(ChessboardDetection_cpp 
    src/src/main.cpp 
    src/src/Chessboard.cpp
    src/src/Disparity.cpp
    src/src/Image.cpp
)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(ChessboardDetection_cpp   ${OpenCV_LIBS})
target_link_libraries(ChessboardDetection_cpp   yaml-cpp)
                                                

