
add_library(util nms.cpp)
#  public关键字保证util中头文件路径对main可见
target_include_directories(util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})