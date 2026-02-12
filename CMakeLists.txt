cmake_minimum_required(VERSION 3.15)
project(MiniTF VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

file(GLOB_RECURSE LIB_SOURCES 
    "src/autograd/*.cpp"
    "src/core/*.cpp"
    "src/nn/*.cpp"
    "src/optim/*.cpp"
)

add_library(mini_tf STATIC ${LIB_SOURCES})

target_include_directories(mini_tf PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(mini_tf_app main.cpp)
target_link_libraries(mini_tf_app PRIVATE mini_tf)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/mnist_train.cpp")
    add_executable(mnist_train examples/mnist_train.cpp)
    target_link_libraries(mnist_train PRIVATE mini_tf)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/xor_train.cpp")
    add_executable(xor_train examples/xor_train.cpp)
    target_link_libraries(xor_train PRIVATE mini_tf)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/xor_load.cpp")
    add_executable(xor_load examples/xor_load.cpp)
    target_link_libraries(xor_load PRIVATE mini_tf)
endif()