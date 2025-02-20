cmake_minimum_required(VERSION 3.2)

project(aicore)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fexceptions -frtti -fPIC -std=c++17 -lstdc++fs")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -pthread")
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})

# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
find_package(PkgConfig REQUIRED)
find_package(CUDA REQUIRED)
find_package(fmt)


add_subdirectory(python/pybind11)
include_directories(python/pybind11_opencv_numpy ${NUMPY_INCLUDE})




if (NCNN)
    message(STATUS "Build NCNN: " ${NCNN})
    add_definitions(-DNCNN)
    set(ncnn_DIR ${CMAKE_SOURCE_DIR}/third_party/ncnn_prebuilt/ncnn-linux/lib/cmake/ncnn)
    include_directories(
        ${CMAKE_SOURCE_DIR}/third_party/ncnn_prebuilt/ncnn-linux/include
        ${PROJECT_SOURCE_DIR}/include/model/ncnn
    )
    find_package(ncnn REQUIRED)
    file(GLOB SOURCE_NCNN ${PROJECT_SOURCE_DIR}/source/model/ncnn/*.cpp)
endif (NCNN)

if (TENSORRT)
    message(STATUS "Build TENSORRT: " ${TENSORRT})
    add_definitions(-DTENSORRT)
    find_package(TensorRT REQUIRED)
    include_directories(
        ${TensorRT_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include/model/tensorrt
    )
    file(GLOB SOURCE_TRT ${PROJECT_SOURCE_DIR}/source/model/tensorrt/*.cpp)
endif (TENSORRT)

if (ONNX)
    message(STATUS "Build ONNX: " ${ONNX})
    add_definitions(-DONNX)
    set(ONNXRUNTIME libonnxruntime.so)
    link_directories(${CMAKE_SOURCE_DIR}/third_party/onnxruntime/onnxruntime-linux-x64-1.15.1/lib)
    include_directories(
        ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/onnxruntime-linux-x64-1.15.1/include
        ${PROJECT_SOURCE_DIR}/include/model/onnx
    )
    file (GLOB SOURCE_ORT ${PROJECT_SOURCE_DIR}/source/model/onnx/*.cpp)
endif (ONNX)

if (QAIC)
    message(STATUS "Build QAIC: " ${QAIC})
    add_definitions(-DQAIC)
    # set(ONNXRUNTIME libonnxruntime.so)
    # link_directories(${CMAKE_SOURCE_DIR}/third_party/onnxruntime/onnxruntime-linux-x64-1.15.1/lib)
    include_directories(
        # ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/onnxruntime-linux-x64-1.15.1/include
        ${PROJECT_SOURCE_DIR}/include/model/aic
    )
    file (GLOB SOURCE_AIC ${PROJECT_SOURCE_DIR}/source/model/aic/*.cpp)
endif (QAIC)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

include_directories("/usr/include/eigen3/")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
# file(GLOB_RECURSE HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/model
    ${PROJECT_SOURCE_DIR}/include/model/tensorrt
    ${PROJECT_SOURCE_DIR}/include/stream
    ${PROJECT_SOURCE_DIR}/include/tracking
    ${PROJECT_SOURCE_DIR}/include/tracking/bytetrack
    ${PROJECT_SOURCE_DIR}/include/utils
    ${PROJECT_SOURCE_DIR}/include/features
    ${PROJECT_SOURCE_DIR}/include/solutions
    ${PROJECT_SOURCE_DIR}/include/solutions/threads
)

file(GLOB SOURCE 
    ${PROJECT_SOURCE_DIR}/source/*.cpp
    ${PROJECT_SOURCE_DIR}/source/model/*.cpp
    ${PROJECT_SOURCE_DIR}/source/model/tensorrt/*.cpp
    ${PROJECT_SOURCE_DIR}/source/stream/*.cpp
    ${PROJECT_SOURCE_DIR}/source/tracking/*.cpp
    ${PROJECT_SOURCE_DIR}/source/tracking/bytetrack/*.cpp
    ${PROJECT_SOURCE_DIR}/source/features/*.cpp
    ${PROJECT_SOURCE_DIR}/source/solutions/*.cpp
    ${PROJECT_SOURCE_DIR}/source/solutions/threads/*.cpp
    # ${PROJECT_SOURCE_DIR}/test/test_vehicle.cpp
    ${PROJECT_SOURCE_DIR}/test/test.cpp
    # ${PROJECT_SOURCE_DIR}/test/testFaceMask.cpp


)

if (NCNN)
    list (APPEND SOURCE ${SOURCE_NCNN})
endif (NCNN)

if (TENSORRT)
    list (APPEND SOURCE ${SOURCE_TRT})
endif (TENSORRT)

if (ONNX)
    list (APPEND SOURCE ${SOURCE_ORT})
endif (ONNX)

if (QAIC)
    list (APPEND SOURCE ${SOURCE_AIC})
endif (QAIC)

add_executable(${PROJECT_NAME} ${SOURCE})
# add_library(${PROJECT_NAME} STATIC ${SOURCE})


target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    stdc++fs fmt::fmt
)

if (NCNN)
    # target_link_libraries(${PROJECT_NAME} ncnn)
endif (NCNN)

if (TENSORRT)
    target_link_libraries(${PROJECT_NAME} ${TensorRT_LIBRARIES})
endif (TENSORRT)

if (ONNX)
    target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME})
endif (ONNX)

if (QAIC)
    # target_link_libraries(${PROJECT_NAME})
endif (QAIC)
# pybind11_add_module(pyaicore python/source/main.cpp python/pybind11_opencv_numpy/ndarray_converter.cpp)
target_link_libraries(pyaicore PRIVATE ${PROJECT_NAME})