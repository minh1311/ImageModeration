#ifndef TRTCommon_hpp
#define TRTCommon_hpp

#include <string>
#include <fstream>

#include "NvInfer.h"

#define TagTRTLogger "[TRTLogger]"

/**
 * @brief
 * 
*/
class TRTLogger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override;
};

/**
 * @brief
*/
constexpr long long int operator"" _GiB(long long unsigned int val)
{
    return val * (1 << 30);
}
constexpr long long int operator"" _MiB(long long unsigned int val)
{
    return val * (1 << 20);
}
constexpr long long int operator"" _KiB(long long unsigned int val)
{
    return val * (1 << 10);
}

/**
 * @brief enum class Precision
 * FP32: Full precision floating point value
 * FP16: Half prevision floating point value
 * INT8: Int8 quantization
 * @note
 * Has reduced dynamic range, may result in slight loss in accuracy.
 * If INT8 is selected, must provide path to calibration dataset directory.
*/
typedef enum Precision {
    FP32,
    FP16,
    INT8,
} Precision_t;

/**
 * @brief Options for the network
 * 
*/
typedef struct Options {
    // Precision to use for GPU inference.
    Precision precision = Precision::FP16;
    // If INT8 precision is selected, must provide path to calibration dataset directory.
    std::string calibrationDataDirectoryPath;
    // The batch size to be used when computing calibration data for INT8 inference.
    // Should be set to as large a batch number as your GPU will support.
    int32_t calibrationBatchSize = 128;
    // The batch size which should be optimized for.
    int32_t optBatchSize = 1;
    // Maximum allowable batch size
    int32_t maxBatchSize = 16;

    int32_t minWidth = 0;
    int32_t minHeight = 0;
    int32_t maxWidth = 0;
    int32_t maxHeight = 0;
    int32_t optWidth = 0;
    int32_t optHeight = 0;

    //input size
    int32_t inputWidth = 0;
    int32_t inputHeight = 0;
    // GPU device index
    int deviceIndex = 0;
} Options_t;

inline void checkCudaErrorCode(cudaError_t code) {
    if (code != 0) {
        std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) + "), with message: " + cudaGetErrorString(code);
        // std::cout << errMsg << std::endl;
        throw std::runtime_error(errMsg);
    }
}

inline bool doesFileExist(const std::string& filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

#endif // TRTCommon_hpp