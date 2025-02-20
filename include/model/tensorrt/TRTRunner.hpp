#ifndef TRTRunner_hpp
#define TRTRunner_hpp

#include <vector>
#include <memory>
#include <unordered_map>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include "GPSLogger.h"
#include "TRTCommon.hpp"

#define TagTRTRunner "[TRTRunner]"

/**
 * @brief
 * 
*/
class TRTRunner
{
    public:

        /**
         * @brief
        */
        TRTRunner(const std::string& strModelPath, Options_t& stOptions);

        /**
         * @brief
        */
        ~TRTRunner();

        /**
         * @brief
        */
        void allocateIOBuffers(std::vector<void*>& buffers, std::unordered_map<std::string, std::vector<size_t>>& umpIOTensors, 
                                std::unordered_map<std::string, nvinfer1::Dims>& umpIOTensorsShape);

        /**
         * @brief
        */
        void runModel(std::vector<void*>& buffers);

    private:
        std::string serializeEngineOptions(const std::string& strModelPath);
        std::vector<std::string> getDeviceNames();

        bool buildEngine(const std::string& strModelPath);
        bool loadEngine(std::string& strPathEngine);

        size_t getSizeByDims(const nvinfer1::Dims& tensorShape);

    private:
        TRTLogger m_TRTLogger;
        Options_t m_stOptions;
        std::string m_strEngineName;

        std::vector<std::string> m_IOTensorNames;
        std::unordered_map<std::string, std::vector<size_t>> m_umpIOTensors;
        std::unordered_map<std::string, nvinfer1::Dims> m_umpIOTensorsShape;

        std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
        std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
        std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
        cudaStream_t m_stream = nullptr;

};

#endif // TRTRunner_hpp