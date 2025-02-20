#include "TRTModel.hpp"

#include <iostream>

TRTModel::TRTModel()
{
    
}

TRTModel::TRTModel(nlohmann::json& jModelConfig, const std::string& strModelName)
{
    m_strModelName = strModelName;
    std::string strModelPath = jModelConfig[JN_ASSETS_DIR].get<std::string>() + jModelConfig[m_strModelName][JN_ONNX_FILE].get<std::string>();
    Options_t stOptions;
    stOptions.deviceIndex = jModelConfig[m_strModelName][JN_DEVICE_INDEX].get<int>();
    std::string strPrecision = jModelConfig[m_strModelName][JN_PRECISION].get<std::string>();
    if (strPrecision == "fp32")
        stOptions.precision = Precision_t::FP32;
    else if (strPrecision == "fp16")
        stOptions.precision = Precision_t::FP16;
    else if (strPrecision == "int8")
        stOptions.precision = Precision_t::INT8;
    else
        stOptions.precision = Precision_t::FP16;

    stOptions.optBatchSize = jModelConfig[m_strModelName][JN_OPT_PROFILE]["opt"].get<std::vector<int>>()[0];
    stOptions.maxBatchSize = jModelConfig[m_strModelName][JN_OPT_PROFILE]["max"].get<std::vector<int>>()[0];
    stOptions.optWidth = jModelConfig[m_strModelName][JN_OPT_PROFILE]["opt"].get<std::vector<int>>()[2];
    stOptions.optHeight = jModelConfig[m_strModelName][JN_OPT_PROFILE]["opt"].get<std::vector<int>>()[1];
    stOptions.minWidth = jModelConfig[m_strModelName][JN_OPT_PROFILE]["min"].get<std::vector<int>>()[2];
    stOptions.minHeight = jModelConfig[m_strModelName][JN_OPT_PROFILE]["min"].get<std::vector<int>>()[1];
    stOptions.maxWidth = jModelConfig[m_strModelName][JN_OPT_PROFILE]["max"].get<std::vector<int>>()[2];
    stOptions.maxHeight = jModelConfig[m_strModelName][JN_OPT_PROFILE]["max"].get<std::vector<int>>()[1];
    shpTRTRunner = std::make_shared<TRTRunner>(strModelPath, stOptions);
    shpTRTRunner->allocateIOBuffers(buffers, umpIOTensors, umpIOTensorsShape);
}

TRTModel::~TRTModel()
{
    for (int i = 0; i < this->buffers.size(); i++)
    {
        if (this->buffers[i] != nullptr)
        {
            // free(this->buffers[i]);
            checkCudaErrorCode(cudaFree(this->buffers[i]));
        }
    }
    this->buffers.clear();
}

void TRTModel::setScoreThreshold(float fScoreThreshold)
{
    m_fScoreThreshold = fScoreThreshold;
}

void TRTModel::setNMSThreshold(float fNMSThreshold)
{
    m_fNMSThreshold = fNMSThreshold;
}

void TRTModel::L2Normalize(std::vector<float>& feature) {
    float sum = 0.0f;
    for (const auto& val : feature) {
        sum += val * val;
    }
    float norm = std::sqrt(sum);
    for (auto& val : feature) {
        val /= norm;
    }
}