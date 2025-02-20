#include "TRTPplcNet.hpp"

#include <algorithm>
#include <iostream>

TRTPplcNet::TRTPplcNet(nlohmann::json& jModelConfig) : TRTModel(jModelConfig, JN_MODEL_PPLCNET)
{
    m_iWidthModel = umpIOTensorsShape[m_strInputName].d[3];
    m_iHeightModel = umpIOTensorsShape[m_strInputName].d[2];
    
    m_iOutputWidth = umpIOTensorsShape[m_strOutputName].d[1];
    m_iOutputHeight = umpIOTensorsShape[m_strOutputName].d[0];
}

TRTPplcNet::~TRTPplcNet()
{

}

void TRTPplcNet::run(cv::Mat& mImage, stHumanAttribute_t& stHumanAttribute)
{
    preprocess(mImage);
    shpTRTRunner->runModel(this->buffers);
    postprocess(this->buffers, stHumanAttribute);
}

void TRTPplcNet::preprocess(cv::Mat& mImage)
{
    cv::cuda::GpuMat mGpuImage;
    mGpuImage.upload(mImage);

    cv::cuda::resize(mGpuImage, mGpuImage, cv::Size(m_iWidthModel, m_iHeightModel));

    cv::cuda::cvtColor(mGpuImage, mGpuImage, cv::COLOR_BGR2RGB);

    cv::cuda::GpuMat mGpuFloat;
    mGpuImage.convertTo(mGpuFloat, CV_32FC3, 1.f / 255.f);

    cv::cuda::subtract(mGpuFloat, cv::Scalar(m_subVals[0], m_subVals[1], m_subVals[2]), mGpuFloat, cv::noArray(), -1);
    cv::cuda::divide(mGpuFloat, cv::Scalar(m_divVals[0], m_divVals[1], m_divVals[2]), mGpuFloat, 1, -1);

    cv::cuda::GpuMat mGpuTranspose(m_iHeightModel, m_iWidthModel, CV_32FC3);
    size_t size = m_iWidthModel * m_iHeightModel * sizeof(float);
    std::vector<cv::cuda::GpuMat> mGpuChannels
    {
        cv::cuda::GpuMat(m_iHeightModel, m_iWidthModel, CV_32FC1, &(mGpuTranspose.ptr()[0])),
        cv::cuda::GpuMat(m_iHeightModel, m_iWidthModel, CV_32FC1, &(mGpuTranspose.ptr()[size])),
        cv::cuda::GpuMat(m_iHeightModel, m_iWidthModel, CV_32FC1, &(mGpuTranspose.ptr()[size * 2]))
    };
    cv::cuda::split(mGpuFloat, mGpuChannels);

    cudaMemcpy(buffers[umpIOTensors[m_strInputName][0]], mGpuTranspose.ptr<float>(), umpIOTensors[m_strInputName][1] * sizeof(float), cudaMemcpyHostToDevice);
}

// void TRTPplcNet::postprocess(std::vector<void*>& buffers)
// {
//     float output[umpIOTensors[m_strOutputName][1]];
//     cudaMemcpy(output, buffers[umpIOTensors[m_strOutputName][0]], umpIOTensors[m_strOutputName][1] * sizeof(float), cudaMemcpyDeviceToHost);

//     // for (int i = 0; i < m_iOutputWidth; i++)
//     // {
//     //     std::cout << i << " = " << output[i] << std::endl;
//     // }

//     stHumanAttribute_t stHumanAttribute;

//     m_fnCallback(stHumanAttribute);
// }

void TRTPplcNet::postprocess(std::vector<void*>& buffers, stHumanAttribute_t& stHumanAttribute)
{
    float output[umpIOTensors[m_strOutputName][1]];
    cudaMemcpy(output, buffers[umpIOTensors[m_strOutputName][0]], umpIOTensors[m_strOutputName][1] * sizeof(float), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < m_iOutputWidth; i++)
    // {
    //     std::cout << i << " = " << output[i] << std::endl;
    // }

    stHumanAttribute.strGender = "";
    stHumanAttribute.fScoreGender = 0.f;
}
