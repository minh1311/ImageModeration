/**
 * @file TRTFaceQualityNet.cpp
 * @author HuyNQ
 * @brief 
 * @version 0.1
 * @date 2024-06-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "TRTFaceQualityNet.hpp"

#include <iostream>
TRTFaceQualityNet::TRTFaceQualityNet(nlohmann::json& jModelConfig) : TRTModel(jModelConfig, JN_MODEL_FACEQNET)
{
    m_iWidthModel = umpIOTensorsShape[m_strInputName].d[3];
    m_iHeightModel = umpIOTensorsShape[m_strInputName].d[2];
    std::cout <<"m_iWidthModel1: " << m_iWidthModel << std::endl;
    std::cout <<"m_iHeightModel1: " << m_iHeightModel << std::endl;
}

TRTFaceQualityNet::~TRTFaceQualityNet(){}

void TRTFaceQualityNet::preprocess(cv::Mat &mImage) {
    m_iInputWidth = mImage.cols;
    m_iInputHeight = mImage.rows;

    cv::cuda::GpuMat mGpuImage;
    mGpuImage.upload(mImage);


    cv::cuda::resize(mGpuImage, mGpuImage, cv::Size(m_iWidthModel, m_iHeightModel));

//    cv::cuda::cvtColor(mGpuImage, mGpuImage, cv::COLOR_BGR2RGB);

    cv::cuda::GpuMat mGpuFloat;
    mGpuImage.convertTo(mGpuFloat, CV_32FC3, 1.f/255.f);

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

void TRTFaceQualityNet::run(cv::Mat &image, float &conf) {
    preprocess(image);
    shpTRTRunner->runModel(this->buffers);
    conf = postprocess(this->buffers);
}

float TRTFaceQualityNet::postprocess(std::vector<void *> &buffer) {
    float output[umpIOTensors[m_strOutput][1]];
    cudaMemcpy(output, buffers[umpIOTensors[m_strOutput][0]], umpIOTensors[m_strOutput][1] * sizeof(float), cudaMemcpyDeviceToHost);
    return output[0];
}
