//
// Created by hoanglm on 15/05/2024.
//
#include "TRTAdaface.hpp"
#include <iostream>
TRTAdaface::TRTAdaface(nlohmann::json& jModelConfig) : TRTModel(jModelConfig, JN_MODEL_ADAFACE)
{
    m_strInputName = jModelConfig[JN_MODEL_ADAFACE]["input"].get<std::string>();
    m_strOutput = jModelConfig[JN_MODEL_ADAFACE]["output"].get<std::string>();
    m_iWidthModel = umpIOTensorsShape[m_strInputName].d[3];
    m_iHeightModel = umpIOTensorsShape[m_strInputName].d[2];
    std::cout <<"m_iWidthModel1: " << m_iWidthModel << std::endl;
    std::cout <<"m_iHeightModel1: " << m_iHeightModel << std::endl;
}

TRTAdaface::~TRTAdaface(){}

void TRTAdaface::preprocess(cv::Mat &mImage) {
    m_iInputWidth = mImage.cols;
    m_iInputHeight = mImage.rows;

//    m_fRatioWidth = 1.0f / (m_iWidthModel / static_cast<float>(m_iInputWidth));
//    m_fRatioHeight = 1.0f / (m_iHeightModel / static_cast<float>(m_iInputHeight));

    cv::cuda::GpuMat mGpuImage;
    mGpuImage.upload(mImage);


    cv::cuda::resize(mGpuImage, mGpuImage, cv::Size(m_iWidthModel, m_iHeightModel));

//    cv::cuda::cvtColor(mGpuImage, mGpuImage, cv::COLOR_BGR2RGB);

    cv::cuda::GpuMat mGpuFloat;
    mGpuImage.convertTo(mGpuFloat, CV_32FC3, 1.f/255.f);

    // cv::cuda::subtract(mGpuFloat, cv::Scalar(m_subVals[0], m_subVals[1], m_subVals[2]), mGpuFloat, cv::noArray(), -1);
    // cv::cuda::divide(mGpuFloat, cv::Scalar(m_divVals[0], m_divVals[1], m_divVals[2]), mGpuFloat, 1, -1);

    cv::cuda::GpuMat mGpuTranspose(m_iHeightModel, m_iWidthModel, CV_32FC3);
    size_t size = m_iWidthModel * m_iHeightModel * sizeof(float);
    std::vector<cv::cuda::GpuMat> mGpuChannels
            {
                    cv::cuda::GpuMat(m_iHeightModel, m_iWidthModel, CV_32FC1, &(mGpuTranspose.ptr()[0])),
                    cv::cuda::GpuMat(m_iHeightModel, m_iWidthModel, CV_32FC1, &(mGpuTranspose.ptr()[size])),
                    cv::cuda::GpuMat(m_iHeightModel, m_iWidthModel, CV_32FC1, &(mGpuTranspose.ptr()[size * 2]))
            };
    cv::cuda::split(mGpuFloat, mGpuChannels);
//    for (int i = 0; i < 10; ++i) {
//        std::cout << mGpuTranspose.ptr<float>()[0] << std::endl;
//    }
    cudaMemcpy(buffers[umpIOTensors[m_strInputName][0]], mGpuTranspose.ptr<float>(), umpIOTensors[m_strInputName][1] * sizeof(float), cudaMemcpyHostToDevice);
}

void TRTAdaface::run(const cv::Mat &image, std::vector<float>& objects) {
    std::cout << "run"<< std::endl;
    cv::Mat img= image.clone();
    preprocess(img);
    std::cout << "run2"<< std::endl;
    shpTRTRunner->runModel(this->buffers);
    postprocess(this->buffers, objects);
    std::cout << "run3"<< std::endl;

}

void TRTAdaface::postprocess(std::vector<void *> &buffer, std::vector<float>& objects) {
    std::cout << m_strOutput<< std::endl;
    float output[umpIOTensors[m_strOutput][1]];
    cudaMemcpy(output, buffers[umpIOTensors[m_strOutput][0]], umpIOTensors[m_strOutput][1] * sizeof(float), cudaMemcpyDeviceToHost);
    float quality[umpIOTensors[m_strOutputDiv704][1]];
    std::vector<float> feature(output, output + 512);
    L2Normalize(feature);
    objects = feature;
    
}
