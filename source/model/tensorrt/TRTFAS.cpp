/**
 * @file TRTFAS.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-07-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "TRTFAS.hpp"
#include <iostream>
TRTFas::TRTFas(nlohmann::json& jModelConfig) : TRTModel(jModelConfig, JN_MODEL_FAS)
{
    m_iWidthModel = umpIOTensorsShape[m_strInputName].d[3];
    m_iHeightModel = umpIOTensorsShape[m_strInputName].d[2];
    m_fScale = jModelConfig[JN_MODEL_FAS][JN_SCALE].get<float>();
    std::cout <<"m_iWidthModel1: " << m_iWidthModel << std::endl;
    std::cout <<"m_iHeightModel1: " << m_iHeightModel << std::endl;
    std::cout <<"m_fScale: " << m_fScale << std::endl;
}

TRTFas::~TRTFas(){}

void TRTFas::preprocess(cv::Mat &mImage, cerberus::CerberusFace & face ) {
    auto rect= calculateFASBox(face, mImage.cols,mImage.rows,m_fScale) ;
    if (rect.x <0) rect.x =0;
    if (rect.y <0) rect.y =0;
    if (rect.x + rect.width >mImage.cols) rect.width =mImage.cols - rect.x;
    if (rect.y + rect.height >mImage.rows) rect.height =mImage.rows - rect.y;
    cv::Mat img_crop = mImage(rect).clone();
    cv::resize(img_crop, img_crop, cv::Size(m_iWidthModel, m_iHeightModel));
    m_iInputWidth = img_crop.cols;
    m_iInputHeight = img_crop.rows;
    float* data = new float[m_iWidthModel * m_iHeightModel * 3];
    cv::Mat pr_img;
    cv::resize(img_crop, pr_img, cv::Size(m_iWidthModel, m_iHeightModel));
    int i = 0;
    for (int row = 0; row < m_iHeightModel; ++row)
    {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < m_iWidthModel; ++col)
        {
            data[i] = (float)uc_pixel[0] ;  // R
            data[i + m_iHeightModel * m_iWidthModel] = (float)uc_pixel[1] ;  // G
            data[i + 2 * m_iHeightModel * m_iWidthModel] = (float)uc_pixel[2] ;  // B
            uc_pixel += 3;
            ++i;
        }
    }
    cudaMemcpy(buffers[umpIOTensors[m_strInputName][0]], data,3 * m_iHeightModel * m_iWidthModel * sizeof(float), 
           cudaMemcpyHostToDevice);
    delete[] data;
}

void TRTFas::run(const cv::Mat &image, std::vector<float>& objects, cerberus::CerberusFace & face) {
    cv::Mat img= image.clone();
    preprocess(img, face);
    shpTRTRunner->runModel(this->buffers);
    postprocess(this->buffers, objects);
}

void TRTFas::postprocess(std::vector<void *> &buffer, std::vector<float>& objects) {
    float* output = new float[umpIOTensors[m_strOutput][1]];
    cudaMemcpy(output, buffers[umpIOTensors[m_strOutput][0]], umpIOTensors[m_strOutput][1] * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::vector<float> result = softmax(std::vector<float>(output, output + 3));
    
    objects = result;

    // std::cout << "Spoofing probabilities: " 
    //           << result[0] << " " 
    //           << result[1] << " " 
    //           << result[2] << std::endl;
    
    delete[] output;
}