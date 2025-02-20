//
// Created by hoanglm on 09/04/2024.
//
#include "TRTPPOcrRec.hpp"

#include <iostream>

TRTPPOcrRec::TRTPPOcrRec(nlohmann::json& jModelConfig) : TRTModel(jModelConfig, JN_MODEL_PPOCR_REC) 
{
    m_iWidthModel = umpIOTensorsShape[m_strInputName].d[3];
    m_iHeightModel = umpIOTensorsShape[m_strInputName].d[2];

    m_iOutputWidth = umpIOTensorsShape[m_strOutputName].d[2];
    m_iOutputHeight = umpIOTensorsShape[m_strOutputName].d[1];
    std::string strLabelPath = jModelConfig[JN_ASSETS_DIR].get<std::string>() + jModelConfig[JN_MODEL_PPOCR_REC][JN_LABEL_NAME].get<std::string>();
    setLabels(strLabelPath);
}

TRTPPOcrRec::~TRTPPOcrRec() {}

std::vector<std::string> TRTPPOcrRec::readDict(const std::string &path) {
    std::ifstream in(path);
    std::string line;
    std::vector<std::string> m_vec;
    if (in) {
        while (getline(in, line)) {
            m_vec.push_back(line);
//            std::cout << "------------------line-----------: " << line << std::endl;
        }
    } else {
        std::cout << "no such label file: " << path << ", exit the program..."
                  << std::endl;
        exit(1);
    }
    return m_vec;
}

void TRTPPOcrRec::preprocess(cv::Mat &mImage) {

    m_iInputWidth = mImage.cols;
    m_iInputHeight = mImage.rows;

    m_fRatioWidth = 1.0f / (m_iWidthModel / static_cast<float>(m_iInputWidth));
    m_fRatioHeight = 1.0f / (m_iHeightModel / static_cast<float>(m_iInputHeight));

    cv::cuda::GpuMat mGpuImage;

    mGpuImage.upload(mImage.clone());

    cv::cuda::resize(mGpuImage, mGpuImage, cv::Size(m_iWidthModel, m_iHeightModel));
    cv::cuda::cvtColor(mGpuImage, mGpuImage, cv::COLOR_BGR2RGB);

    cv::cuda::GpuMat mGpuFloat;
    mGpuImage.convertTo(mGpuFloat, CV_32FC3, 1.f / 255.f);

    cv::cuda::GpuMat mGpuTranspose(m_iHeightModel, m_iWidthModel, CV_32FC3);
    size_t size = m_iWidthModel * m_iHeightModel * sizeof(float);
    std::vector<cv::cuda::GpuMat> mGpuChannels
    {
        cv::cuda::GpuMat(m_iHeightModel, m_iWidthModel, CV_32FC1, &(mGpuTranspose.ptr()[0])),
        cv::cuda::GpuMat(m_iHeightModel, m_iWidthModel, CV_32FC1, &(mGpuTranspose.ptr()[size])),
        cv::cuda::GpuMat(m_iHeightModel, m_iWidthModel, CV_32FC1, &(mGpuTranspose.ptr()[size * 2]))
    };
    cv::cuda::split(mGpuFloat, mGpuChannels); // HWC -> CHW

    float* data = mGpuTranspose.ptr<float>();

    cudaMemcpy(buffers[umpIOTensors[m_strInputName][0]], data, umpIOTensors[m_strInputName][1] * sizeof(float), cudaMemcpyHostToDevice);
}

void TRTPPOcrRec::postprocess(std::vector<void *> &buffers, stTextRec_t &textObject)
{
    float output[umpIOTensors[m_strOutputName][1]];
    cudaMemcpy(output, buffers[umpIOTensors[m_strOutputName][0] ], umpIOTensors[m_strOutputName][1] * sizeof(float) , cudaMemcpyDeviceToHost);

    // std::pair<std::vector<std::string>, double> temp_box_res;
    // std::vector<std::string> str_res;
    int argmax_idx;
    int last_index = 0;
    float score = 0.f;
    int count = 0;
    float max_value = 0.0f;
    std::string res = "";

    for (int n = 0; n < m_iOutputHeight; n++)
    {
        argmax_idx = std::distance(&output[(n) * m_iOutputWidth], std::max_element(&output[(n) * m_iOutputWidth], &output[(n + 1) * m_iOutputWidth]));

        max_value = float(*std::max_element(&output[( n) * m_iOutputWidth], &output[( n + 1) * m_iOutputWidth]));
        if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
            score += max_value;
            count += 1;
            res += label_list_[argmax_idx];
        }
        last_index = argmax_idx;
    }

    textObject.text = res;
    if (count > 0)
        textObject.fScore = score / count;
    else
        textObject.fScore = score;
}

template <class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

void TRTPPOcrRec::run(cv::Mat& mImage, stTextRec_t& stTextRec)
{
    preprocess(mImage);

    shpTRTRunner->runModel(this->buffers);
    postprocess(this->buffers, stTextRec);
}

void TRTPPOcrRec::setLabels(std::string &strPath) {
    label_list_ = this->readDict(strPath);
    label_list_.insert(label_list_.begin(), "#");
    label_list_.push_back(" ");
}