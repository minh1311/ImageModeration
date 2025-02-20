#include "TRTYoloV5.hpp"

#include <algorithm>
#include <iostream>

TRTYoloV5::TRTYoloV5(nlohmann::json& jModelConfig) : TRTModel(jModelConfig, JN_MODEL_YOLOV5)
{
    m_iWidthModel = umpIOTensorsShape[m_strInputName].d[3];
    m_iHeightModel = umpIOTensorsShape[m_strInputName].d[2];
    
    m_iOutputWidth = umpIOTensorsShape[m_strOutputName].d[2];
    m_iOutputHeight = umpIOTensorsShape[m_strOutputName].d[1];
}

TRTYoloV5::~TRTYoloV5()
{

}


void TRTYoloV5::run(cv::Mat& mImage, std::vector<stObject_t>& stObjects)
{
    preprocess(mImage);
    shpTRTRunner->runModel(this->buffers);
    postprocess(this->buffers, stObjects);
}

void TRTYoloV5::preprocess(cv::Mat& mImage)
{
    m_iInputWidth = mImage.cols;
    m_iInputHeight = mImage.rows;

    m_fRatioWidth = 1.0f / (m_iWidthModel / static_cast<float>(m_iInputWidth));
    m_fRatioHeight = 1.0f / (m_iHeightModel / static_cast<float>(m_iInputHeight));

    cv::cuda::GpuMat mGpuImage;
    mGpuImage.upload(mImage);

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
    cv::cuda::split(mGpuFloat, mGpuChannels);

    cudaMemcpy(buffers[umpIOTensors[m_strInputName][0]], mGpuTranspose.ptr<float>(), umpIOTensors[m_strInputName][1] * sizeof(float), cudaMemcpyHostToDevice);
}

void TRTYoloV5::postprocess(std::vector<void*>& buffers, std::vector<stObject_t>& stObjects)
{
    stObjects.clear();

    float output[umpIOTensors[m_strOutputName][1]];
    cudaMemcpy(output, buffers[umpIOTensors[m_strOutputName][0]], umpIOTensors[m_strOutputName][1] * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> classes;

    for (int i = 0; i < umpIOTensors[m_strOutputName][1]; i += m_iOutputWidth)
    {
        float fScore = output[i + 4];
        if (fScore >= m_fScoreThreshold)
        {
            std::vector<float> fScores;
            for (int j = 5; j < m_iOutputWidth; j++)
            {
                fScores.push_back(output[i + j]);
            }
            auto maxScorePtr = std::max_element(fScores.begin(), fScores.end());
            float fScoreClass = *maxScorePtr;
            int iId = std::distance(fScores.begin(), maxScorePtr);

            float fScoreObject = fScore * fScoreClass;
            if (fScoreObject >= m_fScoreThreshold)
            {
                float x = output[i + 0];
                float y = output[i + 1];
                float w = output[i + 2];
                float h = output[i + 3];

                float x0 = std::clamp((x - 0.5f * w) * m_fRatioWidth, 0.f, (float)m_iInputWidth);
                float y0 = std::clamp((y - 0.5f * h) * m_fRatioHeight, 0.f, (float)m_iInputHeight);
                float x1 = std::clamp((x + 0.5f * w) * m_fRatioWidth, 0.f, (float)m_iInputWidth);
                float y1 = std::clamp((y + 0.5f * h) * m_fRatioHeight, 0.f, (float)m_iInputHeight);

                cv::Rect_<float> bbox;
                bbox.x = x0;
                bbox.y = y0;
                bbox.width = x1 - x0;
                bbox.height = y1 - y0;

                bboxes.push_back(bbox);
                scores.push_back(fScoreObject);
                classes.push_back(iId);
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxesBatched(bboxes, scores, classes, m_fScoreThreshold, m_fNMSThreshold, indices);

    int cnt = 0;
    for (auto& chosenIdx : indices)
    {
        stObject_t obj;
        obj.rfBox = bboxes[chosenIdx];
        obj.fScore = scores[chosenIdx];
        obj.iId = classes[chosenIdx];
        obj.strLabel = std::to_string(classes[chosenIdx]);
        stObjects.push_back(obj);

        cnt += 1;
    }
}
