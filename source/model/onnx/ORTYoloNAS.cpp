#include "ORTYoloNAS.hpp"

ORTYoloNAS::ORTYoloNAS(nlohmann::json& jModelConfig)
    : ORTModel(jModelConfig, JN_MODEL_YOLONAS)
{
    m_iWidthModel = umpInputTensorsShape[m_strInputName][3];
    m_iHeightModel = umpInputTensorsShape[m_strInputName][2];
    
    setScoreThreshold(jModelConfig[JN_MODEL_YOLONAS][JN_CONFIDENCE_THRESHOLD].get<float>());
    setNMSThreshold(jModelConfig[JN_MODEL_YOLONAS][JN_NMS_THRESHOLD].get<float>());
}

ORTYoloNAS::~ORTYoloNAS()
{
    
}

void ORTYoloNAS::run(cv::Mat& mImage, std::vector<stObject_t>& stObjects)
{
    preprocess(mImage);
    shpORTRunner->runModel(inputOrtValues, outputOrtValues);
    postprocess(stObjects);
}

void ORTYoloNAS::preprocess(cv::Mat& mImage)
{
    m_iInputWidth = mImage.cols;
    m_iInputHeight = mImage.rows;

    m_fRatioWidth = 1.0f / (m_iWidthModel / static_cast<float>(m_iInputWidth));
    m_fRatioHeight = 1.0f / (m_iHeightModel / static_cast<float>(m_iInputHeight));

    cv::Mat mInput;
    cv::resize(mImage, mInput, cv::Size(m_iWidthModel, m_iHeightModel), 0, 0, cv::INTER_LINEAR);

    cv::cvtColor(mInput, mInput, cv::COLOR_BGR2RGB);
    cv::Mat mFloat;
    mInput.convertTo(mFloat, CV_32FC3, 1.f / 255.f);

    cv::Mat mChannels[3];
    cv::split(mFloat, mChannels);

    inputOrtValues.clear();
    for (auto& channel : mChannels)
    {
        std::vector<float> fVec(channel.begin<float>(), channel.end<float>());
        inputOrtValues.insert(inputOrtValues.end(), fVec.begin(), fVec.end());
    }
}

void ORTYoloNAS::postprocess(std::vector<stObject_t>& stObjects)
{
    stObjects.clear();

    std::vector<float> fOutputSigmoid = outputOrtValues[umpOutputTensors[m_strOutputSigmoid][0]];
    std::vector<float> fOutputMul = outputOrtValues[umpOutputTensors[m_strOutputMul][0]];

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> classes;

    m_iOutputWidth = fOutputSigmoid.size() / m_iNumClasses;
    for (int i = 0; i < m_iOutputWidth; i++)
    {
        std::vector<float> fScores;
        for (int j = 0; j < m_iNumClasses; j++)
        {
            fScores.push_back(fOutputSigmoid[i * m_iNumClasses + j]);
        }
        auto maxScorePtr = std::max_element(fScores.begin(), fScores.end());
        float fScore = *maxScorePtr;
        int iId = std::distance(fScores.begin(), maxScorePtr);
        if (fScore >= m_fScoreThreshold)
        {
            float x1 = fOutputMul[i * 4 + 0];
            float y1 = fOutputMul[i * 4 + 1];
            float x2 = fOutputMul[i * 4 + 2];
            float y2 = fOutputMul[i * 4 + 3];

            float ox0 = std::clamp(x1 * m_fRatioWidth, 0.f, (float)m_iInputWidth);
            float oy0 = std::clamp(y1 * m_fRatioHeight, 0.f, (float)m_iInputHeight);
            float ox1 = std::clamp(x2 * m_fRatioWidth, 0.f, (float)m_iInputWidth);
            float oy1 = std::clamp(y2 * m_fRatioHeight, 0.f, (float)m_iInputHeight);

            cv::Rect_<float> bbox;
            bbox.x = ox0;
            bbox.y = oy0;
            bbox.width = ox1 - ox0;
            bbox.height = oy1 - oy0;

            bboxes.push_back(bbox);
            scores.push_back(fScore);
            classes.push_back(iId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxesBatched(bboxes, scores, classes, m_fScoreThreshold, m_fNMSThreshold, indices);

    // int cnt = 0;
    for (auto& chosenIdx : indices)
    {
        stObject_t obj;
        obj.rfBox = bboxes[chosenIdx];
        obj.fScore = scores[chosenIdx];
        obj.iId = classes[chosenIdx];
        obj.strLabel = std::to_string(classes[chosenIdx]);
        stObjects.push_back(obj);

        // cnt += 1;
    }
}
