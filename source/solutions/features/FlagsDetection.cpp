#include "FlagsDetection.hpp"
FlagsDetection::FlagsDetection(nlohmann::json& jConfigModel, std::function<void (std::vector<stObject_t> &, cv::Mat &)> fnCallbackEvent)
    : m_fnCallbackEvent(fnCallbackEvent)
{
    m_pFlagsDetection = new AIModel(jConfigModel, JN_MODEL_YOLOV8)
}

FlagsDetection::~FlagsDetection()
{
    if (m_pFlagsDetection != nullptr)
    {
        delete m_pFlagsDetection;
        m_pFlagsDetection = nullptr;
    }
}

void FlagsDetection::process(cv::Mat& mImage)
{
    std::vector<stObject_t> stOutputs;
    std::vector<int> iIndicesToRemove;

    m_pFlagsDetection->run(mImage, stOutputs);

    for(int i=0; i < stOutputs.size(); i++)
    {
        
        cv::rectangle(mImage, stOutputs[i].rfBox, cv::Scalar(0, 0, 255), 2);
    }

    m_fnCallbackEvent(stOutputs, mImage);
}