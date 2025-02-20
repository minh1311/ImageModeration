#include "include/solutions/ImageModerationSolution.hpp"

ImageModerationSolution::ImageModerationSolution(const std::string &strConfigModel)
{
    nlohmann::json jConfigModel = nlohmann::json::parse(strConfigModel);
    m_jConfigModel = jConfigModel;

    isEnable = true;
}

ImageModerationSolution::~ImageModerationSolution()
{
    ;
}

void ImageModerationSolution::predictStream(const std::string& strUrl, const std::string& strAdditionalInfor, 
                                                std::function<void (nlohmann::json &, cv::Mat &)> fnCallbackEvent, 
                                                std::function<void (cv::Mat &)> fnCallbackRestream)
{
    if (!isEnable)
    {
        std::cout << TagImageModerationSolution << "Solution is OFF" << std::endl;
        return;
    }

    nlohmann::json jAdditionalInfor = nlohmann::json::parse(strAdditionalInfor);
    std::string strAiFlowId = jAdditionalInfor["ai_flow_id"].get<std::string>();
    std::string m_strUrl;
    if(strAiFlowId == "")
        m_strUrl = "";
    else 
        m_strUrl = strUrl

    if (m_mpImageModerationThread.find(strAiFlowId) == m_mpImageModerationThread.end())
    {
        std::cout<< TagImageModerationSolution << "Create ImageModerationThread:\t ai_flow_id: "<< strAiFlowId <<std::endl;
        std::cout<< TagImageModerationSolution << "\t URL: "<<m_strUrl<<std::endl;
        std::cout << TagImageModerationSolution << "\tAdditionalInfor: " << jAdditionalInfor.dump() << std::endl;

        //  note thieu
        std::unique_ptr<ImageModerationThread> pImageModerationThread = std::make_unique<ImageModerationThread>();
        m_mpImageModerationThread[strAiFlowId] = std::move(pImageModerationThread);
    }
    else
    {
        std::cout << TagImageModerationSolution << "ImageModerationThread with ai_flow_id: " << strAiFlowId << " already exist!" << std::endl;
        m_mpImageModerationThread[strAiFlowId]->setEnable(true);
    }
}

void ImageModerationSolution::stopStream(const std::string& strAdditionalInfor)
{
    nlohmann::json jAdditionalInfor = nlohmann::json::parse(strAdditionalInfor);
    std::string strAiFlowId = jAdditionalInfor["ai_flow_id"].get<std::string>();

    auto it = m_mpImageModerationThread.find(strAiFlowId);
    if (it != m_mpImageModerationThread.end())
    {
        std::cout<< TagImageModerationSolution<< "stop ImageModerationThread with ai_flow_id: "<< strAiFlowId <<std::endl;
        it->second->stop();
    }

    else
    {
        std::cout<< TagImageModerationSolution << "ImageModeration with ai_flow_id: "<< strAiFlowId << "does not exit!" << std::endl;
    }
    
}

void ImageModerationSolution::onStop(const std::string strAiFlowId)
{
    auto it = m_mpImageModerationThread.find(strAiFlowId);
    if (it != m_mpImageModerationThread.end())
    {
        std::cout << TagImageModerationSolution << "Stop ImageModerationThread with ai_flow_id: " << strAiFlowId << std::endl;
        m_mpImageModerationThread.erase(it);
    }
    else
    {
        std::cout << TagImageModerationSolution << "ImageModerationThread with ai_flow_id: " << strAiFlowId << " does not exist!" << std::endl;
    }
}
