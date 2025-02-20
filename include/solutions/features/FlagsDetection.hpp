#ifndef FlagsDetection_hpp
#define FlagsDetection_hpp

#include "AIModel.hpp"

class FlagsDetection
{
public:
    FlagsDetection(nlohmann::json& jConfigModel, std::function<void (std::vector<stObject_t> &, cv::Mat &)> fnCallbackEvent);
    ~FlagsDetection();
    
    void process(cv::Mat& mImage);
    
private:
    AIModel* m_pFlagsDetection = nullptr;

    std::function<void (std::vector<stObject_t> &, cv::Mat &)> m_fnCallbackEvent;
};



#endif //FlagsDetection_hpp