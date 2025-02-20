#ifndef ImageModerationSolution_hpp
#define ImageModerationSolution_hpp

#include "json.hpp"
#include "threads/ImageModerationThread.hpp"


#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>


#define TagImageModerationSolution "[ImageModerationSolution]"

class ImageModerationSolution 
{
public:
    ImageModerationSolution(const std::string &strConfigModel);
    ~ImageModerationSolution();

    void predictStream(const std::string& strUrl, const std::string& strAdditionalInfor, 
        std::function<void (nlohmann::json &, cv::Mat &)> fnCallbackEvent, 
        std::function<void (cv::Mat &)> fnCallbackRestream);

    void stopStream(const std::string& strAdditionalInfor);

    void onStop(const std::string strAiFlowId);


    

private:
    std::map<std::string, std::unique_ptr<ImageModerationThread>> m_mpImageModerationThread;

    bool isEnable = false;

    nlohmann::json m_jConfigModel;





};
#endif //ImageModerationSolution_hpp