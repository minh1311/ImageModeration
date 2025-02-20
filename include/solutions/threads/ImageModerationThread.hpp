#ifndef ImageModerationThread_hpp
#define ImageModerationThread_hpp

#include <thread>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <queue>
#include "json.hpp"

#include "include/solutions/features/FlagsDetection.hpp"
#include "RestreamRtmp.hpp"
#ifdef WAYLAND
    #include "GstWayland.hpp"
#endif
#include "StreamManager.hpp"

#define TagImageModerationThread "[ImageModerationThread]"
class ImageModerationThread
{
public:
    ImageModerationThread(  const std::string& strUrl, const std::string& strAdditionalInfor,
                            nlohmann::json& jConfigModel,
                            std::function<void (nlohmann::json &, cv::Mat &)> fnCallbackEvent, 
                            std::function<void (cv::Mat &)> fnCallbackRestream, 
                            std::function<void (std::string &)> fnOnStop);

    ~ImageModerationThread();

    void start();
    void pause();
    void stop();

    void setEnable(bool status);

    // void runRestream(std::string& strRtmp);
    // void stopRestream(std::string& strRtmp);

    private:
        void run();
        void processEvent(std::vector<stObject_t>& stOutputs, cv::Mat& mEvent);

    private:
        std::thread m_thread;
        std::mutex m_mtLock;
        std::mutex m_mtCleanup;
        std::condition_variable m_cvCleanup;
        std::condition_variable m_cvSignal;
        
        std::atomic<bool> m_bEnable{false};
        std::atomic<bool> m_bRun{false};
        std::atomic<bool> m_bUpdate{false};
        std::atomic<bool> m_bThreadFinished{false};


        std::shared_ptr<Stream> m_shpStream;
        RestreamRtmp* m_pRestream = nullptr;
        #ifdef WAYLAND
            GstWayland* m_pWayland = nullptr;
        #endif
        int m_iWidth= -1, m_iHeight = -1;

        FlagsDetection* m_pFlagsDetection = nullptr;




        


        
        std::string m_strAiFlowId;
        std::string m_strUrl;
        // std::string m_strRtmp;
        nlohmann::json m_jConfigModel;

        std::function<void (nlohmann::json &, cv::Mat &)> m_fnCallbackEvent;
        std::function<void (cv::Mat &)> m_fnCallbackRestream;
        std::function<void (std::string &)> m_fnOnStop;


private:
    /* data */

};



#endif //ImageModerationThread_hpp