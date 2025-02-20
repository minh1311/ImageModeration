#include "include/solutions/threads/ImageModerationThread.hpp"

ImageModerationThread::ImageModerationThread(   const std::string& strUrl, const std::string& strAdditionalInfor,
                                                nlohmann::json& jConfigModel,
                                                std::function<void (nlohmann::json &, cv::Mat &)> fnCallbackEvent, 
                                                std::function<void (cv::Mat &)> fnCallbackRestream, 
                                                std::function<void (std::string &)> fnOnStop)
    : m_jConfigModel(jConfigModel), m_fnCallbackEvent(fnCallbackEvent), m_fnCallbackRestream(fnCallbackRestream), m_fnOnStop(fnOnStop)
{
    nlohmann::json jAdditionalInfor = nlohmann::json::parse(strAdditionalInfor);
    m_strAiFlowId = jAdditionalInfor["ai_flow_id"].get<std::string>();

    m_strUrl = strUrl;

    m_bEnable = true;
    m_bRun = true;
}

ImageModerationThread::~ImageModerationThread()
{
    if (m_p) != nullptr
    {
        delete m_p;
        m_p = nullptr;
    }
}


void ImageModerationThread::start()
{
    m_thread = std::thread(&ImageModerationThread::run, this);
    m_thread.detach();
    std::cout << TagImageModerationThread << "Start ImageModerationThread, ai_flow_id: " << m_strAiFlowId << std::endl;
}

void ImageModerationThread::pause()
{
    const std::lock_guard<std::mutex> lgLock(m_mtLock);
    m_bRun = false;
    m_cvSignal.notify_all();
    std::cout << TagImageModerationThread << "Pause ImageModerationThread, ai_flow_id: " << m_strAiFlowId << std::endl;
}


void ImageModerationThread::stop()
{
    {
        std::lock_guard<std::mutex> lgLock(m_mtLock);
        m_bEnable = false;
    }
    m_cvSignal.notify_all();
    std::cout << TagImageModerationThread << "Stop ImageModerationThread, ai_flow_id: " << m_strAiFlowId << std::endl;
}

void ImageModerationThread::setEnable(bool status)
{
    {
        std::lock_guard<std::mutex> lgLock(m_mtLock);
        m_bEnable = status;
    }
    m_cvSignal.notify_all();
    std::cout << TagImageModerationThread << "set status ImageModerationThread, ai_flow_id: " << m_strAiFlowId << std::endl;
}


void ImageModerationThread::runRestream(std::string& strRtmp)
{
    std::unique_lock<std::mutex> lock(m_mtLock);
    m_bRun= false;
    m_cvSignal.wait(lock, [this](){ return !m_bUpdate;});
    if (m_pRestream == nullptr)
    {
        m_pRestream = new RestreamRtmp(strRtmp);
    }
    std::cout << TagImageModerationThread << "(Play Restream) ImageModerationThread successfully, ai_flow_id: " << m_strAiFlowId << std::endl;
    m_bRun = true;
}

void ImageModerationThread::stopRestream(std::string& strRtmp)
{
    std::unique_lock<std::mutex> lock(m_mtLock);
    m_bRun.store(false);
    m_cvSignal.wait(lock, [this](){ return !m_bUpdate;});
    sleep(1);
    if (m_pRestream!=nullptr)
    {
        delete m_pRestream;
        m_pRestream= nullptr;
    }
    std::cout << TagImageModerationThread << "(Stop Restream) ImageModerationThread successfully, ai_flow_id: " << m_strAiFlowId << std::endl;
    m_bRun.store(true);
    // m_pRestream->pauseRestream();
}


void ImageModerationThread::run()
{
    while (true)
    {   
        std::unique_lock<std::mutex> lock(m_mtLock);
        if (m_bEnable == false)
        {
            lock.unlock();
            break;
        }
        if (m_bRun== false)
        {
            m_bUpdate= true;
            lock.unlock();
            continue;
        }
        lock.unlock();
        m_bUpdate= false;

        if (m_shpStream == nullptr) {
            // m_shpStream = new GstDecode(m_strUrl);
            m_shpStream = StreamManager::GetInstance().getStream(m_strUrl);
            if (m_shpStream != nullptr){
                m_iWidth = m_shpStream->getWidth();
                m_iHeight = m_shpStream->getHeight();
                std::cout << "stream: "<< m_iWidth << "--"<< m_iHeight<< std::endl;
            }else {
                sleep(1);
                continue;
            }
        }


        #ifdef WAYLAND
            if (m_pWayland == nullptr)
            {
                std::array<int, 4> arGrid = StreamManager::GetInstance().getGrid(m_strUrl);
                m_pWayland = new GstWayland(m_strUrl, arGrid);
            }
        #endif

        if (m_pFlagsDetection == nullptr)
        {
            m_pFlagsDetection = new FlagsDetection(m_jConfigModel, 
                                std::bind(&ImageModerationThread::processEvent, this, std::placeholders::_1, std::placeholders::_2));
        }

        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat mFrame;
        if (m_shpStream== nullptr){
            continue;
        }
        m_shpStream->getFrame(mFrame);
        if (mFrame.empty())
        {
            continue;
        }


        // cv::Mat mEvent = mFrame.clone();
        m_pFlagsDetection->process(mFrame);

        auto end = std::chrono::high_resolution_clock::now();

        // Tính toán thời gian
        std::chrono::duration<double> duration = end - start;
        std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

        if (m_pRestream != nullptr)
        {
            m_pRestream->setFrame(mFrame);
        }

        #ifdef WAYLAND
            m_pWayland->setFrame(mFrame);
        #endif

        mFrame.release();    
        // mEvent.release();

        m_bUpdate= true;
    }
    if (m_shpStream!= nullptr){

        StreamManager::GetInstance().unregister(m_strUrl);
        m_shpStream.reset();
    }
    m_fnOnStop(m_strAiFlowId);
    std::cout<<"THREAD:: stop thread"<<std::endl;
}

void ImageModerationThread::processEvent(std::vector<stObject_t>& stOutputs, cv::Mat& mEvent)
{
    nlohmann::json jEvent;

    // cv::Mat m_Event_draw;

    for(int i=0; i < stOutputs.size(); i++)
    {
        if (stOutputs[i].iId == 0)
        {
            jEvent = {};
            jEvent["type"] = "FIRE";

            // m_Event_draw = mEvent.clone();
            // cv::rectangle(m_Event_draw, stOutputs[i].rfBox, cv::Scalar(0, 0, 255), 2);

            jEvent["label"] = stOutputs[i].iId;
            jEvent["score"] = stOutputs[i].fScore;
            jEvent["bbox"] = {stOutputs[i].rfBox.x, stOutputs[i].rfBox.y, stOutputs[i].rfBox.width, stOutputs[i].rfBox.height};

            m_fnCallbackEvent(jEvent, mEvent);
        }
        else if (stOutputs[i].iId == 1)
        {
            jEvent = {};
            jEvent["type"] = "SMOKE";

            // m_Event_draw = mEvent.clone();
            // cv::rectangle(m_Event_draw, stOutputs[i].rfBox, cv::Scalar(0, 255, 255), 2);

            jEvent["label"] = stOutputs[i].iId;
            jEvent["score"] = stOutputs[i].fScore;
            jEvent["bbox"] = {stOutputs[i].rfBox.x, stOutputs[i].rfBox.y, stOutputs[i].rfBox.width, stOutputs[i].rfBox.height};

            m_fnCallbackEvent(jEvent, mEvent);
        }
        else
            ;
    }
}
