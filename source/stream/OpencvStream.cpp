#include "stream/OpencvStream.hpp"

OpencvStream::OpencvStream(std::string strUrl)
{
    m_strUrl = strUrl;
    m_bEnable = true;
    m_bRun = true;

    #if DEV_VIDEO
    m_videoCapture.open(m_strUrl);
    #else
    m_videoCapture.open(m_strUrl);
    if (m_videoCapture.isOpened() == false)
        return ;
    
    double dFourcc = m_videoCapture.get(cv::CAP_PROP_FOURCC);
    const char* pCodec;
    int iFourcc = static_cast<int>(dFourcc);
    if (iFourcc == 0)
        pCodec = nullptr;
    
    static char arCodec[5];
    for (int i=0; i < 4; i++)
    {
        arCodec[i] = static_cast<char>((iFourcc >> (8 * i)) & 0x0FF);
    }
    arCodec[4] = '\0';
    pCodec = arCodec;

    m_videoCapture.release();

    if (pCodec != nullptr) {
        std::string rtspsrcNode = "rtspsrc latency=0 protocols=tcp location=" + m_strUrl;
        std::string decoderNode = "nvv4l2decoder ! nvvideoconvert ! video/x-raw,width=1920,height=1080,format=BGR ! appsink";
        std::string pipeline = rtspsrcNode + " ! rtph264depay ! " + decoderNode;

        std::string codecNameStr(pCodec);
        if (codecNameStr.find("h264") != std::string::npos) {
            std::cout << "RTSP stream is encoded with H.264" << std::endl;
            m_videoCapture.open(pipeline, cv::CAP_GSTREAMER);
        } else if (codecNameStr.find("hevc") != std::string::npos) {
            std::cout << "RTSP stream is encoded with H.265 (HEVC)" << std::endl;
            pipeline = rtspsrcNode + " ! rtph265depay ! " + decoderNode;
            m_videoCapture.open(pipeline, cv::CAP_GSTREAMER);
        } else {
            m_videoCapture.open(m_strUrl);
        }
    }
    else
    {
        m_videoCapture.open(m_strUrl);
    }
    #endif // DEV_VIDEO

    if (!m_videoCapture.isOpened())
    {
        std::cout << "Camera not opened!" << std::endl;
        // m_bEnable = false;
        return ;
    }

    pthread_create(&m_pthread, NULL, opencvStreamFunc, this);
    pthread_detach(m_pthread);
}

OpencvStream::~OpencvStream()
{
    pthread_cancel(m_pthread);
    m_videoCapture.release();

    while (m_quFrame.getSize() > 0)
    {
        cv::Mat mFrame;
        m_quFrame.tryWaitAndPop(mFrame, 50);
    }
}

std::string OpencvStream::getUrl()
{
    return m_strUrl;
}

int OpencvStream::getWidth()
{
    return m_videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
}

int OpencvStream::getHeight()
{
    return m_videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
}

void OpencvStream::getFrame(cv::Mat& mFrame)
{
    m_quFrame.tryWaitAndPop(mFrame, 50);
}

void OpencvStream::runStream()
{
    pthread_mutex_lock(&m_mtLock);
    m_bRun = true;
    pthread_mutex_unlock(&m_mtLock);
}

void OpencvStream::pauseStream()
{
    pthread_mutex_lock(&m_mtLock);
    m_bRun = false;
    pthread_mutex_unlock(&m_mtLock);
}

void OpencvStream::stopStream()
{
    pthread_mutex_lock(&m_mtLock);
    m_bEnable = false;
    pthread_mutex_unlock(&m_mtLock);
}

void OpencvStream::join()
{
    pthread_join(m_pthread, NULL);
}

void* OpencvStream::opencvStreamFunc(void* arg)
{
    OpencvStream* pStream = (OpencvStream*)arg;
    pStream->runStreamThread();
    pthread_exit(NULL);
}

void OpencvStream::runStreamThread()
{
    while (true)
    {
        if (m_bEnable == false)
            break;

        if (m_bRun == false)
            continue;

        #if DEV_VIDEO
        std::this_thread::sleep_for(std::chrono::milliseconds(70));
        #endif // DEV_VIDEO

        cv::Mat mFrame;
        m_videoCapture >> mFrame;
        if (mFrame.empty())
            continue;
        if (m_quFrame.getSize() < m_iMaxQueueSize)
        {
            m_quFrame.push(mFrame);
        }
        else
        {
            cv::Mat mTemp;
            m_quFrame.tryWaitAndPop(mTemp, 50);
            m_quFrame.push(mFrame);
        }
    }
}
