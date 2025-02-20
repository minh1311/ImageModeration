#ifndef OpencvStream_hpp
#define OpencvStream_hpp

#include <iostream>
#include <string>
#include <pthread.h>
#include <thread>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio/videoio.hpp>

#include "BlockingQueue.hpp"
#include "stream/Stream.hpp"

#define DEV_VIDEO 1

class OpencvStream : public Stream
{
    public:
        OpencvStream(std::string strUrl);
        ~OpencvStream();

        std::string getUrl() override;
        int getWidth() override;
        int getHeight() override;
        void getFrame(cv::Mat& mFrame) override;

        void runStream() override;
        void pauseStream() override;
        void stopStream() override;
        void join() override;

    private:
        static void* opencvStreamFunc(void* arg);
        void runStreamThread();

    private:
        std::string m_strUrl;
        cv::VideoCapture m_videoCapture;

        pthread_t m_pthread;
        pthread_mutex_t m_mtLock;

        bool m_bEnable = false;
        bool m_bRun = false;

        BlockingQueue<cv::Mat> m_quFrame;
        int m_iMaxQueueSize = 5;
};

#endif // OpencvStream_hpp