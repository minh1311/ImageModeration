#ifndef FFmpegStream_hpp
#define FFmpegStream_hpp

#include <iostream>
#include <string>
#include <pthread.h>
#include <thread>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
#include <libavutil/log.h>
#include <libavutil/opt.h>
#include <libavutil/hwcontext.h>
#include <libavutil/avassert.h>
#include <libavutil/imgutils.h>
}

#include "BlockingQueue.hpp"
#include "stream/Stream.hpp"

class FFmpegStream : public Stream
{
    public:
        FFmpegStream(std::string strUrl);
        ~FFmpegStream();

        std::string getUrl() override;
        int getWidth() override;
        int getHeight() override;
        void getFrame(cv::Mat& mFrame) override; 

        void runStream() override;
        void pauseStream() override;
        void stopStream() override;
        void join() override;

    private:
        static void* ffmpegStreamFunc(void* arg);
        void runStreamThread();

    private:
        std::string m_strUrl;
        std::string m_strStreamType;
        
        int m_iFrameWidth;
        int m_iFrameHeight;

        AVPixelFormat m_avPixelFormat;
        AVFormatContext* m_avFormatCtx;
        AVCodecContext* m_avCodecCtx;
        // AVStream* m_avStream;
        const AVCodec* m_avCodec;
        int m_iStreamIdx;
        SwsContext* m_swsCtx;
        AVDictionary* m_avDict;

        pthread_t m_pthread;
        pthread_mutex_t m_mtLock;

        bool m_bEnable = false;
        bool m_bRun = false;

        BlockingQueue<cv::Mat> m_quFrame;
        int m_iMaxQueueSize = 5;

};

#endif // FFmpegStream_hpp