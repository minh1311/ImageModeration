#include "stream/FFmpegStream.hpp"

#include "Logger.hpp"

FFmpegStream::FFmpegStream(std::string strUrl)
{
    m_strUrl = strUrl;
    m_strStreamType = strUrl.substr(0, 4);

    m_avFormatCtx = nullptr;
    m_avCodecCtx = nullptr;
    // m_avStream = nullptr;
    m_avCodec = nullptr;
    m_iStreamIdx = -1;
    m_swsCtx = nullptr;
    m_avDict = nullptr;

    // avcodec_register_all();
    // av_register_all();
    avformat_network_init();
    av_log_set_level(AV_LOG_QUIET);
    // av_dict_set_int(&m_avDict, "rw_timeout", 5000000, 0);
    // av_dict_set(&m_avDict, "rtsp_transport", "tcp", 0);
    // av_dict_set_int(&m_avDict, "tcp_nodelay", 1, 0);
    // av_dict_set(&m_avDict, "stimeout", "5000000", 0);
    // av_dict_set(&m_avDict, "rtsp_flags", "prefer_tcp", 0);
    // av_dict_set(&m_avDict, "allowed_media_types", "video", 0);
    // av_dict_set(&m_avDict, "max_delay", "5000000", 0);

    m_avPixelFormat = AV_PIX_FMT_BGR24;

    if (pthread_mutex_init(&m_mtLock, NULL) != 0)
    {
        // std::cerr << "mutex init failed!" << std::endl;
        ERROR("mutex init failed!");
        return ;
    }

    pthread_create(&m_pthread, NULL, ffmpegStreamFunc, this);
    pthread_detach(m_pthread);

    m_bEnable = true;
    m_bRun = true;
    INFO("Stream created successfully");
}

FFmpegStream::~FFmpegStream()
{
    pthread_cancel(m_pthread);

    while (m_quFrame.getSize() > 0)
    {
        cv::Mat mFrame;
        m_quFrame.tryWaitAndPop(mFrame, 50);
    }
}

std::string FFmpegStream::getUrl()
{
    return m_strUrl;
}

int FFmpegStream::getWidth()
{
    return m_iFrameWidth;
}

int FFmpegStream::getHeight()
{
    return m_iFrameHeight;
}

void FFmpegStream::getFrame(cv::Mat& mFrame)
{
    m_quFrame.tryWaitAndPop(mFrame, 50);
}

void FFmpegStream::runStream()
{
    pthread_mutex_lock(&m_mtLock);
    m_bRun = true;
    pthread_mutex_unlock(&m_mtLock);
}

void FFmpegStream::pauseStream()
{
    pthread_mutex_lock(&m_mtLock);
    m_bRun = false;
    pthread_mutex_unlock(&m_mtLock);
}

void FFmpegStream::stopStream()
{
    pthread_mutex_lock(&m_mtLock);
    m_bEnable = false;
    pthread_mutex_unlock(&m_mtLock);
}

void FFmpegStream::join()
{
    pthread_join(m_pthread, NULL);
}

void* FFmpegStream::ffmpegStreamFunc(void* arg)
{
    FFmpegStream* pStream = (FFmpegStream*)arg;
    pStream->runStreamThread();
    pthread_exit(NULL);
}

void FFmpegStream::runStreamThread()
{
    while (true)
    {
        pthread_mutex_lock(&m_mtLock);
        if (m_bEnable == false)
            break;

        if (m_bRun == false)
            continue;
        pthread_mutex_unlock(&m_mtLock);
    
        // int iVideoStreamIdx;
        {
            if (m_strStreamType == "rtsp")
            {
                av_dict_set_int(&m_avDict, "rw_timeout", 5000000, 0);
                av_dict_set(&m_avDict, "rtsp_transport", "tcp", 0);
                av_dict_set_int(&m_avDict, "tcp_nodelay", 1, 0);
                av_dict_set(&m_avDict, "stimeout", "5000000", 0);
                av_dict_set(&m_avDict, "rtsp_flags", "prefer_tcp", 0);
                av_dict_set(&m_avDict, "allowed_media_types", "video", 0);
                av_dict_set(&m_avDict, "max_delay", "5000000", 0);
            }

            m_avFormatCtx = avformat_alloc_context();

            m_avFormatCtx->flags |= AVFMT_FLAG_NONBLOCK;
            m_avFormatCtx->flags |= AVFMT_FLAG_DISCARD_CORRUPT;
            m_avFormatCtx->flags |= AVFMT_FLAG_NOBUFFER;

            int iRet = avformat_open_input(&m_avFormatCtx, m_strUrl.c_str(), nullptr, &m_avDict);
            if (iRet < 0)
            {
                // std::cerr << "Fail to open input:" << m_strUrl.c_str() << std::endl;
                // ERROR("Fail to open input: {}", m_strUrl);
                goto reconnect;
            }

            // av_dict_set(&m_avDict, "analyzeduration", "15000000 ", 0);
            // av_dict_set(&m_avDict, "probesize", "15000000 ", 0);

            iRet = avformat_find_stream_info(m_avFormatCtx, nullptr);
            // iRet = avformat_find_stream_info(m_avFormatCtx, &m_avDict);
            if (iRet < 0)
            {
                // std::cerr << "fail to avformat_find_stream_info: ret=" << iRet << std::endl;
                ERROR("fail to avformat_find_stream_info: ret={}", iRet);
                goto reconnect;
            }

            for (int i=0; i < m_avFormatCtx->nb_streams; i++)
            {
                // if (m_avFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO)
                if (m_avFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
                    m_iStreamIdx = i;
            }
            // av_read_play(m_avFormatCtx);

            auto codecID = m_avFormatCtx->streams[m_iStreamIdx]->codecpar->codec_id;
            m_avCodec = avcodec_find_decoder(codecID);

            m_avCodecCtx = avcodec_alloc_context3(m_avCodec);
            if (m_avCodecCtx == NULL)
                // std::cerr << "fail to avcodec_alloc_context3" << std::endl;
                ERROR("fail to avcodec_alloc_context3");

            // avcodec_get_context_defaults3(m_avCodecCtx, m_avCodec);
            // iRet = avcodec_copy_context(m_avCodecCtx, m_avFormatCtx->streams[m_iStreamIdx]->codec);
            // if (iRet < 0)
            //     std::cerr << "fail to avcodec_copy_context" << std::endl;

            avcodec_parameters_to_context(m_avCodecCtx, m_avFormatCtx->streams[m_iStreamIdx]->codecpar);

            // iRet = avcodec_open2(m_avCodecCtx, m_avCodec, nullptr);
            iRet = avcodec_open2(m_avCodecCtx, m_avCodec, &m_avDict);
            if (iRet < 0)
            {
                // std::cerr << "fail to avcodec_open2: ret=" << iRet << std::endl;
                ERROR("fail to avcodec_open2: ret={}", iRet);
                goto reconnect;
            }

            m_iFrameWidth = m_avCodecCtx->width;
            m_iFrameHeight = m_avCodecCtx->height;
            if (m_iFrameWidth <= 0 || m_iFrameHeight <= 0){
                goto reconnect;
            }

            DEBUG("===============================");
            DEBUG("url:    {}", m_strUrl);
            DEBUG("format: {}", m_avFormatCtx->iformat->name);
            DEBUG("vcodec: {}", m_avCodec->name);
            DEBUG("size:   {} x {}", m_iFrameWidth, m_iFrameHeight);
            DEBUG("fps:    {}", av_q2d(m_avCodecCtx->framerate));
            DEBUG("pixfmt: {}", av_get_pix_fmt_name(m_avCodecCtx->pix_fmt));
            DEBUG("===============================");

            m_swsCtx = sws_getContext(m_avCodecCtx->width, m_avCodecCtx->height, m_avCodecCtx->pix_fmt,
                                        m_iFrameWidth, m_iFrameHeight, m_avPixelFormat, SWS_BILINEAR,
                                        nullptr, nullptr, nullptr);

            AVFrame* avFrame = av_frame_alloc();
            AVFrame* avDecFrame = av_frame_alloc();

            // int iSizeFrameOut = avpicture_get_size(m_avPixelFormat, m_iFrameWidth, m_iFrameHeight);
            // auto* bufferOut = (uint8_t*)av_malloc(iSizeFrameOut * sizeof(uint8_t));
            // avpicture_fill((AVPicture*)avFrame, bufferOut, m_avPixelFormat, m_iFrameWidth, m_iFrameHeight);

            bool bEndStream = false;
            AVPacket* avPkt = av_packet_alloc();
            av_init_packet(avPkt);
            // int iFrameFailCount = 0;
            do {
                usleep(10000);
                if (bEndStream == false)
                {
                    iRet = av_read_frame(m_avFormatCtx, avPkt);
                    if ((iRet < 0) && (iRet != AVERROR_EOF))
                    {
                        // std::cerr << "fail to av_read_frame: ret= " << iRet << " - " << m_strUrl << std::endl;
                        // WARN("fail to av_read_frame: ret={}", iRet);
                        // iFrameFailCount ++;
                        goto next_packet;
                    }
                    if ((iRet == 0) && (avPkt->stream_index != m_iStreamIdx))
                    {
                        // std::cerr << "fail stream index - " << m_strUrl << std::endl;
                        // WARN("fail stream index - {}", m_strUrl);
                        // iFrameFailCount ++;
                        goto next_packet;
                    }
                    bEndStream = (iRet == AVERROR_EOF);
                }
                else
                {
                    // std::cerr << "==>> EOF: ret = " << iRet  << " - " << m_strUrl << std::endl;
                    WARN("==>> EOF: ret = {} - {}",iRet, m_strUrl);
                    goto next_packet;
                }

                if (avcodec_send_packet(m_avCodecCtx, avPkt) >= 0)
                {
                    iRet = avcodec_receive_frame(m_avCodecCtx, avDecFrame);
                    if (iRet < 0)
                    {
                        // std::cerr << "receive packet fail!" << iRet << std::endl;
                        // WARN("receive packet fail - {}", iRet);
                        goto next_packet;
                    }
                }
                else
                {
                    // std::cerr << "send packet fail!" << std::endl;
                    // WARN("send packet fail!");
                    goto next_packet;
                }

                {
                    cv::Mat mImage(m_avCodecCtx->height, m_avCodecCtx->width, CV_8UC3);
                    int cvLinesizes[1];
                    cvLinesizes[0] = mImage.step1();
                    sws_scale(m_swsCtx, avDecFrame->data, avDecFrame->linesize, 0, avDecFrame->height, &mImage.data, cvLinesizes);

                    // sws_scale(m_swsCtx, avDecFrame->data, avDecFrame->linesize, 0, avDecFrame->height, avFrame->data, avFrame->linesize);
                    // cv::Mat mImage(m_avCodecCtx->height, m_avCodecCtx->width, CV_8UC3, bufferOut, avFrame->linesize[0]);

                    if (mImage.empty() == false)
                    {
                        if (m_quFrame.getSize() < m_iMaxQueueSize)
                            m_quFrame.push(mImage);
                        else
                        {
                            cv::Mat mTemp;
                            m_quFrame.tryWaitAndPop(mTemp, 50);
                            m_quFrame.push(mImage);
                        }
                    }
                }
                next_packet:
                // ;
                    // if (iFrameFailCount > 50)
                    // {
                    //     std::cerr << "To many frame read fail. Reconnect - " << m_strUrl << std::endl;
                    //     bEndStream = true;
                    // }
                    av_packet_unref(avPkt);
            } while (bEndStream == false);

            INFO("Stream disconnect - {}", m_strUrl);
            while (m_quFrame.getSize() > 0)
            {
                cv::Mat mTemp;
                m_quFrame.tryWaitAndPop(mTemp, 50);
            }

            av_packet_free(&avPkt);
            sws_freeContext(m_swsCtx);
            // av_free(bufferOut);
            av_frame_free(&avFrame);
            av_frame_free(&avDecFrame);
            avcodec_free_context(&m_avCodecCtx);
        }

        reconnect:
            avcodec_close(m_avCodecCtx);
            avformat_close_input(&m_avFormatCtx);
            av_dict_free(&m_avDict);
            INFO("Reconnect stream after 60 seconds - {}", m_strUrl);
            sleep(60);
    }
}
