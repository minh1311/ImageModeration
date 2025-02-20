/**
 * @file TRTScrfd.hpp
 * @author HuyNQ
 * @brief 
 * @version 0.1
 * @date 2024-06-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef TRTSCRFD_HPP
#define TRTSCRFD_HPP

#include <functional>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "TRTModel.hpp"
#include "AbstractRuntime.hpp"
#include "Types.hpp"
#include "common/FaceObject.h"

class TRTScrfd :  public TRTModel, public AbstractRuntime
{
    public:
        TRTScrfd(nlohmann::json& jModelConfig);
        ~TRTScrfd();

        void run(const cv::Mat &mImage, std::vector<cerberus::FaceObject> &vtFaceObject) override;

    protected:
        void preprocess(cv::Mat& mImage) override;
        // void postprocess(std::vector<void*>& buffer) override{};

    private:
        void postprocess(std::vector<void*>& buffers, std::vector<cerberus::FaceObject>& stObjects);

    private:
        std::string m_strInputName = "input.1";

        std::string m_strOutputScore8 = "446";
        std::string m_strOutputBbox8 = "449";
        std::string m_strOutputKps8 = "452";

        std::string m_strOutputScore16 = "466";
        std::string m_strOutputBbox16 = "469";
        std::string m_strOutputKps16 = "472";

        std::string m_strOutputScore32 = "486";
        std::string m_strOutputBbox32 = "489";
        std::string m_strOutputKps32 = "492";

        std::array<float, 3> m_subVals = {127.5f, 127.5f, 127.5f};
        std::array<float, 3> m_divVals = {128.f, 128.f, 128.f};

        int m_iWidthModel, m_iHeightModel;

        int m_iInputWidth, m_iInputHeight;
        float m_fRatioWidth, m_fRatioHeight;

        float m_fScoreThreshold = 0.5f;
        float m_fNMSThreshold = 0.4f;

        std::vector<int> m_featStrideFpn = {8, 16, 32};
        std::vector<std::vector<std::vector<float>>> m_anchors;

        std::function<void (const std::vector<cerberus::FaceObject>&)> m_fnCallback;
};

#endif // TRTSCRFD_HPP