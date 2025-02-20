/**
 * @file TRTFaceQualityNet.hpp
 * @author HuyNQ
 * @brief 
 * @version 0.1
 * @date 2024-06-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef TRTFACEQUALITYNET_HPP
#define TRTFACEQUALITYNET_HPP
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


class TRTFaceQualityNet : public TRTModel, public AbstractRuntime 
{
public:
    TRTFaceQualityNet(nlohmann::json& jModelConfig);
    ~TRTFaceQualityNet();

    void run(cv::Mat &image, float& conf) override;

protected:
    void preprocess(cv::Mat& mImage) override;
    float postprocess(std::vector<void*>& buffer);

private:
    std::string m_strInputName = "input:0";
    std::string m_strOutput = "confidence_st:0";

    std::array<float, 3> m_subVals = {0.5f, 0.5f, 0.5f};
    std::array<float, 3> m_divVals = {0.5f, 0.5f, 0.5f};
    int m_iWidthModel, m_iHeightModel;

    int m_iInputWidth, m_iInputHeight;
};  
#endif //TRT_FACEQUALITYNET_HPP