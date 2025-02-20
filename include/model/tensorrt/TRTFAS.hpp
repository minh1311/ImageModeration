/**
 * @file TRTFAS.hpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-07-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef TRTFAS_HPP
#define TRTFAS_HPP

#include <functional>

#include "TRTModel.hpp"
#include "AbstractRuntime.hpp"
#include "common/CerberusFace.h"

class TRTFas : public TRTModel, public AbstractRuntime
{
public:
    TRTFas(nlohmann::json& jModelConfig);
    ~TRTFas();

    void run(const cv::Mat &image, std::vector<float>& objects, cerberus::CerberusFace & face) override;

protected:
    // std::vector<float> softmax(const std::vector<float>& input);
    void preprocess(cv::Mat& mImage, cerberus::CerberusFace & face);
    void preprocess(cv::Mat& mImage) override{};
    // void postprocess(std::vector<void*>& buffer) override{};
    void postprocess(std::vector<void*>& buffer, std::vector<float>& objects);
    // cv::Rect calculateFASBox(cerberus::CerberusFace &box, int w, int h);

private:
    std::string m_strInputName = "input";
    std::string m_strOutput = "output";

    // std::string m_strOutputDiv704 = "onnx::Div_704";

//    std::array<float, 3> m_subVals = {127.5f, 127.5f, 127.5f};
//    std::array<float, 3> m_divVals = {128.f, 128.f, 128.f};
    std::array<float, 3> m_subVals = {0, 0, 0};
    std::array<float, 3> m_divVals = {1, 1, 1};
    int m_iWidthModel, m_iHeightModel;
    int m_iInputWidth, m_iInputHeight;
    float m_fScale;
};
#endif //TRTFAS_HPP
