//
// Created by hoanglm on 15/05/2024.
//

#ifndef TRTADAFACE_HPP
#define TRTADAFACE_HPP

#include <functional>

#include "TRTModel.hpp"
#include "AbstractRuntime.hpp"


class TRTAdaface : public TRTModel, public AbstractRuntime
{
public:
    TRTAdaface(nlohmann::json& jModelConfig);
    ~TRTAdaface();

    void run(const cv::Mat &image, std::vector<float>& objects) override;

protected:
    void preprocess(cv::Mat& mImage) override;
    // void postprocess(std::vector<void*>& buffer) override{};
    void postprocess(std::vector<void*>& buffer, std::vector<float>& objects);

private:
    std::string m_strInputName = "onnx::Slice_0";
    std::string m_strOutput = "1358";

    std::string m_strOutputDiv704 = "onnx::Div_704";

//    std::array<float, 3> m_subVals = {127.5f, 127.5f, 127.5f};
//    std::array<float, 3> m_divVals = {128.f, 128.f, 128.f};
    std::array<float, 3> m_subVals = {0.5f, 0.5f, 0.5f};
    std::array<float, 3> m_divVals = {0.5f, 0.5f, 0.5f};
    int m_iWidthModel, m_iHeightModel;

    int m_iInputWidth, m_iInputHeight;
};
#endif //TRTADAFACE_HPP
