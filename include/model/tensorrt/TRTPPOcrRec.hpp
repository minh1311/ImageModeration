//
// Created by hoanglm on 09/04/2024.
//

#ifndef TRT_TRTPPOCRREC_HPP
#define TRT_TRTPPOCRREC_HPP

#include "TRTModel.hpp"
#include "AbstractRuntime.hpp"

class TRTPPOcrRec : public TRTModel, public AbstractRuntime
{
    public:
        TRTPPOcrRec(nlohmann::json& jModelConfig);
        ~TRTPPOcrRec();

        void setLabels(std::string& strPath);

        void run(cv::Mat& mImage, stTextRec_t& stTextRec) override;

    protected:
        void preprocess(cv::Mat& mImage) override;

    private:
        void postprocess(std::vector<void*>& buffers, stTextRec_t& textObject);

    private:
        std::vector<std::string> readDict(const std::string &path);

    private:
        nlohmann::json runtimeConfig;
        nlohmann::json initConfig;
        std::vector<std::string> label_list_;

        int m_iWidthModel, m_iHeightModel;
        int m_iOutputWidth, m_iOutputHeight;

        int m_iInputWidth, m_iInputHeight;
        float m_fRatioWidth, m_fRatioHeight;
        int miBatch;

        std::string m_strInputName = "x";
        std::string m_strOutputName = "softmax_2.tmp_0";
};
#endif //TRT_TRTPPOCRREC_HPP
