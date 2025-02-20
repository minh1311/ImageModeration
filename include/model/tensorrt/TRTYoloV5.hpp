#ifndef TRTYoloV5_hpp
#define TRTYoloV5_hpp

#include <functional>

#include "TRTModel.hpp"
#include "AbstractRuntime.hpp"

class TRTYoloV5 : public TRTModel, public AbstractRuntime
{
    public:
        TRTYoloV5(nlohmann::json& jModelConfig);
        ~TRTYoloV5();

        void run(cv::Mat& mImage, std::vector<stObject_t>& stObjects) override;

    protected:
        void preprocess(cv::Mat& mImage) override;
        // void postprocess(std::vector<void*>& buffers) override;

    private:
        void postprocess(std::vector<void*>& buffers, std::vector<stObject_t>& stObjects);

    private:
        std::string m_strInputName = "images";
        std::string m_strOutputName = "output";

        int m_iNumClasses = 2;
        int m_iWidthModel, m_iHeightModel;
        int m_iOutputWidth, m_iOutputHeight;

        int m_iInputWidth, m_iInputHeight;
        float m_fRatioWidth, m_fRatioHeight;

        float m_fScoreThreshold = 0.5f;
        float m_fNMSThreshold = 0.4f;

        std::function<void (const std::vector<stObject_t>&)> m_fnCallback;
};

#endif // TRTYoloV5_hpp