#ifndef ORTYoloV11_hpp
#define ORTYoloV11_hpp

// #include <functional>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "ORTModel.hpp"
#include "AbstractRuntime.hpp"
#include "common/Types.hpp"

class ORTYoloV11 : public ORTModel, public AbstractRuntime
{
    public:
        ORTYoloV11(nlohmann::json& jModelConfig);
        ~ORTYoloV11();

        void run(cv::Mat& mImage, std::vector<stObject_t>& stObjects) override;

    protected:
        void preprocess(cv::Mat& mImage) override;
        // void postprocess() override;

    private:
        void postprocess(std::vector<stObject_t>& stObjects);

    private:
        std::string m_strInputName = "images";
        std::string m_strOutputName = "output0";

        int m_iNumClasses = 3;
        int m_iWidthModel, m_iHeightModel;
        int m_iOutputWidth, m_iOutputHeight;

        int m_iInputWidth, m_iInputHeight;
        float m_fRatioWidth, m_fRatioHeight;

        // float m_fScoreThreshold = 0.5f;
        // float m_fNMSThreshold = 0.4f;

        // std::function<void (const std::vector<stObject_t>&)> m_fnCallback;
};

#endif // ORTYoloV11_hpp