#ifndef TRTPplcNet_hpp
#define TRTPplcNet_hpp

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

class TRTPplcNet : public TRTModel, public AbstractRuntime
{
    public:
        TRTPplcNet(nlohmann::json& jModelConfig);
        ~TRTPplcNet();

        void run(cv::Mat& mImage, stHumanAttribute_t& stHumanAttribute) override;

    protected:
        void preprocess(cv::Mat& mImage) override;
        // void postprocess(std::vector<void*>& buffers) override;

    private:
        void postprocess(std::vector<void*>& buffers, stHumanAttribute_t& stHumanAttribute);

    private:
        std::string m_strInputName = "x";
        std::string m_strOutputName = "sigmoid_2.tmp_0";

        std::array<float, 3> m_subVals = {0.485f, 0.456f, 0.406f};
        std::array<float, 3> m_divVals = {0.229f, 0.224f, 0.225f};

        // int m_iNumClasses = 40;
        int m_iWidthModel, m_iHeightModel;
        int m_iOutputWidth, m_iOutputHeight;

        // int m_iInputWidth, m_iInputHeight;

    std::function<void (const stHumanAttribute_t &)> m_fnCallback;
};

#endif // TRTPplcNet_hpp