#ifndef TRTModel_hpp
#define TRTModel_hpp

#include <string>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "TRTCommon.hpp"
#include "TRTRunner.hpp"
#include "Types.hpp"
#include "json.hpp"
#include "JsonTypes.hpp"
#define TagTRTModel "[TRTModel]"

/**
 * @brief
 * 
*/
class TRTModel
{
    public:
        TRTModel();
        /**
         * @brief
        */
        TRTModel(nlohmann::json& jModelConfig, const std::string& strModelName);

        /**
         * @brief
        */
        ~TRTModel();

    protected:
        virtual void preprocess(cv::Mat& mImage)=0 ;
        // virtual void postprocess(std::vector<void*>& buffer)= 0;
        virtual void setNMSThreshold(float fNMSThreshold);
        virtual void setScoreThreshold(float fScoreThreshold);
        void L2Normalize(std::vector<float>& feature);
    protected:
        std::shared_ptr<TRTRunner> shpTRTRunner;

        std::unordered_map<std::string, std::vector<size_t>> umpIOTensors;
        std::unordered_map<std::string, nvinfer1::Dims> umpIOTensorsShape;

        std::vector<void*> buffers;
        float m_fScoreThreshold;
        float m_fNMSThreshold;
        std::string m_strModelName;
        
};

#endif // TRTModel_hpp