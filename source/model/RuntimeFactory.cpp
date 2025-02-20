/**
 * @file RuntimeFactory.cpp
 * @author HuyNQ
 * @brief 
 * @version 0.1
 * @date 2024-06-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "RuntimeFactory.hpp"
#include "JsonTypes.hpp"

#ifdef TENSORRT_RUNTIME
#include "TRTScrfd.hpp"
#include "TRTAdaface.hpp"
#include "TRTFaceQualityNet.hpp"
#include "TRTPplcNet.hpp"
#include "TRTPPOcrDet.hpp"
#include "TRTYoloV5.hpp"
#include "TRTYoloV8.hpp"
#include "TRTPPOcrRec.hpp"
#include "TRTFAS.hpp"
#endif

#ifdef SNPE_RUNTIME
#include "SNPEAdaface.hpp"
#include "SNPEScrfd.hpp"
#include "SNPEFaceLightQNet.hpp"
#include "SNPEFAS.hpp"
#include "SNPEYoloNAS.hpp"
#include "SNPEYoloV11.hpp"
#include "SNPEYoloV8.hpp"
#include "model/snpe/SNPEYolov11_cus.hpp"
#endif


#ifdef NCNN_RUNTIME
#include "NCNNYoloV4.hpp"
#endif

// #include "ORTYoloV8.hpp"
// #include "ORTYoloV11.hpp"
// #include "ORTYoloNAS.hpp"

RuntimeFactory::RuntimeFactory(){

}

RuntimeFactory::~RuntimeFactory()
{

}

AbstractRuntime* RuntimeFactory::createRuntime(std::string& strModel, std::string& strFrameWork, nlohmann::json &jConfigModel){
    if (strFrameWork == JN_FRAMEWORK_TRT){
#ifdef TENSORRT_RUNTIME
        if (strModel == JN_MODEL_SCRFD)
            return new TRTScrfd(jConfigModel);
        else if (strModel == JN_MODEL_ADAFACE)
            return new TRTAdaface(jConfigModel);
        else if (strModel == JN_MODEL_FACEQNET)
            return new TRTFaceQualityNet(jConfigModel);
        else if (strModel == JN_MODEL_PPLCNET)
            return new TRTPplcNet(jConfigModel);
        else if (strModel == JN_MODEL_PPOCR_DET)
            return new TRTPPOcrDet(jConfigModel);
        else if (strModel == JN_MODEL_PPOCR_REC)
            return new TRTPPOcrRec(jConfigModel);
        else if (strModel == JN_MODEL_YOLOV5)
            return new TRTYoloV5(jConfigModel);
        else if (strModel == JN_MODEL_YOLOV8)
            return new TRTYoloV8(jConfigModel);
        else if (strModel == JN_MODEL_FAS)
            return new TRTFas(jConfigModel);
#endif
        return nullptr;
    }
    else if (strFrameWork == JN_FRAMEWORK_ONNX)
    {
        // if (strModel == JN_MODEL_YOLOV8)
        // {
        //     return new ORTYoloV8(jConfigModel);
        // }
        // else if (strModel == JN_MODEL_YOLOV11)
        // {
        //     return new ORTYoloV11(jConfigModel);
        // }
        // else if (strModel == JN_MODEL_YOLONAS)
        // {
        //     return new ORTYoloNAS(jConfigModel);
        // }
        return nullptr;
    }
    else if (strFrameWork == JN_FRAMEWORK_SNPE){
#ifdef SNPE_RUNTIME
        if (strModel == JN_MODEL_ADAFACE)
            return new SNPEAdaface(jConfigModel);
        if (strModel == JN_MODEL_SCRFD)
            return new SNPEScrfd(jConfigModel);
        if (strModel == JN_MODEL_FACEQNET)
            return new SNPEFaceLightQNet(jConfigModel);
        if (strModel == JN_MODEL_FAS)
            return new SNPEFAS(jConfigModel);
        if (strModel == JN_MODEL_YOLONAS)
            return new SNPEYoloNAS(jConfigModel);
        if (strModel == JN_MODEL_YOLOV8)
            return new SNPEYoloV8(jConfigModel);
        if (strModel == JN_MODEL_YOLOV11)
            return new SNPEYoloV11(jConfigModel);
        if (strModel == JN_MODEL_YOLOV11_CUS)
            return new SNPEYolov11_cus(jConfigModel);
#endif
        return nullptr;
    }
    else if (strFrameWork == JN_FRAMEWORK_NCNN)
    {
#ifdef NCNN_RUNTIME
        if (strModel == JN_MODEL_YOLOV4)
        {
            return new NCNNYoloV4(jConfigModel);
        }
#endif
    }
}