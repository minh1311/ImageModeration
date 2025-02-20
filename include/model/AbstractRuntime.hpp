/**
 * @file AbstractRuntime.hpp
 * @author HuyNQ
 * @brief 
 * @version 0.1
 * @date 2024-06-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef AbstractRuntime_hpp
#define AbstractRuntime_hpp
#include <functional>
#ifdef TENSORRT_RUNTIME
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#endif
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "JsonTypes.hpp"
#include "Types.hpp"
#include "FaceObject.h"
#include "CerberusFace.h"
#include "common/common.hpp"
class AbstractRuntime
{
public:
    // AbstractRuntime();
    virtual ~AbstractRuntime(){};
    virtual void run(const cv::Mat &mImage, std::vector<cerberus::FaceObject>& vtFaceObjects){};
    virtual void run(cv::Mat &image, std::vector<std::vector<float>>& objects){};
    // virtual void run(cv::Mat &image, std::vector<stFaceObject_t>& objects){};
    virtual void run(const cv::Mat &image, std::vector<float>& objects){};
    virtual void run(cv::Mat &image, float& objects){};
    virtual void run(cv::Mat& mImage, stHumanAttribute_t& stHumanAttribute){};
    virtual void run(cv::Mat& mImage, stTextBox_t& stObjects){};
    virtual void run(cv::Mat& mImage, std::vector<stObject_t>& stObjects){};
    virtual void run(cv::Mat& mImage, stTextRec_t& stTextRec){};
    virtual void run(const cv::Mat &image, std::vector<float>& objects, cerberus::CerberusFace & face){};

    virtual std::vector<float> softmax(const std::vector<float>& input);
    virtual cv::Rect calculateFASBox(cerberus::CerberusFace &box, int w, int h, float fScale);
protected:

private:
    // T event;
    /* data */
};



#endif //AbstractRuntime_hpp