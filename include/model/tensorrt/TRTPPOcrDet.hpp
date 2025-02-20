//
// Created by hoanglm on 29/03/2024.
//

#ifndef TRT_TRTPPOCRDET_HPP
#define TRT_TRTPPOCRDET_HPP

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

class TRTPPOcrDet : public TRTModel, public AbstractRuntime
{
    public:
        TRTPPOcrDet(nlohmann::json& jModelConfig);
        ~TRTPPOcrDet();

        void run(cv::Mat& mImage, stTextBox_t& stObjects) override;

        // static cv::Mat getRotateCropImage(const cv::Mat &srcimage, std::vector<std::vector<int>> box);

    protected:
        void preprocess(cv::Mat& mImage) override;
        // void postprocess(std::vector<void*>& buffers) override;

    private:
        void postprocess(std::vector<void*>& buffers, stTextBox_t& stObjects);
        void preprocess(cv::Mat& mImage, stTextBox_t& stObjects) ;


    private:
        int input_width_, input_height_;
        void getContourArea(const std::vector<std::vector<float>> &box, float unclip_ratio, float &distance);
        cv::RotatedRect unClip(std::vector<std::vector<float>> box, const float &unclip_ratio);
        float **Mat2Vec(cv::Mat mat);
        std::vector<std::vector<int>> orderPointsClockwise(std::vector<std::vector<int>> pts);
        std::vector<std::vector<float>> getMiniBoxes(cv::RotatedRect box, float &ssid);
        float boxScoreFast(std::vector<std::vector<float>> box_array, cv::Mat pred);
        float polygonScoreAcc(std::vector<cv::Point> contour, cv::Mat pred);

        std::vector<std::vector<std::vector<int>>>
        boxesFromBitmap(const cv::Mat pred, const cv::Mat bitmap,
                        const float &box_thresh, const float &det_db_unclip_ratio,
                        const bool &use_polygon_score);

        std::vector<std::vector<std::vector<int>>>
        filterTagDetRes(std::vector<std::vector<std::vector<int>>> boxes,
                        float ratio_h, float ratio_w, cv::Mat srcimg);

    private:
        static bool xSortInt(std::vector<int> a, std::vector<int> b);
        static bool xSortFp32(std::vector<float> a, std::vector<float> b);

        std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);

        inline int _max(int a, int b) { return a >= b ? a : b; }

        inline int _min(int a, int b) { return a >= b ? b : a; }

        template <class T> inline T clamp(T x, T min, T max) {
            if (x > max)
                return max;
            if (x < min)
                return min;
            return x;
        }

        inline float clampf(float x, float min, float max) {
            if (x > max)
                return max;
            if (x < min)
                return min;
            return x;
        }

    private:
        int m_iWidthModel, m_iHeightModel;
        int m_iOutputWidth, m_iOutputHeight;

        int m_iInputWidth, m_iInputHeight;
        float m_fRatioWidth, m_fRatioHeight;

        double detDBThreshold = 0.1f;
        double detDBBoxThreshold = 0.5f;
        double detDBUnClipRatio = 1.5f;
        float m_ratio = 1;
        bool usePolygonScore = false;

        std::string m_strInputName = "x";
        std::string m_strOutputName = "sigmoid_0.tmp_0";

        std::array<float, 3> m_subVals = {0.485f, 0.456f, 0.406f};
        std::array<float, 3> m_divVals = {0.229f, 0.224f, 0.225f};

        std::function<void (const std::vector<stTextBox_t>&)> m_fnCallback;
};
#endif //TRT_TRTPPOCRDET_HPP
