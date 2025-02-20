//
// Created by hoanglm on 26/12/2024.
//

#ifndef COLOR_H
#define COLOR_H

#include <opencv2/opencv.hpp>
class Color {
public:
    static const cv::Scalar COMMON_OBJECT_BBOX_COLOR;
    static const cv::Scalar EVENT_OBJECT_BBOX_COLOR;
    static const cv::Scalar COUNT_IN_ZONE_COLOR;
    static const cv::Scalar COUNT_OUT_ZONE_COLOR;
    static const cv::Scalar ACTIVATE_LINE_ZONE_COLOR;
    static const cv::Scalar ACTIVATE_OBJECT_COUNT_IN_ZONE_COLOR;
    static const cv::Scalar ACTIVATE_OBJECT_COUNT_OUT_ZONE_COLOR;
};

// Define the static members outside the class (usually in a source file)
inline const cv::Scalar Color::COMMON_OBJECT_BBOX_COLOR = cv::Scalar(102, 204, 0);
inline const cv::Scalar Color::EVENT_OBJECT_BBOX_COLOR = cv::Scalar(76, 153, 0);
inline const cv::Scalar Color::COUNT_IN_ZONE_COLOR = cv::Scalar(51, 153, 255);
inline const cv::Scalar Color::COUNT_OUT_ZONE_COLOR = cv::Scalar(255, 128, 0);
inline const cv::Scalar Color::ACTIVATE_LINE_ZONE_COLOR = cv::Scalar(153, 51, 255);
inline const cv::Scalar Color::ACTIVATE_OBJECT_COUNT_IN_ZONE_COLOR = cv::Scalar(20, 128, 255);
inline const cv::Scalar Color::ACTIVATE_OBJECT_COUNT_OUT_ZONE_COLOR = cv::Scalar(204, 102, 0);

#endif //COLOR_H
