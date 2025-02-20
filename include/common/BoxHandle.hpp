#ifndef BOX_HANDLE_HPP
#define BOX_HANDLE_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include "clipper.hpp"

namespace BoxHandle{
    // inline bool cvPointCompare(const cv::Point& a, const cv::Point& b) {
    //     return a.x < b.x;
    // }

    inline std::vector<cv::Point> getMinBoxes(const std::vector<cv::Point>& inVec, float& minSideLen, float& allEdgeSize) {
        std::vector<cv::Point> minBoxVec;
        cv::RotatedRect textRect = cv::minAreaRect(inVec);
        cv::Mat boxPoints2f;
        cv::boxPoints(textRect, boxPoints2f);

        auto* p1 = (float*)boxPoints2f.data;
        std::vector<cv::Point> tmpVec;
        for (int i = 0; i < 4; ++i, p1 += 2) {
            tmpVec.emplace_back(int(p1[0]), int(p1[1]));
        }

        std::sort(tmpVec.begin(), tmpVec.end(), [](const cv::Point& a, const cv::Point& b){
            return a.x < b.x;
        });

        minBoxVec.clear();

        int index1, index2, index3, index4;
        if (tmpVec[1].y > tmpVec[0].y) {
            index1 = 0;
            index4 = 1;
        }
        else {
            index1 = 1;
            index4 = 0;
        }

        if (tmpVec[3].y > tmpVec[2].y) {
            index2 = 2;
            index3 = 3;
        }
        else {
            index2 = 3;
            index3 = 2;
        }

        minBoxVec.clear();

        minBoxVec.push_back(tmpVec[index1]);
        minBoxVec.push_back(tmpVec[index2]);
        minBoxVec.push_back(tmpVec[index3]);
        minBoxVec.push_back(tmpVec[index4]);

        minSideLen = (std::min)(textRect.size.width, textRect.size.height);
        allEdgeSize = 2.f * (textRect.size.width + textRect.size.height);

        return minBoxVec;
    }

    inline float boxScoreFast(const cv::Mat & inMat, const std::vector<cv::Point> & inBox) {
        std::vector<cv::Point> box = inBox;
        int width = inMat.cols;
        int height = inMat.rows;
        int maxX = -1, minX = 1000000, maxY = -1, minY = 1000000;
        for (auto & i : box) {
            if (maxX < i.x)
                maxX = i.x;
            if (minX > i.x)
                minX = i.x;
            if (maxY < i.y)
                maxY = i.y;
            if (minY > i.y)
                minY = i.y;
        }
        maxX = (std::min)((std::max)(maxX, 0), width - 1);
        minX = (std::max)((std::min)(minX, width - 1), 0);
        maxY = (std::min)((std::max)(maxY, 0), height - 1);
        minY = (std::max)((std::min)(minY, height - 1), 0);

        for (auto & i : box) {
            i.x = i.x - minX;
            i.y = i.y - minY;
        }

        std::vector<std::vector<cv::Point>> maskBox;
        maskBox.push_back(box);
        cv::Mat maskMat(maxY - minY + 1, maxX - minX + 1, CV_8UC1, cv::Scalar(0, 0, 0));
        cv::fillPoly(maskMat, maskBox, cv::Scalar(1, 1, 1), 1);
        return cv::mean(inMat(cv::Rect(cv::Point(minX, minY), cv::Point(maxX + 1, maxY + 1))).clone(),
            maskMat).val[0];
    }

    inline std::vector<cv::Point> unClip(const std::vector<cv::Point> & inBox, float perimeter, float unClipRatio) {
        std::vector<cv::Point> outBox;
        ClipperLib::Path poly;

        for (const auto & i : inBox) {
            poly.emplace_back(i.x, i.y);
        }

        double distance = unClipRatio * ClipperLib::Area(poly) / (double)perimeter;

        ClipperLib::ClipperOffset clipperOffset;
        clipperOffset.AddPath(poly, ClipperLib::JoinType::jtRound, ClipperLib::EndType::etClosedPolygon);
        ClipperLib::Paths polys;
        polys.push_back(poly);
        clipperOffset.Execute(polys, distance);

        outBox.clear();
        std::vector<cv::Point> rsVec;
        for (const auto& tmpPoly : polys) {
            for (auto & j : tmpPoly) {
                outBox.emplace_back(j.X, j.Y);
            }
        }
        return outBox;
    }

    inline cv::Mat getRotateCropImage(const cv::Mat& img, const std::vector<std::vector<int>>& points) {
        // Convert points to float32
        std::vector<cv::Point2f> srcPoints;
        for (const auto& point : points) {
            srcPoints.push_back(cv::Point2f(static_cast<float>(point[0]), static_cast<float>(point[1])));
        }

        // Calculate width and height
        float width1 = cv::norm(srcPoints[0] - srcPoints[1]);
        float width2 = cv::norm(srcPoints[2] - srcPoints[3]);
        float height1 = cv::norm(srcPoints[0] - srcPoints[3]);
        float height2 = cv::norm(srcPoints[1] - srcPoints[2]);

        int imgCropWidth = static_cast<int>(std::max(width1, width2));
        int imgCropHeight = static_cast<int>(std::max(height1, height2));

        // Define destination points
        std::vector<cv::Point2f> dstPoints = {
            cv::Point2f(0, 0),
            cv::Point2f(static_cast<float>(imgCropWidth), 0),
            cv::Point2f(static_cast<float>(imgCropWidth), static_cast<float>(imgCropHeight)),
            cv::Point2f(0, static_cast<float>(imgCropHeight))
        };

        // Get perspective transform and warp image
        cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);
        cv::Mat dstImg;
        
        cv::warpPerspective(img, dstImg, M, cv::Size(imgCropWidth, imgCropHeight),
                        cv::INTER_CUBIC, cv::BORDER_REPLICATE);

        // Rotate image if height/width ratio >= 1.5
        if (static_cast<float>(dstImg.rows) / static_cast<float>(dstImg.cols) >= 1.5) {
            cv::Mat rotated;
            cv::rotate(dstImg, rotated, cv::ROTATE_90_CLOCKWISE);
            return rotated;
        }

        return dstImg;
    }
    inline std::vector<std::vector<std::vector<int>>> findRsBoxes(const cv::Mat& fMapMat, const cv::Mat& norfMapMat,
                                 const float boxScoreThresh, const float unClipRatio)
    {
        float minArea = 3;
        std::vector<std::vector<std::vector<int>>> textBoxes;
        textBoxes.clear();
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < contours.size(); ++i)
        {
            float minSideLen, perimeter;
            std::vector<cv::Point> minBox = BoxHandle::getMinBoxes(contours[i], minSideLen, perimeter);
            if (minSideLen < minArea)
                continue;
            float score = BoxHandle::boxScoreFast(fMapMat, contours[i]);
            if (score < boxScoreThresh)
                continue;
            //---use clipper start---
            std::vector<cv::Point> clipBox = BoxHandle::unClip(minBox, perimeter, unClipRatio);
            std::vector<cv::Point> clipMinBox = BoxHandle::getMinBoxes(clipBox, minSideLen, perimeter);
            //---use clipper end---

            if (minSideLen < minArea + 2)
                continue;
            std::vector<std::vector<int>> points;
            for (auto & j : clipMinBox)
            {
                std::vector<int> point;
                j.x = (j.x / 1.0);
                j.x = (std::min)((std::max)(j.x, 0), norfMapMat.cols);

                j.y = (j.y / 1.0);
                j.y = (std::min)((std::max)(j.y, 0), norfMapMat.rows);
                point.emplace_back((int)(j.x));
                point.emplace_back((int)(j.y));

                points.emplace_back(point);
                point.clear();
                point.shrink_to_fit();

            }
            textBoxes.emplace_back(points);
            points.clear();
            points.shrink_to_fit();

        }
        // reverse(rsBoxes.begin(), rsBoxes.end());

        return textBoxes;
    }

}



// inline cv::Mat getRotateCropImage(const cv::Mat& src, std::vector<cv::Point> box) {
//     cv::Mat image;
//     src.copyTo(image);
//     std::vector<cv::Point> points = box;

//     int collectX[4] = { box[0].x, box[1].x, box[2].x, box[3].x };
//     int collectY[4] = { box[0].y, box[1].y, box[2].y, box[3].y };
//     int left = int(*std::min_element(collectX, collectX + 4));
//     int right = int(*std::max_element(collectX, collectX + 4));
//     int top = int(*std::min_element(collectY, collectY + 4));
//     int bottom = int(*std::max_element(collectY, collectY + 4));

//     cv::Mat imgCrop;
//     image(cv::Rect(left, top, right - left, bottom - top)).copyTo(imgCrop);

//     for (auto & point : points) {
//         point.x -= left;
//         point.y -= top;
//     }


//     auto imgCropWidth = int(sqrt(pow(points[0].x - points[1].x, 2) +
//         pow(points[0].y - points[1].y, 2)));
//     auto imgCropHeight = int(sqrt(pow(points[0].x - points[3].x, 2) +
//         pow(points[0].y - points[3].y, 2)));

//     cv::Point2f ptsDst[4];
//     ptsDst[0] = cv::Point2f(0., 0.);
//     ptsDst[1] = cv::Point2f(imgCropWidth, 0.);
//     ptsDst[2] = cv::Point2f(imgCropWidth, imgCropHeight);
//     ptsDst[3] = cv::Point2f(0.f, imgCropHeight);

//     cv::Point2f ptsSrc[4];
//     ptsSrc[0] = cv::Point2f(points[0].x, points[0].y);
//     ptsSrc[1] = cv::Point2f(points[1].x, points[1].y);
//     ptsSrc[2] = cv::Point2f(points[2].x, points[2].y);
//     ptsSrc[3] = cv::Point2f(points[3].x, points[3].y);

//     cv::Mat M = cv::getPerspectiveTransform(ptsSrc, ptsDst);

//     cv::Mat partImg;
//     // check empty imgCrop
//     if (imgCrop.empty()) {
//         return partImg;
//     }
//     cv::warpPerspective(imgCrop, partImg, M,
//         cv::Size(imgCropWidth, imgCropHeight),
//         cv::BORDER_REPLICATE);

//     if (float(partImg.rows) >= float(partImg.cols) * 1.5) {
//         cv::Mat srcCopy = cv::Mat(partImg.rows, partImg.cols, partImg.depth());
//         cv::transpose(partImg, srcCopy);
//         cv::flip(srcCopy, srcCopy, 0);
//         return srcCopy;
//     }
//     else {
//         return partImg;
//     }
// }

#endif //BOX_HANDLE_HPP