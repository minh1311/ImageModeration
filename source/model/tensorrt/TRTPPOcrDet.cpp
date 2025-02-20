//
// Created by hoanglm on 01/04/2024.
//
#include "TRTPPOcrDet.hpp"

#include <iostream>

#include "clipper.hpp"

TRTPPOcrDet::TRTPPOcrDet(nlohmann::json& jModelConfig) : TRTModel(jModelConfig, JN_MODEL_PPOCR_DET)
{
    m_iWidthModel = umpIOTensorsShape[m_strInputName].d[3];
    m_iHeightModel = umpIOTensorsShape[m_strInputName].d[2];

    m_iOutputWidth = umpIOTensorsShape[m_strOutputName].d[2];
    m_iOutputHeight = umpIOTensorsShape[m_strOutputName].d[1];
}

TRTPPOcrDet::~TRTPPOcrDet() {
    //
}

void TRTPPOcrDet::postprocess(std::vector<void*> &buffers, stTextBox_t& stObjects)
{
    float output[umpIOTensors[m_strOutputName][1]];
    cudaMemcpy(output, buffers[umpIOTensors[m_strOutputName][0]], umpIOTensors[m_strOutputName][1] * sizeof(float), cudaMemcpyDeviceToHost);

    // int outputSize = sizeof(output) / sizeof(output[0]);
    int outputSize = umpIOTensors[m_strOutputName][1];
    std::vector<float> pred(outputSize, 0.0);
    std::vector<unsigned char> cbuf(outputSize, ' ');
    for (int i = 0; i < outputSize; i++) {
        pred[i] = float(output[i]);
        cbuf[i] = (unsigned char)((output[i]) * 255);
    }
    cv::Mat cbuf_map(m_iWidthModel, m_iHeightModel, CV_8UC1, (unsigned char *)cbuf.data());
    cv::Mat pred_map(m_iWidthModel, m_iHeightModel, CV_32F, (float *)pred.data());

    const double threshold = 0.1f * 255;
    const double maxvalue = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
    cv::Mat dilation_map;
    cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(bit_map, dilation_map, dila_ele);

    std::vector<std::vector<std::vector<int>>> textBoxes = boxesFromBitmap(pred_map, dilation_map, detDBBoxThreshold, detDBUnClipRatio, usePolygonScore);
    textBoxes = TRTPPOcrDet::filterTagDetRes(textBoxes, 1/m_ratio, 1/m_ratio, stObjects.inputCPUBGR);
    stObjects.textBoxes = textBoxes;
}

void TRTPPOcrDet::preprocess(cv::Mat &mImage)
{

}

void TRTPPOcrDet::preprocess(cv::Mat &mImage, stTextBox_t& stObjects)
{
    m_iInputWidth = mImage.cols;
    m_iInputHeight = mImage.rows;

    m_fRatioWidth = 1.0f / (m_iWidthModel / static_cast<float>(m_iInputWidth));
    m_fRatioHeight = 1.0f / (m_iHeightModel / static_cast<float>(m_iInputHeight));

    cv::cuda::GpuMat mGpuImage;

    mGpuImage.upload(mImage.clone());

    cv::cuda::resize(mGpuImage, mGpuImage, cv::Size(m_iWidthModel, m_iHeightModel));
    cv::resize(mImage, mImage, cv::Size(m_iWidthModel, m_iHeightModel));
    stObjects.inputCPUBGR = mImage;
    cv::cuda::cvtColor(mGpuImage, mGpuImage, cv::COLOR_BGR2RGB);

    cv::cuda::GpuMat mGpuFloat;
    mGpuImage.convertTo(mGpuFloat, CV_32FC3, 1.f / 255.f);

    // cv::cuda::subtract(mGpuFloat, cv::Scalar(m_subVals[0], m_subVals[1], m_subVals[2]), mGpuFloat, cv::noArray(), -1);
    // cv::cuda::divide(mGpuFloat, cv::Scalar(m_divVals[0], m_divVals[1], m_divVals[2]), mGpuFloat, 1, -1);

    cv::cuda::GpuMat mGpuTranspose(m_iHeightModel, m_iWidthModel, CV_32FC3);
    size_t size = m_iWidthModel * m_iHeightModel * sizeof(float);
    std::vector<cv::cuda::GpuMat> mGpuChannels
            {
                    cv::cuda::GpuMat(m_iHeightModel, m_iWidthModel, CV_32FC1, &(mGpuTranspose.ptr()[0])),
                    cv::cuda::GpuMat(m_iHeightModel, m_iWidthModel, CV_32FC1, &(mGpuTranspose.ptr()[size])),
                    cv::cuda::GpuMat(m_iHeightModel, m_iWidthModel, CV_32FC1, &(mGpuTranspose.ptr()[size * 2]))
            };
    cv::cuda::split(mGpuFloat, mGpuChannels);

    cudaMemcpy(buffers[umpIOTensors[m_strInputName][0]], mGpuTranspose.ptr<float>(), umpIOTensors[m_strInputName][1] * sizeof(float), cudaMemcpyHostToDevice);
}

void TRTPPOcrDet::run(cv::Mat &mImage, stTextBox_t& stObjects)
{
    preprocess(mImage, stObjects);
    shpTRTRunner->runModel(this->buffers);
    postprocess(this->buffers, stObjects);
}

void TRTPPOcrDet::getContourArea(const std::vector<std::vector<float>> &box, float unclip_ratio, float &distance)
{
    int pts_num = 4;
    float area = 0.0f;
    float dist = 0.0f;
    for (int i = 0; i < pts_num; i++) {
        area += box[i][0] * box[(i + 1) % pts_num][1] -
                box[i][1] * box[(i + 1) % pts_num][0];
        dist += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) *
                      (box[i][0] - box[(i + 1) % pts_num][0]) +
                      (box[i][1] - box[(i + 1) % pts_num][1]) *
                      (box[i][1] - box[(i + 1) % pts_num][1]));
    }
    area = fabs(float(area / 2.0));

    distance = area * unclip_ratio / dist;
}

cv::RotatedRect TRTPPOcrDet::unClip(std::vector<std::vector<float>> box, const float &unclip_ratio) {
    float distance = 1.0;

    getContourArea(box, unclip_ratio, distance);

    ClipperLib::ClipperOffset offset;
    ClipperLib::Path p;
    p << ClipperLib::IntPoint(int(box[0][0]), int(box[0][1]))
      << ClipperLib::IntPoint(int(box[1][0]), int(box[1][1]))
      << ClipperLib::IntPoint(int(box[2][0]), int(box[2][1]))
      << ClipperLib::IntPoint(int(box[3][0]), int(box[3][1]));
    offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

    ClipperLib::Paths soln;
    offset.Execute(soln, distance);
    std::vector<cv::Point2f> points;

    for (int j = 0; j < soln.size(); j++) {
        for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
            points.emplace_back(soln[j][i].X, soln[j][i].Y);
        }
    }
    cv::RotatedRect res;
    if (points.size() <= 0) {
        res = cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
    } else {
        res = cv::minAreaRect(points);
    }
    return res;
}

float **TRTPPOcrDet::Mat2Vec(cv::Mat mat) {
    auto **array = new float *[mat.rows];
    for (int i = 0; i < mat.rows; ++i)
        array[i] = new float[mat.cols];
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            array[i][j] = mat.at<float>(i, j);
        }
    }

    return array;
}

std::vector<std::vector<int>> TRTPPOcrDet::orderPointsClockwise(std::vector<std::vector<int>> pts) {
    std::vector<std::vector<int>> box = pts;
    std::sort(box.begin(), box.end(), xSortInt);

    std::vector<std::vector<int>> leftmost = {box[0], box[1]};
    std::vector<std::vector<int>> rightmost = {box[2], box[3]};

    if (leftmost[0][1] > leftmost[1][1])
        std::swap(leftmost[0], leftmost[1]);

    if (rightmost[0][1] > rightmost[1][1])
        std::swap(rightmost[0], rightmost[1]);

    std::vector<std::vector<int>> rect = {leftmost[0], rightmost[0], rightmost[1],
                                          leftmost[1]};
    return rect;
}

std::vector<std::vector<float>> TRTPPOcrDet::Mat2Vector(cv::Mat mat) {
    std::vector<std::vector<float>> img_vec;
    std::vector<float> tmp;

    for (int i = 0; i < mat.rows; ++i) {
        tmp.clear();
        for (int j = 0; j < mat.cols; ++j) {
            tmp.push_back(mat.at<float>(i, j));
        }
        img_vec.push_back(tmp);
    }
    return img_vec;
}

bool TRTPPOcrDet::xSortFp32(std::vector<float> a, std::vector<float> b) {
    if (a[0] != b[0])
        return a[0] < b[0];
    return false;
}

bool TRTPPOcrDet::xSortInt(std::vector<int> a, std::vector<int> b) {
    if (a[0] != b[0])
        return a[0] < b[0];
    return false;
}

std::vector<std::vector<float>> TRTPPOcrDet::getMiniBoxes(cv::RotatedRect box, float &ssid) {
    ssid = std::max(box.size.width, box.size.height);

    cv::Mat points;
    cv::boxPoints(box, points);

    auto array = Mat2Vector(points);
    std::sort(array.begin(), array.end(), xSortFp32);

    std::vector<float> idx1 = array[0], idx2 = array[1], idx3 = array[2],
            idx4 = array[3];
    if (array[3][1] <= array[2][1]) {
        idx2 = array[3];
        idx3 = array[2];
    } else {
        idx2 = array[2];
        idx3 = array[3];
    }
    if (array[1][1] <= array[0][1]) {
        idx1 = array[1];
        idx4 = array[0];
    } else {
        idx1 = array[0];
        idx4 = array[1];
    }

    array[0] = idx1;
    array[1] = idx2;
    array[2] = idx3;
    array[3] = idx4;

    return array;
}

float TRTPPOcrDet::boxScoreFast(std::vector<std::vector<float>> box_array, cv::Mat pred) {
    auto array = box_array;
    int width = pred.cols;
    int height = pred.rows;

    float box_x[4] = {array[0][0], array[1][0], array[2][0], array[3][0]};
    float box_y[4] = {array[0][1], array[1][1], array[2][1], array[3][1]};

    int xmin = clamp(int(std::floor(*(std::min_element(box_x, box_x + 4)))), 0,
                     width - 1);
    int xmax = clamp(int(std::ceil(*(std::max_element(box_x, box_x + 4)))), 0,
                     width - 1);
    int ymin = clamp(int(std::floor(*(std::min_element(box_y, box_y + 4)))), 0,
                     height - 1);
    int ymax = clamp(int(std::ceil(*(std::max_element(box_y, box_y + 4)))), 0,
                     height - 1);

    cv::Mat mask;
    mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

    cv::Point root_point[4];
    root_point[0] = cv::Point(int(array[0][0]) - xmin, int(array[0][1]) - ymin);
    root_point[1] = cv::Point(int(array[1][0]) - xmin, int(array[1][1]) - ymin);
    root_point[2] = cv::Point(int(array[2][0]) - xmin, int(array[2][1]) - ymin);
    root_point[3] = cv::Point(int(array[3][0]) - xmin, int(array[3][1]) - ymin);
    const cv::Point *ppt[1] = {root_point};
    int npt[] = {4};
    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

    cv::Mat croppedImg;
    pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
            .copyTo(croppedImg);

    auto score = cv::mean(croppedImg, mask)[0];
    return score;
}

float TRTPPOcrDet::polygonScoreAcc(std::vector<cv::Point> contour, cv::Mat pred) {
    int width = pred.cols;
    int height = pred.rows;
    std::vector<float> box_x;
    std::vector<float> box_y;
    for (int i = 0; i < contour.size(); ++i) {
        box_x.push_back(contour[i].x);
        box_y.push_back(contour[i].y);
    }

    int xmin =
            clamp(int(std::floor(*(std::min_element(box_x.begin(), box_x.end())))), 0,
                  width - 1);
    int xmax =
            clamp(int(std::ceil(*(std::max_element(box_x.begin(), box_x.end())))), 0,
                  width - 1);
    int ymin =
            clamp(int(std::floor(*(std::min_element(box_y.begin(), box_y.end())))), 0,
                  height - 1);
    int ymax =
            clamp(int(std::ceil(*(std::max_element(box_y.begin(), box_y.end())))), 0,
                  height - 1);

    cv::Mat mask;
    mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);


    cv::Point* rook_point = new cv::Point[contour.size()];

    for (int i = 0; i < contour.size(); ++i) {
        rook_point[i] = cv::Point(int(box_x[i]) - xmin, int(box_y[i]) - ymin);
    }
    const cv::Point *ppt[1] = {rook_point};
    int npt[] = {int(contour.size())};


    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

    cv::Mat croppedImg;
    pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)).copyTo(croppedImg);
    float score = cv::mean(croppedImg, mask)[0];

    delete []rook_point;
    return score;
}

std::vector<std::vector<std::vector<int>>>
TRTPPOcrDet::boxesFromBitmap(const cv::Mat pred, const cv::Mat bitmap, const float &box_thresh,
                             const float &det_db_unclip_ratio, const bool &use_polygon_score) {
    const int min_size = 0;
    const int max_candidates = 1000;

    int width = bitmap.cols;
    int height = bitmap.rows;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST,
                     cv::CHAIN_APPROX_SIMPLE);

    int num_contours =
            contours.size() >= max_candidates ? max_candidates : contours.size();

    std::vector<std::vector<std::vector<int>>> boxes;

    for (int _i = 0; _i < num_contours; _i++) {
        if (contours[_i].size() <= 2) {
            continue;
        }
        float ssid;
        cv::RotatedRect box = cv::minAreaRect(contours[_i]);
        auto array = getMiniBoxes(box, ssid);

        auto box_for_unclip = array;
        // end get_mini_box

        if (ssid < min_size) {
            continue;
        }

        float score;
        if (use_polygon_score)
            /* compute using polygon*/
            score = polygonScoreAcc(contours[_i], pred);
        else
            score = boxScoreFast(array, pred);

        if (score < box_thresh)
            continue;

        // start for unclip
        cv::RotatedRect points = unClip(box_for_unclip, det_db_unclip_ratio);
        if (points.size.height < 1.001 && points.size.width < 1.001) {
            continue;
        }
        // end for unclip

        cv::RotatedRect clipbox = points;
        auto cliparray = getMiniBoxes(clipbox, ssid);

        if (ssid < min_size + 2)
            continue;

        int dest_width = pred.cols;
        int dest_height = pred.rows;
        std::vector<std::vector<int>> intcliparray;

        for (int num_pt = 0; num_pt < 4; num_pt++) {
            std::vector<int> a{int(clampf(roundf(cliparray[num_pt][0] / float(width) *
                                                 float(dest_width)),
                                          0, float(dest_width))),
                               int(clampf(roundf(cliparray[num_pt][1] /
                                                 float(height) * float(dest_height)),
                                          0, float(dest_height)))};
            intcliparray.push_back(a);
        }
        boxes.push_back(intcliparray);
    } // end for
    return boxes;
}

std::vector<std::vector<std::vector<int>>>
TRTPPOcrDet::filterTagDetRes(std::vector<std::vector<std::vector<int>>> boxes, float ratio_h, float ratio_w,
                             cv::Mat srcimg) {
    int oriimg_h = srcimg.rows;
    int oriimg_w = srcimg.cols;

    std::vector<std::vector<std::vector<int>>> root_points;
    for (int n = 0; n < boxes.size(); n++) {
        boxes[n] = orderPointsClockwise(boxes[n]);
        for (int m = 0; m < boxes[0].size(); m++) {
            boxes[n][m][0] /= ratio_w;
            boxes[n][m][1] /= ratio_h;

            boxes[n][m][0] = int(_min(_max(boxes[n][m][0], 0), oriimg_w - 1));
            boxes[n][m][1] = int(_min(_max(boxes[n][m][1], 0), oriimg_h - 1));
        }
    }

    for (int n = 0; n < boxes.size(); n++) {
        int rect_width, rect_height;
        rect_width = int(sqrt(pow(boxes[n][0][0] - boxes[n][1][0], 2) +
                              pow(boxes[n][0][1] - boxes[n][1][1], 2)));
        rect_height = int(sqrt(pow(boxes[n][0][0] - boxes[n][3][0], 2) +
                               pow(boxes[n][0][1] - boxes[n][3][1], 2)));
        
        if (rect_width <= 4 || rect_height <= 4)
            continue;
        root_points.push_back(boxes[n]);
    }
    return root_points;
}