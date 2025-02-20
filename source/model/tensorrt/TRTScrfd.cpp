/**
 * @file TRTScrfd.cpp
 * @author HuyNQ
 * @brief 
 * @version 0.1
 * @date 2024-06-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "TRTScrfd.hpp"

#include <algorithm>

TRTScrfd::TRTScrfd(nlohmann::json& jModelConfig)
    : TRTModel(jModelConfig, JN_MODEL_SCRFD)
{
    m_iWidthModel = umpIOTensorsShape[m_strInputName].d[3];
    m_iHeightModel = umpIOTensorsShape[m_strInputName].d[2];

    for (int i = 0; i < m_featStrideFpn.size(); i++)
    {
        float stride = m_featStrideFpn[i];
        int anchorWidth = m_iWidthModel / stride;
        int anchorHeight = m_iHeightModel / stride;

        std::vector<std::vector<float>> anchorCenters;
        for (int j = 0; j < anchorWidth; j++)
        {
            for (int k = 0; k < anchorHeight; k++)
            {
                std::vector<float> anchorCenter = {k * stride, j * stride};
                anchorCenters.push_back(anchorCenter);
                anchorCenters.push_back(anchorCenter);
            }
        }
        m_anchors.push_back(anchorCenters);
    }
}


TRTScrfd::~TRTScrfd()
{

}

void TRTScrfd::run(const cv::Mat &mImage, std::vector<cerberus::FaceObject> &vtFaceObject) 
{
    cv::Mat img = mImage.clone();
    preprocess(img);
    shpTRTRunner->runModel(this->buffers);
    postprocess(this->buffers, vtFaceObject);
}

void TRTScrfd::preprocess(cv::Mat& mImage)
{
    int m_iInputWidth = mImage.cols;
    int m_iInputHeight = mImage.rows;

    // Ensure input_size is set (though in this case, it's always set to 640 in the Python code)
    assert(m_iInputWidth > 0 && m_iInputHeight > 0);

    float im_ratio = static_cast<float>(m_iInputHeight) / m_iInputWidth;
    float model_ratio = static_cast<float>(m_iWidthModel) / m_iHeightModel;
    int new_width, new_height;

    if (im_ratio > model_ratio) {
        new_height = m_iHeightModel;
        new_width = static_cast<int>(new_height / im_ratio);
    } else {
        new_width = m_iWidthModel;
        new_height = static_cast<int>(new_width * im_ratio);
    }

    m_fRatioWidth = 1.0f / (new_width / static_cast<float>(m_iInputWidth));
    m_fRatioHeight = 1.0f / (new_height / static_cast<float>(m_iInputHeight));

    float det_scale = static_cast<float>(new_height) / m_iInputHeight;

    cv::Mat resized_img;
    cv::resize(mImage, resized_img, cv::Size(new_width, new_height));

    // Create a GPU matrix and upload the resized image to the GPU
    cv::cuda::GpuMat gpu_resized_img, gpu_input;
    gpu_resized_img.upload(resized_img);

    // Create a black image of size 640x640 on GPU
    gpu_input = cv::cuda::GpuMat(cv::Size(640, 640), CV_8UC3, cv::Scalar(0, 0, 0));

    // Copy the resized image to the top-left corner of the 640x640 image
    gpu_resized_img.copyTo(gpu_input(cv::Rect(0, 0, new_width, new_height)));

    // Convert the image from BGR to RGB
    cv::cuda::cvtColor(gpu_input, gpu_input, cv::COLOR_BGR2RGB);
    // std::cout << "size: " << gpu_input.cols << " " << gpu_input.rows << std::endl;

// Convert the image to float
    cv::cuda::GpuMat mGpuFloat;
    gpu_input.convertTo(mGpuFloat, CV_32FC3, 1.f);

    cv::cuda::subtract(mGpuFloat, cv::Scalar(m_subVals[0], m_subVals[1], m_subVals[2]), mGpuFloat, cv::noArray(), -1);
    cv::cuda::divide(mGpuFloat, cv::Scalar(m_divVals[0], m_divVals[1], m_divVals[2]), mGpuFloat, 1, -1);

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

void TRTScrfd::postprocess(std::vector<void*>& buffers, std::vector<cerberus::FaceObject>& stObjects)
{
    stObjects.clear();

    float score_8[umpIOTensors[m_strOutputScore8][1]];
    cudaMemcpy(score_8, buffers[umpIOTensors[m_strOutputScore8][0]], umpIOTensors[m_strOutputScore8][1] * sizeof(float), cudaMemcpyDeviceToHost);
    float bbox_8[umpIOTensors[m_strOutputBbox8][1]];
    cudaMemcpy(bbox_8, buffers[umpIOTensors[m_strOutputBbox8][0]], umpIOTensors[m_strOutputBbox8][1] * sizeof(float), cudaMemcpyDeviceToHost);
    float kps_8[umpIOTensors[m_strOutputKps8][1]];
    cudaMemcpy(kps_8, buffers[umpIOTensors[m_strOutputKps8][0]], umpIOTensors[m_strOutputKps8][1] * sizeof(float), cudaMemcpyDeviceToHost);

    float score_16[umpIOTensors[m_strOutputScore16][1]];
    cudaMemcpy(score_16, buffers[umpIOTensors[m_strOutputScore16][0]], umpIOTensors[m_strOutputScore16][1] * sizeof(float), cudaMemcpyDeviceToHost);
    float bbox_16[umpIOTensors[m_strOutputBbox16][1]];
    cudaMemcpy(bbox_16, buffers[umpIOTensors[m_strOutputBbox16][0]], umpIOTensors[m_strOutputBbox16][1] * sizeof(float), cudaMemcpyDeviceToHost);
    float kps_16[umpIOTensors[m_strOutputKps16][1]];
    cudaMemcpy(kps_16, buffers[umpIOTensors[m_strOutputKps16][0]], umpIOTensors[m_strOutputKps16][1] * sizeof(float), cudaMemcpyDeviceToHost);

    float score_32[umpIOTensors[m_strOutputScore32][1]];
    cudaMemcpy(score_32, buffers[umpIOTensors[m_strOutputScore32][0]], umpIOTensors[m_strOutputScore32][1] * sizeof(float), cudaMemcpyDeviceToHost);
    float bbox_32[umpIOTensors[m_strOutputBbox32][1]];
    cudaMemcpy(bbox_32, buffers[umpIOTensors[m_strOutputBbox32][0]], umpIOTensors[m_strOutputBbox32][1] * sizeof(float), cudaMemcpyDeviceToHost);
    float kps_32[umpIOTensors[m_strOutputKps32][1]];
    cudaMemcpy(kps_32, buffers[umpIOTensors[m_strOutputKps32][0]], umpIOTensors[m_strOutputKps32][1] * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<cerberus::FaceObject> detected;

    for (int i = 0; i < umpIOTensors[m_strOutputScore8][1]; i++)
    {
        float fScore = score_8[i];
        if (fScore >= m_fScoreThreshold)
        {
            float stride = m_featStrideFpn[0];
            std::vector<float> anchorCenter = m_anchors[0][i];
            
            float x1 = (anchorCenter[0] - bbox_8[i * 4 + 0] * stride) * m_fRatioWidth;
            float y1 = (anchorCenter[1] - bbox_8[i * 4 + 1] * stride) * m_fRatioHeight;
            float x2 = (anchorCenter[0] + bbox_8[i * 4 + 2] * stride) * m_fRatioWidth;
            float y2 = (anchorCenter[1] + bbox_8[i * 4 + 3] * stride) * m_fRatioHeight;

            cerberus::Box bbox;
            bbox.x = x1;
            bbox.y = y1;
            bbox.w = x2 - x1;
            bbox.h = y2 - y1;

            std::vector<cerberus::Keypoint> kps;
            for (int k = 0; k < 5; k++)
            {
                cerberus::Keypoint rfPt;
                rfPt.x = (anchorCenter[0] + kps_8[i * 10 + 2 * k] * stride) * m_fRatioWidth;
                rfPt.y = (anchorCenter[1] + kps_8[i * 10 + 2 * k + 1] * stride) * m_fRatioHeight;
                kps.push_back(rfPt);
            }

            cerberus::FaceObject stFace;
            stFace.prob = fScore;
            stFace.rect = bbox;
            stFace.landmark = kps;

            detected.push_back(stFace);
        }
    }

    for (int i = 0; i < umpIOTensors[m_strOutputScore16][1]; i++)
    {
        float fScore = score_16[i];
        if (fScore >= m_fScoreThreshold)
        {
            float stride = m_featStrideFpn[1];
            std::vector<float> anchorCenter = m_anchors[1][i];
            
            float x1 = (anchorCenter[0] - bbox_16[i * 4 + 0] * stride) * m_fRatioWidth;
            float y1 = (anchorCenter[1] - bbox_16[i * 4 + 1] * stride) * m_fRatioHeight;
            float x2 = (anchorCenter[0] + bbox_16[i * 4 + 2] * stride) * m_fRatioWidth;
            float y2 = (anchorCenter[1] + bbox_16[i * 4 + 3] * stride) * m_fRatioHeight;

            cerberus::Box bbox;
            bbox.x = x1;
            bbox.y = y1;
            bbox.w = x2 - x1;
            bbox.h = y2 - y1;

            std::vector<cerberus::Keypoint> kps;
            for (int k = 0; k < 5; k++)
            {
                cerberus::Keypoint rfPt;
                rfPt.x = (anchorCenter[0] + kps_16[i * 10 + 2 * k] * stride) * m_fRatioWidth;
                rfPt.y = (anchorCenter[1] + kps_16[i * 10 + 2 * k + 1] * stride) * m_fRatioHeight;
                kps.push_back(rfPt);
            }

            cerberus::FaceObject stFace;
            stFace.prob = fScore;
            stFace.rect = bbox;
            stFace.landmark = kps;

            detected.push_back(stFace);
        }
    }

    for (int i = 0; i < umpIOTensors[m_strOutputScore32][1]; i++)
    {
        float fScore = score_32[i];
        if (fScore >= m_fScoreThreshold)
        {
            float stride = m_featStrideFpn[2];
            std::vector<float> anchorCenter = m_anchors[2][i];
            
            float x1 = (anchorCenter[0] - bbox_32[i * 4 + 0] * stride) * m_fRatioWidth;
            float y1 = (anchorCenter[1] - bbox_32[i * 4 + 1] * stride) * m_fRatioHeight;
            float x2 = (anchorCenter[0] + bbox_32[i * 4 + 2] * stride) * m_fRatioWidth;
            float y2 = (anchorCenter[1] + bbox_32[i * 4 + 3] * stride) * m_fRatioHeight;

            cerberus::Box bbox;
            bbox.x = x1;
            bbox.y = y1;
            bbox.w = x2 - x1;
            bbox.h = y2 - y1;

            std::vector<cerberus::Keypoint> kps;
            for (int k = 0; k < 5; k++)
            {
                cerberus::Keypoint rfPt;
                rfPt.x = (anchorCenter[0] + kps_32[i * 10 + 2 * k] * stride) * m_fRatioWidth;
                rfPt.y = (anchorCenter[1] + kps_32[i * 10 + 2 * k + 1] * stride) * m_fRatioHeight;
                kps.push_back(rfPt);
            }

            cerberus::FaceObject stFace;
            stFace.prob = fScore;
            stFace.rect = bbox;
            stFace.landmark = kps;

            detected.push_back(stFace);
        }
    }

    std::sort(detected.begin(), detected.end(), 
                [](const cerberus::FaceObject& a, const cerberus::FaceObject& b)
                {
                    return (a.prob > b.prob);
                });

    while (detected.size() > 0)
    {
        cerberus::FaceObject stFace = detected[0];
        stObjects.push_back(stFace);
        detected.erase(detected.begin());

        float area = (stFace.rect.w + 1) * (stFace.rect.h + 1);

        for (int i = 0; i < detected.size(); i++)
        {
            cerberus::FaceObject stFaceI = detected[i];
            float xx1 = std::max(stFace.rect.x, stFaceI.rect.x);
            float yy1 = std::max(stFace.rect.y, stFaceI.rect.y);
            float xx2 = std::min((stFace.rect.x + stFace.rect.w), (stFaceI.rect.x + stFaceI.rect.w));
            float yy2 = std::min((stFace.rect.y + stFace.rect.h), (stFaceI.rect.y + stFaceI.rect.h));
            float w = std::max(0.f, xx2 - xx1 + 1);
            float h = std::max(0.f, yy2 - yy1 + 1);

            float inter = w * h;
            float areaI = ((stFaceI.rect.x + stFaceI.rect.w) - stFaceI.rect.x + 1) * ((stFaceI.rect.y + stFaceI.rect.h) - stFaceI.rect.y + 1);
            float ovr = inter / (area + areaI - inter);
            if (ovr > m_fNMSThreshold)
            {
                detected.erase(detected.begin() + i);
                i--;
            }
        }
    }
}