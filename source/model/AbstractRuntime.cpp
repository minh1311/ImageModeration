/**
 * @file AbstractRuntime.cpp
 * @author HuyNQ
 * @brief 
 * @version 0.1
 * @date 2024-06-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "AbstractRuntime.hpp"

std::vector<float> AbstractRuntime::softmax(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    // Tìm giá trị lớn nhất trong vector để tránh vấn đề tràn số
    float max_val = *std::max_element(input.begin(), input.end());
    
    // Tính exp cho mỗi phần tử và tổng của chúng
    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Chia mỗi phần tử cho tổng để chuẩn hóa
    for (float& val : output) {
        val /= sum;
    }
    
    return output;
}

cv::Rect AbstractRuntime::calculateFASBox(cerberus::CerberusFace &box, int w, int h, float fScale) {
    int x = static_cast<int>(box.x);
    int y = static_cast<int>(box.y);
    int box_width = static_cast<int>(box.w);
    int box_height = static_cast<int>(box.h);

    // int shift_x = static_cast<int>(box_width * config.shift_x);
    // int shift_y = static_cast<int>(box_height * config.shift_y);
    int shift_x =0;
    int shift_y =0;
    float scale = std::min(fScale, std::min((w-1)/(float)box_width, (h-1)/(float)box_height));

    int box_center_x = box_width / 2 + x;
    int box_center_y = box_height / 2 + y;

    int new_width = static_cast<int>(box_width * scale);
    int new_height = static_cast<int>(box_height * scale);

    int left_top_x = box_center_x - new_width / 2 + shift_x;
    int left_top_y = box_center_y - new_height / 2 + shift_y;
    int right_bottom_x = box_center_x + new_width / 2 + shift_x;
    int right_bottom_y = box_center_y + new_height / 2 + shift_y;

    if (left_top_x < 0) {
        right_bottom_x -= left_top_x;
        left_top_x = 0;
    }

    if (left_top_y < 0) {
        right_bottom_y -= left_top_y;
        left_top_y = 0;
    }

    if (right_bottom_x >= w) {
        int s = right_bottom_x - w + 1;
        left_top_x -= s;
        right_bottom_x -= s;
    }

    if (right_bottom_y >= h) {
        int s = right_bottom_y - h + 1;
        left_top_y -= s;
        right_bottom_y -= s;
    }
    return cv::Rect(left_top_x, left_top_y, new_width, new_height);
}