/**
 * @file Common.hpp
 * @author HuyNQ (huy.nguyen@gpstech.vn)
 * @brief 
 * @version 0.1
 * @date 2025-01-20
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef COMMON_HPP
#define COMMON_HPP
#include "iostream"
#include <string>
#include <cmath>
#include <vector>
#include <numeric>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <cstring>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <boost/filesystem.hpp>
#include "Logger.hpp"

namespace fs = boost::filesystem;

namespace Common{
    inline void createFolder(std::string dir){
        if (access(dir.c_str(), F_OK) != -1) {
            INFO("Directory already exists");
        } else {
            // Sử dụng hàm mkdir() để tạo thư mục
            int result = mkdir(dir.c_str(), 0777);

            if (result == 0) {
                INFO("Directory created successfully");
            } else {
                ERROR("Failed to create directory");
            }
        }
    }
    // Hàm tính tích vô hướng của hai vector
    inline float dotProduct(const std::vector<float>& vec1, const std::vector<float>& vec2) {
        return std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0);
    }

    // Hàm tính độ dài của một vector
    inline float norm(const std::vector<float>& vec) {
        return std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0));
    }

    // Hàm tính khoảng cách cosine giữa hai vector
    inline float cosineDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
        float dot_product = dotProduct(vec1, vec2);
        float norm_vec1 = norm(vec1);
        float norm_vec2 = norm(vec2);
        return (dot_product / (norm_vec1 * norm_vec2));
    }
    inline int saveFileNameLabel(std::string filename, int& x){
        std::ofstream file(filename);
        file << x;
        file.close();
        return 0;
    }

    inline int readFileNameLabel(std::string filename, int& x){
        std::ifstream file(filename);
        if (!file.good()) {
            x = 0;
            return 0;
        }
        file >> x;
        file.close();
        return x;
    }
    inline void createDirectory(const std::string& directoryPath) {
        if (!boost::filesystem::exists(directoryPath)) {
            if (boost::filesystem::create_directories(directoryPath)) {
                INFO("Directory {} created successfully.",directoryPath);
            } else{
                ERROR("Fails {} created.",directoryPath);
            }
        } else {
            INFO("Directory {} had existed.",directoryPath);
        }
    }

    inline void removeElementsByIndex(std::vector<std::string>& data, std::vector<int64_t>& indices) {
        // Sắp xếp các chỉ mục giảm dần
        std::sort(indices.begin(), indices.end(), std::greater<int>());

        for (int index : indices) {
            if (index >= 0 && index < data.size()) {
                data.erase(data.begin() + index);
            }
        }
    }
    inline float convertCosineDistanceToSimilarity(float cosineDistance, float alpha= 0.85) {
        float percentage = std::pow(cosineDistance, alpha) * 100;
        // round to 2 decimal places
        percentage = std::round(percentage * 100) / 100;
        return percentage;
    }

    inline std::vector<std::string> loadSubfolders(const std::string& parent_folder) {
        std::vector<std::string> subfolders;

        // Kiểm tra nếu thư mục cha tồn tại
        if (fs::exists(parent_folder) && fs::is_directory(parent_folder)) {
            for (const auto& entry : fs::directory_iterator(parent_folder)) {
                if (fs::is_directory(entry.path())) {
                    subfolders.push_back(entry.path().string()); // Lưu đường dẫn thư mục con
                }
            }
        } else {
            ERROR("Error: The provided path is not a directory");
        }

        return subfolders;
    }

    inline std::vector<std::string> listFiles(const fs::path& directory) {
        std::vector<std::string> files;

        if (fs::is_directory(directory)) {
            for (const auto& entry : fs::directory_iterator(directory)) {
                if (fs::is_regular_file(entry.path())) {
                    files.push_back(entry.path().string());
                }
            }
        } else {
            // WARN("Error: The provided path is not a directory");
            ERROR("Error: The provided path is not a directory");
        }

        return files;
    }
}

#endif //COMMON_HPP