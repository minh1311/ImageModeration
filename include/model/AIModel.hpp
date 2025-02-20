/**
 * @file AIModel.hpp
 * @author HuyNQ
 * @brief 
 * @version 0.1
 * @date 2024-06-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef AIMODEL_HPP
#define AIMODEL_HPP

#include "json.hpp"
#include "AbstractRuntime.hpp"
#include "RuntimeFactory.hpp"
#include <vector>
#include "JsonTypes.hpp"

class AIModel
{
public:
    AIModel(nlohmann::json &jModelConfig, const std::string& strModelName);
    ~AIModel();
    template<typename T>
    void run(cv::Mat &mImage, T &var);

    template<typename T, typename X>
    void run(cv::Mat &mImage, T &var,X &var2);

private:
    AbstractRuntime* m_pRuntime = nullptr;
};

template<typename T>
void AIModel::run(cv::Mat &mImage, T &var)
{
    m_pRuntime->run(mImage, var);
}

template<typename T, typename X>
void AIModel::run(cv::Mat &mImage, T &var, X &var2)
{
    m_pRuntime->run(mImage, var, var2);
}


#endif //AIMODEL_HPP