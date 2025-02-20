/**
 * @file model.cpp
 * @author HuyNQ
 * @brief 
 * @version 0.1
 * @date 2024-06-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "AIModel.hpp"

AIModel::AIModel(nlohmann::json &jModelConfig,const std::string& strModelName )
{
    std::string modelName =strModelName;
    std::string strFrameWork = jModelConfig[modelName][JN_FRAMEWORK].get<std::string>();
    std::cout << "strFrameWork: " << strFrameWork << std::endl;
    std::cout << "strModelName: " << strModelName << std::endl;
    m_pRuntime = RuntimeFactory::createRuntime(modelName, strFrameWork, jModelConfig);
    std::cout << "Create "<< strFrameWork<<" "<<strModelName<<" successfully!" << std::endl;

}
AIModel::~AIModel()
{
    delete m_pRuntime;
}
