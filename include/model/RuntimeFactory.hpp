/**
 * @file RuntimeFactory.hpp
 * @author HuyNQ
 * @brief 
 * @version 0.1
 * @date 2024-06-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef RUNTIMEFACTORY_HPP
#define RUNTIMEFACTORY_HPP

#include "AbstractRuntime.hpp"
#include "JsonTypes.hpp"
#include "json.hpp"

class RuntimeFactory
{
public:
    RuntimeFactory();
    ~RuntimeFactory();
    static AbstractRuntime* createRuntime(std::string& strModel, std::string& strFrameWork, nlohmann::json& jConfigModel);
};


#endif //RUNTIMEFACTORY_HPP