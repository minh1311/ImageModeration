#include "TRTCommon.hpp"

#include <iostream>

/**
 * @brief
*/
void TRTLogger::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept
{
    if (severity <= Severity::kWARNING)
    {
        std::cout << TagTRTLogger << msg << std::endl;
    }
}
