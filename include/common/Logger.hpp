#ifndef Logger_hpp
#define Logger_hpp
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
/**
 * @brief SPDLOG_LEVEL
 * 
 * @param SPDLOG_LEVEL_TRACE 0
 * @param SPDLOG_LEVEL_DEBUG 1
 * @param SPDLOG_LEVEL_INFO 2
 * @param SPDLOG_LEVEL_WARN 3
 * @param SPDLOG_LEVEL_ERROR 4
 * @param SPDLOG_LEVEL_CRITICAL 5
 * @param SPDLOG_LEVEL_OFF 6
 */

#include "spdlog/spdlog.h"
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
/**
 * @brief spdlog::level
 * 
 * @param spdlog::level::trace
 * @param spdlog::level::debug
 * @param spdlog::level::info
 * @param spdlog::level::warn
 * @param spdlog::level::err
 * @param spdlog::level::critical
 * @param spdlog::level::off
 */


inline void setup()
{
    // Tạo file sink với level TRACE
    auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>("logs/log.log", 0, 0);
    file_sink->set_level(spdlog::level::trace);  // Ghi tất cả các level vào file

    // Tạo console sink với level từ SPDLOG_ACTIVE_LEVEL
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(static_cast<spdlog::level::level_enum>(SPDLOG_CONSOLE_LOG));

    // Tạo logger với cả hai sink
    std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
    auto combined_logger = std::make_shared<spdlog::logger>("combined_logger", sinks.begin(), sinks.end());
    
    // Đặt level của logger là thấp nhất (TRACE) để cho phép ghi mọi level
    combined_logger->set_level(spdlog::level::trace);
    
    // Thiết lập định dạng log
    combined_logger->set_pattern("[%H:%M:%S.%e][%^%l%$][%s:%#] %v");

    // Đặt logger mặc định
    spdlog::set_default_logger(combined_logger);

    // Tự động flush log từ mức trace trở lên
    spdlog::flush_on(spdlog::level::trace);
}

inline std::shared_ptr<spdlog::logger> logger()
{
    if (!spdlog::get("combined_logger"))
    {
        setup();
    }
    return spdlog::get("combined_logger");
}

#define TRACE(...) SPDLOG_LOGGER_TRACE(logger(), __VA_ARGS__);
#define DEBUG(...) SPDLOG_LOGGER_DEBUG(logger(), __VA_ARGS__);
#define INFO(...) SPDLOG_LOGGER_INFO(logger(), __VA_ARGS__);
#define WARN(...) SPDLOG_LOGGER_WARN(logger(), __VA_ARGS__);
#define ERROR(...) SPDLOG_LOGGER_ERROR(logger(), __VA_ARGS__);
#define CRITICAL(...) SPDLOG_LOGGER_CRITICAL(logger(), __VA_ARGS__);

#endif // Logger_hpp
