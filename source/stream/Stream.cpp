#include "stream/Stream.hpp"

#include "stream/OpencvStream.hpp"
#include "stream/FFmpegStream.hpp"

std::shared_ptr<Stream> Stream::CreateStream(std::string strUrl)
{
    std::shared_ptr<Stream> shpStream;

    // check if strUrl is video file
    std::string videoExtensions[] = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"};

    // Lấy phần mở rộng của file
    size_t dotPos = strUrl.rfind('.');


    std::string extension = strUrl.substr(dotPos);
    bool bisVideoFile = false;
    // Kiểm tra xem phần mở rộng có nằm trong danh sách các phần mở rộng video hay không
    for (const auto &videoExt : videoExtensions) {
        if (extension == videoExt) {
            bisVideoFile = true;
        }
    }
    if (bisVideoFile)
    {
        shpStream = std::make_shared<OpencvStream>(strUrl);
    }
    else {
        shpStream = std::make_shared<FFmpegStream>(strUrl);
    }

    return shpStream;
}