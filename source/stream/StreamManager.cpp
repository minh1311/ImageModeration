#include "stream/StreamManager.hpp"

StreamManager& StreamManager::GetInstance()
{
    static StreamManager pInstance;
    return pInstance;
}

std::shared_ptr<Stream> StreamManager::getStream(std::string strUrl)
{
    {
        const std::lock_guard<std::mutex> mtLock(m_mtLock);
        auto it = m_mpStreams.find(strUrl);
        if (it != m_mpStreams.end()) {
            // Stream đã tồn tại, tăng bộ đếm và trả về
            it->second.iCount++;
            return it->second.pStream;
        }
    }
    std::shared_ptr<Stream> shpStream = nullptr;
    {
        // Tạo stream
        shpStream = Stream::CreateStream(strUrl);
        if (!shpStream->isInitialized()) {
            return nullptr;
        }
    }

    // Cập nhật vào m_mpStreams
    {
        const std::lock_guard<std::mutex> mtLock(m_mtLock);
        StreamValue_t stream;
        stream.iCount = 1;
        stream.pStream = shpStream;
        m_mpStreams[strUrl] = stream;
    }

    return shpStream;
}

void StreamManager::unregister(std::string strUrl)
{
    const std::lock_guard<std::mutex> mtLock(m_mtLock);
    if (m_mpStreams.find(strUrl) == m_mpStreams.end())
    {
        WARN("Stream not available");
    } else {
        m_mpStreams[strUrl].iCount--;
        if (m_mpStreams[strUrl].iCount == 0){
            m_mpStreams.erase(strUrl);
        }
    }
    INFO("Stream unsubscription successful");
}
