#ifndef StreamManager_hpp
#define StreamManager_hpp

#include <memory>
#include <string>
#include <map>
#include <mutex>

#include "Logger.hpp"
#include "Stream.hpp"

typedef struct StreamValue{
    std::shared_ptr<Stream> pStream;
    int iCount =0;
}StreamValue_t;

class StreamManager
{
    public:
        static StreamManager& GetInstance();
        std::shared_ptr<Stream> getStream(std::string strUrl);
        void unregister(std::string stUrl);
        // void clean();

    private:
        std::map<std::string, StreamValue> m_mpStreams;

        std::mutex m_mtLock;

};

#endif // StreamManager_hpp