#include <sstream>
#include <fstream>
#include <GSTStream.hpp>
#include <iostream>
#include "AIModel.hpp"
#include "include/model/JsonTypes.hpp"
void testYolov8() 
{
    std::ifstream fsJsonFile("/home/minhnh/Desktop/dev/assets/config_image_moderation.json");
    nlohmann::json jConfig = nlohmann::json::parse(fsJsonFile);
    AIModel *m_pYolov8 = new AIModel(jConfig, JN_MODEL_YOLOV8);
    cv::Mat img = cv::imread("/home/minhnh/Desktop/a2_data/ImageModerationVTV/flag2.jpg");
    
    std::cout << "aaaa" << std::endl;
    std::vector<stObject_t> stobjects; 
    m_pYolov8->run(img, stobjects);
    std::cout << "size: " << stobjects.size() << std::endl;
  

}


int main(int argc, char* argv[])
{
    testYolov8();
    // testPPOcrDet();
    // testPPOcrRec();
    // test_color();
    // test_pipeline();
    return 0;
}