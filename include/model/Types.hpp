#ifndef Types_hpp
#define Types_hpp

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv4/opencv2/opencv.hpp>

/**
 * Enumeration of supported frameworks.
 */
typedef enum emFramework
{
    NCNN = 0, // Run AI with NCNN
    TRT = 1, // Run AI with TensorRT
    ORT = 2, // Run AI with ONNX Runtime
    AIC = 3
} emFramework_t;

/**
 * @brief structure of ...
*/
typedef struct stObject
{
    cv::Rect2f rfBox;
    float fScore;
    int iId=-1;
    std::string strLabel;
} stObject_t;



typedef enum emZoneType
{
    Detection=0, 
    In=1, 
    Out=2, 
    Intrusion=3
} emZoneType_t;


typedef struct stTextBox
{
    std::vector<std::vector<std::vector<int>>> textBoxes;
    cv::Mat inputCPUBGR;
    void sortPolygons(){
        std::sort(textBoxes.begin(), textBoxes.end(), [](const std::vector<std::vector<int>>& poly1, const std::vector<std::vector<int>>& poly2) {
            // Calculate the average y-coordinate of each polygon
            auto avgY1 = std::accumulate(poly1.begin(), poly1.end(), 0.0, [](double sum, const std::vector<int>& point) {
                return sum + point[1]; // Accumulate y-coordinates
            }) / poly1.size();

            auto avgY2 = std::accumulate(poly2.begin(), poly2.end(), 0.0, [](double sum, const std::vector<int>& point) {
                return sum + point[1]; // Accumulate y-coordinates
            }) / poly2.size();

            return avgY1 < avgY2; // Sort by ascending order of average y-coordinate
        });
    }
} stTextBox_t;

typedef struct stTextRec
{
    std::string text;
    float fScore;
} stTextRec_t;

typedef struct stResnetCls {
    std::string label;
    float fScore;
} stResnetCls_t;


#endif // Types_hpp