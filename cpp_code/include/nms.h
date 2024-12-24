#ifndef NMS_H
#define NMS_H

#include <iostream>
#include <vector>
#include <algorithm>

namespace deploy::util {

struct Box {
    float x;
    float y;
    float w;
    float h;
};

struct Detection {
    Box box;
    float confidence;
    int class_id;
};

class nms {
public:
    nms() = default;
    ~nms() = default;

    float computeIOU(Box& box1, Box& box2);

    std::vector<Detection> nonMaxSuppression(
        const std::vector<Detection>& detections, 
        float iou_threshold);

    std::vector<Detection> confidenceFilter(
        const std::vector<Detection>& detections, 
        float threshold);
};

} // namespace deploy::util

#endif // NMS_H
