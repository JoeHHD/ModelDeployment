#include "nms.h"

namespace deploy::util {

// 定义类 nms 的成员函数
float nms::computeIOU(Box& box1, Box& box2) {
    float x1 = std::max(box1.x - box1.w / 2, box2.x - box2.w / 2);
    float y1 = std::max(box1.y - box1.h / 2, box2.y - box2.h / 2);
    float x2 = std::min(box1.x + box1.w / 2, box2.x + box2.w / 2);
    float y2 = std::min(box1.y + box1.h / 2, box2.y + box2.h / 2);
    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float union_area = box1.w * box1.h + box2.w * box2.h - intersection;
    return intersection / union_area;
}

std::vector<Detection> nms::nonMaxSuppression(
        const std::vector<Detection>& detections, 
        float iou_threshold) {
    std::vector<Detection> result;
    std::vector<Detection> sorted_detections = detections;
    std::sort(
        sorted_detections.begin(), 
        sorted_detections.end(), 
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        }
    );

    while (!sorted_detections.empty()) {
        Detection best = sorted_detections.front();
        result.push_back(best);
        sorted_detections.erase(sorted_detections.begin());

        for (auto it = sorted_detections.begin(); it != sorted_detections.end();) {
            if (computeIOU(best.box, it->box) > iou_threshold) {
                it = sorted_detections.erase(it);
            } else {
                ++it;
            }
        }
    }
    return result;
}

std::vector<Detection> nms::confidenceFilter(
        const std::vector<Detection>& detections, 
        float threshold){     
    // 临时变量，可能有问题
    std::vector<Detection> filtered_detection;
    for (auto& box : filtered_detection) {
        if (box.confidence > threshold) {
            filtered_detection.push_back(box);
        }
    }
    return filtered_detection;
}

} // namespace deploy::util
