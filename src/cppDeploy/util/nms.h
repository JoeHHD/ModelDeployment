#ifndef NMS_H
#define NMS_H

#include<iostream>
#include<vector>
#include<algorithm>

namespace deploy::util{
struct Box{
	// (0, 0) 为图像的左上角。
	// x 坐标向右递增，y 坐标向下递增。
	// 边界框，中心点(x,y)，宽度w,高度h
	float x;
	float y;
	float w;
	float h;
};	
struct Detection{
	Box box;
	float confidence;
	int class_id;
};

class nms{
public:
	nms()=default;
	~nms()=default;
	float computeIOU(Box& box1, Box& box2);
	std::vector<Detection> nonMaxSuppression(
		const std::vector<Detection>& detections, 
		float iou_threshold
	);
	std::vector<Detection> confidenceFilter(Detection& detection);
};
} // namespace deploy::util

#endif //NMS_H




