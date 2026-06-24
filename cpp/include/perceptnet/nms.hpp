// PerceptNet C++ post-processing kernels: IoU and greedy NMS.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace perceptnet {

// Boxes are contiguous float arrays laid out as [x1, y1, x2, y2] * n.

float iou(const float* a, const float* b);

// Pairwise IoU: out is (n x m) row-major.
void iou_matrix(const float* a, std::size_t n, const float* b, std::size_t m, float* out);

// Greedy NMS. Returns kept indices in descending score order.
std::vector<int64_t> nms(const float* boxes, const float* scores, std::size_t n,
                         float iou_threshold, float score_threshold = -1.0f,
                         std::size_t max_detections = 0);

// Class-aware NMS: boxes of different labels never suppress each other.
std::vector<int64_t> batched_nms(const float* boxes, const float* scores, const int32_t* labels,
                                 std::size_t n, float iou_threshold, float score_threshold = -1.0f,
                                 std::size_t max_detections = 0);

}  // namespace perceptnet
