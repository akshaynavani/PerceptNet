#include "perceptnet/nms.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace perceptnet {

namespace {

inline float area(const float* b) {
  return std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
}

std::vector<int64_t> sorted_candidates(const float* scores, std::size_t n, float score_threshold) {
  std::vector<int64_t> order;
  order.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (scores[i] > score_threshold) order.push_back(static_cast<int64_t>(i));
  }
  std::stable_sort(order.begin(), order.end(),
                   [scores](int64_t a, int64_t b) { return scores[a] > scores[b]; });
  return order;
}

}  // namespace

float iou(const float* a, const float* b) {
  const float ix1 = std::max(a[0], b[0]);
  const float iy1 = std::max(a[1], b[1]);
  const float ix2 = std::min(a[2], b[2]);
  const float iy2 = std::min(a[3], b[3]);
  const float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
  const float uni = area(a) + area(b) - inter;
  return uni > 1e-9f ? inter / uni : 0.0f;
}

void iou_matrix(const float* a, std::size_t n, const float* b, std::size_t m, float* out) {
  for (std::size_t i = 0; i < n; ++i) {
    const float* ai = a + 4 * i;
    float* row = out + i * m;
    for (std::size_t j = 0; j < m; ++j) row[j] = iou(ai, b + 4 * j);
  }
}

std::vector<int64_t> nms(const float* boxes, const float* scores, std::size_t n,
                         float iou_threshold, float score_threshold, std::size_t max_detections) {
  std::vector<int64_t> order = sorted_candidates(scores, n, score_threshold);
  std::vector<float> areas(n);
  for (std::size_t i = 0; i < n; ++i) areas[i] = area(boxes + 4 * i);

  std::vector<char> suppressed(n, 0);
  std::vector<int64_t> keep;
  keep.reserve(order.size());
  for (std::size_t oi = 0; oi < order.size(); ++oi) {
    const int64_t i = order[oi];
    if (suppressed[i]) continue;
    keep.push_back(i);
    if (max_detections && keep.size() >= max_detections) break;
    const float* bi = boxes + 4 * i;
    for (std::size_t oj = oi + 1; oj < order.size(); ++oj) {
      const int64_t j = order[oj];
      if (suppressed[j]) continue;
      const float* bj = boxes + 4 * j;
      const float ix1 = std::max(bi[0], bj[0]);
      const float iy1 = std::max(bi[1], bj[1]);
      const float ix2 = std::min(bi[2], bj[2]);
      const float iy2 = std::min(bi[3], bj[3]);
      const float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
      const float uni = areas[i] + areas[j] - inter;
      if (uni > 1e-9f && inter / uni > iou_threshold) suppressed[j] = 1;
    }
  }
  return keep;
}

std::vector<int64_t> batched_nms(const float* boxes, const float* scores, const int32_t* labels,
                                 std::size_t n, float iou_threshold, float score_threshold,
                                 std::size_t max_detections) {
  // Partition by label, run NMS per class, then merge by score.
  std::unordered_map<int32_t, std::vector<int64_t>> groups;
  for (std::size_t i = 0; i < n; ++i) groups[labels[i]].push_back(static_cast<int64_t>(i));

  std::vector<int64_t> keep;
  std::vector<float> sub_boxes;
  std::vector<float> sub_scores;
  for (auto& [label, idx] : groups) {
    (void)label;
    sub_boxes.resize(idx.size() * 4);
    sub_scores.resize(idx.size());
    for (std::size_t k = 0; k < idx.size(); ++k) {
      std::copy_n(boxes + 4 * idx[k], 4, sub_boxes.data() + 4 * k);
      sub_scores[k] = scores[idx[k]];
    }
    for (int64_t local : nms(sub_boxes.data(), sub_scores.data(), idx.size(), iou_threshold,
                             score_threshold, 0)) {
      keep.push_back(idx[local]);
    }
  }
  std::stable_sort(keep.begin(), keep.end(),
                   [scores](int64_t a, int64_t b) { return scores[a] > scores[b]; });
  if (max_detections && keep.size() > max_detections) keep.resize(max_detections);
  return keep;
}

}  // namespace perceptnet
