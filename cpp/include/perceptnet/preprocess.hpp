// OpenCV frame preprocessing that mirrors perceptnet/data/preprocess.py exactly.
#pragma once

#include <array>

#include <opencv2/core.hpp>

namespace perceptnet {

struct LetterboxInfo {
  float scale = 1.0f;
  float pad_x = 0.0f;
  float pad_y = 0.0f;
  int orig_h = 0;
  int orig_w = 0;
};

constexpr int kPadValue = 114;

// Aspect-preserving resize + centered constant padding to (out_h, out_w). Returns CV_8UC3.
cv::Mat letterbox(const cv::Mat& image, int out_h, int out_w, LetterboxInfo* info);

// Letterbox + BGR->RGB + scale to [0,1] + (x - mean) / std.
// Writes an HxWx3 float32 NHWC tensor into `out` (must hold out_h*out_w*3 floats).
LetterboxInfo preprocess_bgr(const cv::Mat& bgr, int out_h, int out_w,
                             const std::array<float, 3>& mean, const std::array<float, 3>& stdev,
                             float* out);

}  // namespace perceptnet
