#include "perceptnet/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace perceptnet {

cv::Mat letterbox(const cv::Mat& image, int out_h, int out_w, LetterboxInfo* info) {
  if (image.empty() || image.type() != CV_8UC3) {
    throw std::invalid_argument("letterbox expects a non-empty CV_8UC3 image");
  }
  const int h = image.rows;
  const int w = image.cols;
  const float scale = std::min(static_cast<float>(out_h) / h, static_cast<float>(out_w) / w);
  const int new_h = std::max(1, static_cast<int>(std::lround(h * scale)));
  const int new_w = std::max(1, static_cast<int>(std::lround(w * scale)));

  cv::Mat resized;
  if (new_h != h || new_w != w) {
    const int interp = scale < 1.0f ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, interp);
  } else {
    resized = image;
  }
  const int top = (out_h - new_h) / 2;
  const int left = (out_w - new_w) / 2;
  const int bottom = out_h - new_h - top;
  const int right = out_w - new_w - left;
  cv::Mat padded;
  cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT,
                     cv::Scalar(kPadValue, kPadValue, kPadValue));
  if (info) {
    info->scale = scale;
    info->pad_x = static_cast<float>(left);
    info->pad_y = static_cast<float>(top);
    info->orig_h = h;
    info->orig_w = w;
  }
  return padded;
}

LetterboxInfo preprocess_bgr(const cv::Mat& bgr, int out_h, int out_w,
                             const std::array<float, 3>& mean, const std::array<float, 3>& stdev,
                             float* out) {
  LetterboxInfo info;
  cv::Mat boxed = letterbox(bgr, out_h, out_w, &info);
  // Fused BGR->RGB swap + normalization in a single pass over the pixels.
  const float inv255 = 1.0f / 255.0f;
  const std::array<float, 3> inv_std = {1.0f / stdev[0], 1.0f / stdev[1], 1.0f / stdev[2]};
  for (int y = 0; y < out_h; ++y) {
    const uint8_t* row = boxed.ptr<uint8_t>(y);
    float* dst = out + static_cast<std::size_t>(y) * out_w * 3;
    for (int x = 0; x < out_w; ++x) {
      const uint8_t* px = row + 3 * x;
      for (int c = 0; c < 3; ++c) {
        // output channel c (RGB) reads input channel 2 - c (BGR)
        dst[3 * x + c] = (px[2 - c] * inv255 - mean[c]) * inv_std[c];
      }
    }
  }
  return info;
}

}  // namespace perceptnet
