// Standalone micro-benchmarks for the C++ kernels (no Python involved).
//
//   ./build/bench_ops [iterations]
//
// Prints a table of median latencies so they can be compared against the NumPy / JAX numbers
// produced by `perceptnet bench ops`.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "perceptnet/nms.hpp"

#ifdef PERCEPTNET_WITH_OPENCV
#include <opencv2/core.hpp>

#include "perceptnet/preprocess.hpp"
#endif

namespace {

using Clock = std::chrono::steady_clock;

double median_us(const std::function<void()>& fn, int iters) {
  fn();  // warmup
  std::vector<double> times;
  times.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    const auto t0 = Clock::now();
    fn();
    const auto t1 = Clock::now();
    times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  std::nth_element(times.begin(), times.begin() + times.size() / 2, times.end());
  return times[times.size() / 2];
}

// Boxes clustered around a few centers so NMS has real work to do.
void random_boxes(std::size_t n, std::mt19937& rng, std::vector<float>* boxes,
                  std::vector<float>* scores, std::vector<int32_t>* labels) {
  std::uniform_real_distribution<float> center(0.0f, 1000.0f);
  std::normal_distribution<float> jitter(0.0f, 8.0f);
  std::uniform_real_distribution<float> size(20.0f, 120.0f);
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  std::uniform_int_distribution<int32_t> label(0, 2);
  const std::size_t clusters = std::max<std::size_t>(1, n / 20);
  std::vector<float> cx(clusters), cy(clusters), w(clusters), h(clusters);
  for (std::size_t c = 0; c < clusters; ++c) {
    cx[c] = center(rng);
    cy[c] = center(rng);
    w[c] = size(rng);
    h[c] = size(rng);
  }
  boxes->resize(4 * n);
  scores->resize(n);
  labels->resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = i % clusters;
    const float x = cx[c] + jitter(rng), y = cy[c] + jitter(rng);
    (*boxes)[4 * i + 0] = x - w[c] / 2;
    (*boxes)[4 * i + 1] = y - h[c] / 2;
    (*boxes)[4 * i + 2] = x + w[c] / 2;
    (*boxes)[4 * i + 3] = y + h[c] / 2;
    (*scores)[i] = unit(rng);
    (*labels)[i] = label(rng);
  }
}

void row(const std::string& op, const std::string& shape, double us) {
  std::printf("%-22s %-20s %12.2f us\n", op.c_str(), shape.c_str(), us);
}

}  // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 50;
  std::mt19937 rng(0);
  std::printf("%-22s %-20s %15s\n", "op", "shape", "median");

  for (std::size_t n : {100, 1000, 5000}) {
    std::vector<float> boxes, scores;
    std::vector<int32_t> labels;
    random_boxes(n, rng, &boxes, &scores, &labels);
    const std::string shape = "N=" + std::to_string(n);
    row("nms", shape, median_us([&] { perceptnet::nms(boxes.data(), scores.data(), n, 0.5f); }, iters));
    row("batched_nms", shape, median_us([&] {
          perceptnet::batched_nms(boxes.data(), scores.data(), labels.data(), n, 0.5f);
        }, iters));
    if (n <= 1000) {
      std::vector<float> out(n * n);
      row("iou_matrix", shape + "x" + std::to_string(n),
          median_us([&] { perceptnet::iou_matrix(boxes.data(), n, boxes.data(), n, out.data()); }, iters));
    }
  }

#ifdef PERCEPTNET_WITH_OPENCV
  cv::Mat frame(1080, 1920, CV_8UC3);
  cv::randu(frame, cv::Scalar::all(0), cv::Scalar::all(255));
  const std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
  const std::array<float, 3> stdev{0.229f, 0.224f, 0.225f};
  for (auto [h, w] : std::vector<std::pair<int, int>>{{384, 640}, {384, 1248}, {640, 640}}) {
    std::vector<float> out(static_cast<std::size_t>(h) * w * 3);
    const std::string shape = "1080p->" + std::to_string(h) + "x" + std::to_string(w);
    row("letterbox", shape, median_us([&] {
          perceptnet::LetterboxInfo info;
          perceptnet::letterbox(frame, h, w, &info);
        }, iters));
    row("preprocess_bgr", shape,
        median_us([&] { perceptnet::preprocess_bgr(frame, h, w, mean, stdev, out.data()); }, iters));
  }
#endif
  return 0;
}
