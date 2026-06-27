// pybind11 bindings exposing the C++ kernels as perceptnet.ops._perceptnet_cpp.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "perceptnet/nms.hpp"

#ifdef PERCEPTNET_WITH_OPENCV
#include <opencv2/core.hpp>

#include "perceptnet/preprocess.hpp"
#endif

namespace py = pybind11;
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

namespace {

void check_boxes(const FloatArray& boxes, const char* name) {
  if (boxes.ndim() != 2 || boxes.shape(1) != 4) {
    throw std::invalid_argument(std::string(name) + " must have shape (N, 4)");
  }
}

py::array_t<int64_t> to_numpy(const std::vector<int64_t>& v) {
  py::array_t<int64_t> out(static_cast<py::ssize_t>(v.size()));
  std::copy(v.begin(), v.end(), out.mutable_data());
  return out;
}

}  // namespace

PYBIND11_MODULE(_perceptnet_cpp, m) {
  m.doc() = "PerceptNet C++ kernels (NMS, IoU, OpenCV preprocessing)";

  m.def(
      "iou_matrix",
      [](FloatArray a, FloatArray b) {
        check_boxes(a, "a");
        check_boxes(b, "b");
        const auto n = static_cast<std::size_t>(a.shape(0));
        const auto mm = static_cast<std::size_t>(b.shape(0));
        py::array_t<float> out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(mm)});
        {
          py::gil_scoped_release release;
          perceptnet::iou_matrix(a.data(), n, b.data(), mm, out.mutable_data());
        }
        return out;
      },
      py::arg("a"), py::arg("b"));

  m.def(
      "nms",
      [](FloatArray boxes, FloatArray scores, float iou_threshold, float score_threshold,
         std::size_t max_detections) {
        check_boxes(boxes, "boxes");
        if (scores.ndim() != 1 || scores.shape(0) != boxes.shape(0)) {
          throw std::invalid_argument("scores must have shape (N,)");
        }
        std::vector<int64_t> keep;
        {
          py::gil_scoped_release release;
          keep = perceptnet::nms(boxes.data(), scores.data(), static_cast<std::size_t>(boxes.shape(0)),
                                 iou_threshold, score_threshold, max_detections);
        }
        return to_numpy(keep);
      },
      py::arg("boxes"), py::arg("scores"), py::arg("iou_threshold"),
      py::arg("score_threshold") = -1.0f, py::arg("max_detections") = 0);

  m.def(
      "batched_nms",
      [](FloatArray boxes, FloatArray scores, IntArray labels, float iou_threshold,
         float score_threshold, std::size_t max_detections) {
        check_boxes(boxes, "boxes");
        const auto n = boxes.shape(0);
        if (scores.ndim() != 1 || scores.shape(0) != n || labels.ndim() != 1 || labels.shape(0) != n) {
          throw std::invalid_argument("scores and labels must have shape (N,)");
        }
        std::vector<int64_t> keep;
        {
          py::gil_scoped_release release;
          keep = perceptnet::batched_nms(boxes.data(), scores.data(), labels.data(),
                                         static_cast<std::size_t>(n), iou_threshold,
                                         score_threshold, max_detections);
        }
        return to_numpy(keep);
      },
      py::arg("boxes"), py::arg("scores"), py::arg("labels"), py::arg("iou_threshold"),
      py::arg("score_threshold") = -1.0f, py::arg("max_detections") = 0);

#ifdef PERCEPTNET_WITH_OPENCV
  m.def(
      "preprocess_bgr",
      [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image, int out_h,
         int out_w, std::array<float, 3> mean, std::array<float, 3> stdev) {
        if (image.ndim() != 3 || image.shape(2) != 3) {
          throw std::invalid_argument("image must be HxWx3 uint8 (BGR)");
        }
        cv::Mat mat(static_cast<int>(image.shape(0)), static_cast<int>(image.shape(1)), CV_8UC3,
                    const_cast<uint8_t*>(image.data()));
        py::array_t<float> out({static_cast<py::ssize_t>(out_h), static_cast<py::ssize_t>(out_w),
                                static_cast<py::ssize_t>(3)});
        perceptnet::LetterboxInfo info;
        {
          py::gil_scoped_release release;
          info = perceptnet::preprocess_bgr(mat, out_h, out_w, mean, stdev, out.mutable_data());
        }
        return py::make_tuple(out, py::make_tuple(info.scale, info.pad_x, info.pad_y, info.orig_h,
                                                  info.orig_w));
      },
      py::arg("image"), py::arg("out_h"), py::arg("out_w"), py::arg("mean"), py::arg("std"));
  m.attr("has_opencv") = true;
#else
  m.attr("has_opencv") = false;
#endif
}
