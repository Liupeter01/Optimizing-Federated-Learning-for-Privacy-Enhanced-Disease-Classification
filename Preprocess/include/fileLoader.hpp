#pragma once
#include <opencv2/core/mat.hpp>
#ifndef _LOADER_HPP_
#define _LOADER_HPP_
#include <opencv2/opencv.hpp>
#include <optional>
#include <tbb/concurrent_vector.h>
#include <vector>

namespace preprocess {

class ImageLoader {
public:
  ImageLoader(const std::string &directory,
              const std::vector<std::string> &extensions = {".jpg", ".png",
                                                            ".bmp"});

public:
  std::optional<cv::Mat> loadImage(const std::string &path);
  bool writeImage(const cv::Mat &data, const std::string &path);

  [[nodiscard]] tbb::concurrent_vector<cv::Mat> loadImages();
  [[nodiscard]] tbb::concurrent_vector<std::string> getImagePaths();

protected:
  bool isValidExtension(const std::string &ext);

private:
  std::string imageDir;
  std::vector<std::string> validExtensions;
};
} // namespace preprocess

#endif
