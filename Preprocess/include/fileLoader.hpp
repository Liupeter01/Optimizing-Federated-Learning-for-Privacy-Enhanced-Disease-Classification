#pragma once
#ifndef _LOADER_HPP_
#define _LOADER_HPP_
#include <opencv2/opencv.hpp>
#include <tbb/concurrent_vector.h>
#include <vector>

namespace preprocess {

class ImageLoader {
public:
  ImageLoader(const std::string &directory,
              const std::vector<std::string> &extensions = {".jpg", ".png",
                                                            ".bmp"});

public:
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
