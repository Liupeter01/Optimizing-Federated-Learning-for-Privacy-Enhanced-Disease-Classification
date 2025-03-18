#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#ifndef _PREPROCESS_HPP_
#define _PREPROCESS_HPP_
#include <fileLoader.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <splitter.hpp>
#include <tbb/parallel_for.h>
namespace preprocess {
class NIHPreprocess {
public:
  explicit NIHPreprocess(const std::string &csv, const std::string &dir);

public:
  void processCSV();
  void loadOriginImages();
  void shuffleRandomByID();

  void write2NewTarget(const std::string &target);

protected:
  // Generate new directory
  void createNewDir(const std::string &target);
  [[nodiscard]] cv::Mat &imageNormalized(cv::Mat &origin);

public:
  // Constants for ResNet preprocessing
  static constexpr std::size_t IMAGE_SIZE = 224; // ResNet expects 224x224
  const cv::Scalar RESNET_MEAN = cv::Scalar(0.485f, 0.456f, 0.406f);
  const cv::Scalar RESNET_STD = cv::Scalar(0.229f, 0.224f, 0.225f);
  const cv::Mat meanMat = cv::Mat(1,1, CV_32FC3, RESNET_MEAN);
  const cv::Mat stdMat = cv::Mat(1,1, CV_32FC3, RESNET_STD);
  static constexpr std::size_t size = 224;

private:
  oneapi::tbb::affinity_partitioner ap;

  std::string origin_dir;
  std::string main_dir;
  std::unique_ptr<ImageLoader> loader;
  std::unique_ptr<DatasetSplitter> splitter;

  tbb::concurrent_vector<std::string> m_originImagesList;
  DatasetSplitter::GroupMap groupedMapping;
};

} // namespace preprocess

#endif //
