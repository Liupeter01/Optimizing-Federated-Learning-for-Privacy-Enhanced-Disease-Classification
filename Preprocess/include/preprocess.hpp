#pragma once
#include <opencv2/core/mat.hpp>
#ifndef _PREPROCESS_HPP_
#define _PREPROCESS_HPP_
#include <fileLoader.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <splitter.hpp>

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

private:
  std::string origin_dir;
  std::string main_dir;
  std::unique_ptr<ImageLoader> loader;
  std::unique_ptr<DatasetSplitter> splitter;

  tbb::concurrent_vector<std::string> m_originImagesList;
  DatasetSplitter::GroupMap groupedMapping;
};

} // namespace preprocess

#endif //
