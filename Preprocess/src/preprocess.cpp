#include <filesystem>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <preprocess.hpp>
#include <tbb/parallel_for.h>

preprocess::NIHPreprocess::NIHPreprocess(const std::string &csv,
                                         const std::string &dir)
    : origin_dir(dir), loader(std::make_unique<ImageLoader>(dir)),
      splitter(std::make_unique<DatasetSplitter>(csv)) {}

void preprocess::NIHPreprocess::processCSV() { splitter->groupByPatientID(); }

void preprocess::NIHPreprocess::loadOriginImages() {
  m_originImagesList = std::move(loader->getImagePaths());
  std::cout << "Total " << m_originImagesList.size()
            << " valid images are identified from image path" << std::endl;
}

void preprocess::NIHPreprocess::shuffleRandomByID() {
  groupedMapping = std::move(splitter->assignPatients2Group());
}

void preprocess::NIHPreprocess::createNewDir(const std::string &target) {
  main_dir = target;
  std::filesystem::create_directories(target);
}

cv::Mat &preprocess::NIHPreprocess::imageNormalized(cv::Mat &origin) {

  cv::Mat resized, floatImg, normalized, displayable;

  // Resize to match ResNet input size
  cv::resize(origin, resized, cv::Size(IMAGE_SIZE, IMAGE_SIZE));

   // Convert to floating-point representation
   resized.convertTo(floatImg, CV_32FC3);

   cv::Mat meanImg, stdImg;
   cv::resize(meanMat, meanImg, floatImg.size());
   cv::resize(stdMat, stdImg, floatImg.size());

   // Normalize the image using (image - mean) / std
   cv::subtract(floatImg, meanImg, normalized);
   cv::divide(normalized, stdImg, normalized);

 origin = normalized;

  return origin;
}

void preprocess::NIHPreprocess::write2NewTarget(const std::string &target) {

  createNewDir(target);

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, groupedMapping.size()),
      [&](const tbb::blocked_range<size_t> &range) {
        for (size_t i = range.begin(); i < range.end(); ++i) {

          auto it = std::next(groupedMapping.begin(), i);
          const std::string &splitName = it->first;
          const auto &records = it->second;

          // create new sub dir
          std::string savePath = main_dir + "/" + splitName;
          std::filesystem::create_directories(savePath);

          for (const auto &patient : records) {
            std::string srcPath = origin_dir + "/" + patient.imageIndex;
            std::string destPath = savePath + "/" + patient.imageIndex;

            cv::Mat origin{};

            if (auto oriOpt = loader->loadImage(srcPath); oriOpt) {
              origin = oriOpt.value();
              origin = this->imageNormalized(oriOpt.value());
            }

            if (origin.empty()) {
              continue;
            }

            loader->writeImage(origin, destPath);
          }
        }
      },
      ap);
}
