#include <fileLoader.hpp>
#include <filesystem>
#include <optional>

preprocess::ImageLoader::ImageLoader(const std::string &directory,
                                     const std::vector<std::string> &extensions)
    : imageDir(directory), validExtensions(extensions) {}

bool preprocess::ImageLoader::isValidExtension(const std::string &ext) {
  for (const auto &validExt : validExtensions) {
    if (ext == validExt)
      return true;
  }
  return false;
}

tbb::concurrent_vector<std::string> preprocess::ImageLoader::getImagePaths() {
  tbb::concurrent_vector<std::string> paths;

  for (const auto &entry : std::filesystem::directory_iterator(imageDir)) {
    std::string filePath = entry.path().string();
    std::string ext = entry.path().extension().string();

    // std::cout << "path = " << filePath << std::endl;

    if (isValidExtension(ext)) {
      paths.push_back(filePath);
    }
  }

  return paths;
}

std::optional<cv::Mat>
preprocess::ImageLoader::loadImage(const std::string &path) {

  auto res = cv::imread(path, cv::IMREAD_COLOR);
  if (res.empty()) {
    return std::nullopt;
  }
  return res;
}

bool preprocess::ImageLoader::writeImage(const cv::Mat &data,
                                         const std::string &path) {
  if (!cv::imwrite(path, data)) {
    std::cerr << "Error writing image: " << path << std::endl;
    return false;
  }
  return true;
}

tbb::concurrent_vector<cv::Mat> preprocess::ImageLoader::loadImages() {
  tbb::concurrent_vector<cv::Mat> images;

  for (const auto &entry : std::filesystem::directory_iterator(imageDir)) {
    std::string filePath = entry.path().string();
    std::string ext = entry.path().extension().string();

    if (isValidExtension(ext)) {
      cv::Mat img = cv::imread(filePath);

      if (!img.empty()) {
        images.push_back(img);
      } else {
      }
    }
  }

  return images;
}
