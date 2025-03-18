#include <filesystem>
#include <memory>
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

            // std::cout << srcPath << "\n";

            if (std::find(m_originImagesList.begin(), m_originImagesList.end(),
                          patient.imageIndex) != m_originImagesList.end()) {

              std::string destPath = savePath + "/" + patient.imageIndex;

              std::filesystem::copy_file(
                  srcPath, destPath,
                  std::filesystem::copy_options::overwrite_existing);
            }
          }
        }
      });

    //   for (auto it = groupedMapping.begin(); it != groupedMapping.end(); ++it) {
    //     const std::string &splitName = it->first;
    //     const auto &records = it->second;
    
    //     // Create new subdirectory
    //     std::string savePath = main_dir + "/" + splitName;
    //     std::filesystem::create_directories(savePath);
    
    //     for (const auto &patient : records) {
    //         std::string srcPath = origin_dir + "/" + patient.imageIndex;
    
    //         // std::cout << srcPath << "\n";
    
    //         if (std::find(m_originImagesList.begin(), m_originImagesList.end(),
    //                       patient.imageIndex) != m_originImagesList.end()) {
    
    //             std::string destPath = savePath + "/" + patient.imageIndex;
    
    //             std::filesystem::copy_file(
    //                 srcPath, destPath,
    //                 std::filesystem::copy_options::overwrite_existing);
    //         }
    //     }
    // }
    
}
