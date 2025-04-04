#pragma once
#include <valarray>
#ifndef _SPLITTER_HPP_
#define _SPLITTER_HPP_
#include <string>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <unordered_map>
namespace preprocess {

enum class Gender { Male, Female, Unknown };

struct RawPatientData {
  std::string imageIndex;    // 图像文件名
  std::string findingLabels; // 诊断标签
  std::string followUp;      // 随访编号
  std::string patientID;     // 患者 ID
  unsigned int age;          // 年龄
  Gender gender;             // 性别
  std::string viewPosition;  // 视角
  unsigned int width;        // 原始图像宽度
  unsigned int height;       // 原始图像高度
  float pixelSpacingX;       // 像素间距 X
  float pixelSpacingY;       // 像素间距 Y

  static Gender parseGender(const std::string &genderStr) {
    if (genderStr == "M")
      return Gender::Male;
    if (genderStr == "F")
      return Gender::Female;
    return Gender::Unknown;
  }

  static std::string genderToString(Gender gender) {
    switch (gender) {
    case Gender::Male:
      return "M";
    case Gender::Female:
      return "F";
    default:
      return "Unknown";
    }
  }
};

class DatasetSplitter {
public:
  using GroupMap = std::unordered_map<
      /*dir_name =*/std::string,
      /*all data in one single group =*/std::vector<RawPatientData>>;

public:
  DatasetSplitter(const std::string &csv);

public:
  tbb::concurrent_vector<RawPatientData> loadCSV(const std::string &csvFile);
  void groupByPatientID();

  [[nodiscard]] GroupMap assignPatients2Group();

protected:
  void randomShufflePatientID();

private:
  tbb::concurrent_unordered_map<std::string, int> uniquePatientIDs;

  /*Raw data from CSV file*/
  std::string m_csv;

  // raw data is going to be removed after grouped!!!
  tbb::concurrent_vector<preprocess::RawPatientData> m_rawData;

  /*group by patientid*/
  tbb::concurrent_vector<std::string> patientIDs;
  tbb::concurrent_unordered_map<
      /*paitient id = */ std::string,
      /*image name = */ tbb::concurrent_vector<RawPatientData>>
      categoryGroups;
};
} // namespace preprocess

#endif
