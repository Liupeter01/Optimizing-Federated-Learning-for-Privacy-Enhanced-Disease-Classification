#include <fstream>
#include <iostream>
#include <random>
#include <splitter.hpp>
#include <sstream>
#include <string>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

preprocess::DatasetSplitter::DatasetSplitter(const std::string &csv)
    : m_csv(csv) {

  /*start loading csv*/
  m_rawData = loadCSV(m_csv);
}

tbb::concurrent_vector<preprocess::RawPatientData>
preprocess::DatasetSplitter::loadCSV(const std::string &csvFile) {
  tbb::concurrent_vector<RawPatientData> patients;

  std::ifstream file(csvFile);
  if (!file.is_open()) {
    std::cout << "error\n";
    return {};
  }

  std::vector<std::string> lines;

  // Read file into a string, then split into lines
  std::string fileContent((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
  file.close();

  // Ensure all line endings are normalized to `\n`
  fileContent.erase(std::remove(fileContent.begin(), fileContent.end(), '\r'),
                    fileContent.end());

  std::stringstream ss(fileContent);
  std::string line;

  // Skip header
  std::getline(ss, line);

  while (std::getline(ss, line)) {
    lines.push_back(line);
  }

  std::cout << "Total lines read: " << lines.size() << std::endl;

  // Parallel processing of the CSV lines
  tbb::parallel_for(tbb::blocked_range<size_t>(0, lines.size()),
                    [&](const tbb::blocked_range<size_t> &range) {
                      for (size_t i = range.begin(); i < range.end(); ++i) {
                        std::stringstream ss(lines[i]);
                        RawPatientData patient;

                        unsigned int followup = 0, patientid = 0, age = 0,
                                     width = 0, height = 0;
                        float pixelX = 0.0f, pixelY = 0.0f;
                        std::string imageIndex, findingLabels, genderStr,
                            viewPosition;

                        // Read string values safely
                        std::getline(ss, imageIndex, ',');
                        std::getline(ss, findingLabels, ',');

                        // Read numeric values safely
                        if (!(ss >> followup))
                          followup = 0;
                        if (ss.peek() == ',')
                          ss.ignore();
                        if (!(ss >> patientid))
                          patientid = 0;
                        if (ss.peek() == ',')
                          ss.ignore();
                        if (!(ss >> age))
                          age = 0;
                        if (ss.peek() == ',')
                          ss.ignore();

                        std::getline(ss, genderStr, ',');
                        std::getline(ss, viewPosition, ',');

                        if (!(ss >> width))
                          width = 0;
                        if (ss.peek() == ',')
                          ss.ignore();
                        if (!(ss >> height))
                          height = 0;
                        if (ss.peek() == ',')
                          ss.ignore();
                        if (!(ss >> pixelX))
                          pixelX = 0.0f;
                        if (ss.peek() == ',')
                          ss.ignore();
                        if (!(ss >> pixelY))
                          pixelY = 0.0f;

                        // Assign values to struct
                        patient.imageIndex = imageIndex;
                        patient.findingLabels = findingLabels;
                        patient.followUp = std::to_string(followup);
                        patient.patientID = std::to_string(patientid);
                        patient.age = age;
                        patient.gender = RawPatientData::parseGender(genderStr);
                        patient.viewPosition = viewPosition;
                        patient.width = width;
                        patient.height = height;
                        patient.pixelSpacingX = pixelX;
                        patient.pixelSpacingY = pixelY;

                        // Push into concurrent vector safely
                        patients.push_back(std::move(patient));
                      }
                    });

  std::cout << "Total " << patients.size() << " patients Loaded To Mem"
            << std::endl;
  return patients;
}

void preprocess::DatasetSplitter::groupByPatientID() {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, m_rawData.size()),
                    [&](const tbb::blocked_range<size_t> &range) {
                      for (size_t i = range.begin(); i < range.end(); ++i) {
                        auto id = m_rawData[i].patientID;

                        // record those id for shuffle
                        patientIDs.push_back(id);

                        // Move data to categoryGroups
                        categoryGroups[id].push_back(std::move(m_rawData[i]));
                      }
                    });

  std::cout << "Grouped " << categoryGroups.size() << " unique patients."
            << std::endl;
}

void preprocess::DatasetSplitter::randomShufflePatientID() {
  // Shuffle for randomness
  std::random_device rd;
  std::mt19937 rng(rd());
  std::shuffle(patientIDs.begin(), patientIDs.end(), rng);
}
