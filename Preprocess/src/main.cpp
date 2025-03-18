#include <chrono>
#include <iostream>
#include <preprocess.hpp>

int main() {

  using Clock = std::chrono::high_resolution_clock;
  auto start = Clock::now();

  // Convert start time to readable format
  auto start_time_t =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::cout << "Execution started at: "
            << std::put_time(std::localtime(&start_time_t), "%Y-%m-%d %H:%M:%S")
            << std::endl;

  preprocess::NIHPreprocess nih(CONFIG_HOME "Data_Entry_2017.csv",
                                CONFIG_HOME "origin-images");

  nih.processCSV();
  nih.loadOriginImages();
  nih.shuffleRandomByID();

  // generate new images dir and write them to new directory
  nih.write2NewTarget(CONFIG_HOME "split-images");

  auto end = Clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Total Execution Time: " << elapsed.count() << " seconds"
            << std::endl;
  return 0;
}
