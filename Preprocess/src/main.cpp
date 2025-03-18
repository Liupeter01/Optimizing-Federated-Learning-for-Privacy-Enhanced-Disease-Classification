#include <iostream>
#include <preprocess.hpp>

int main() {
  preprocess::NIHPreprocess nih(CONFIG_HOME "Data_Entry_2017.csv",
                                CONFIG_HOME "origin-images");

  nih.processCSV();
  nih.loadOriginImages();
  nih.shuffleRandomByID();

  // generate new images
  nih.createNewDir(CONFIG_HOME "split-images");

  return 0;
}
