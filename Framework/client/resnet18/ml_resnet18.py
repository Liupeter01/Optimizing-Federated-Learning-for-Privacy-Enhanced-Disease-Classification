from ml_split import split
from .ml_preprocess import preprocess_image
from .ml_modeltraining import run_all

# This part only needs to be run once
def initialize(image_dir, csv_path, output_dir,  df_train_path):
    split(csv_path)
    preprocess_image(image_dir, csv_path, output_dir, df_train_path)

def resnet18(image_dir, csv_path, output_dir,  df_train_path, model_path):
    run_all(image_dir, csv_path, output_dir, df_train_path, model_path )