from ml_split import split
from .ml_preprocess import preprocess_image
#import .ml_modeltraining

def resnet18(image_dir, csv_path, output_dir,  df_train_path):
    split(csv_path)
    preprocess_image(image_dir, csv_path, output_dir, df_train_path)
    #run_all()