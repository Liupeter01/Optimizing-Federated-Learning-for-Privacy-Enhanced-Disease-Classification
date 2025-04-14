#from ml_split import split
#from .ml_preprocess import preprocess_image
from .ml_modeltraining import run_all, run_all_with_dp

# This part only needs to be run once
def resnet18(csv_path, output_dir,  df_train_path, model_path, config = None):
    #run_all(image_dir, csv_path, output_dir, df_train_path, model_path)
    return run_all_with_dp(csv_path, output_dir, df_train_path, model_path, config)
