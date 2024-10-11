import os
import boto3
import torch
import shutil
from dotenv import load_dotenv
load_dotenv()

model_dir = '/code/files/vbc'
num_folds = 8

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

if not aws_access_key_id or not aws_secret_access_key:
    raise EnvironmentError("AWS credentials not found in environment variables.")

# Safely remove the directory if it exists
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
os.makedirs(model_dir, exist_ok=True)

s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# Download the model files from S3
for i in range(num_folds):
    model_path = os.path.join(model_dir, f'model_fold_{i}.pt')
    bucket_name = 'ml-models-kino'
    file_key = f'vbc/model_fold_{i}.pt'
    
    try:
        print(f"Downloading {file_key} to {model_path}")
        s3.download_file(bucket_name, file_key, model_path)
    except Exception as e:
        print(f"Error downloading {file_key}: {e}")

# Download the .keras model file from S3
measurements_model_dir = '/code/files/measurements'
model_path = os.path.join(measurements_model_dir, 'model.keras')

# Safely remove the directory if it exists
if os.path.exists(measurements_model_dir):
    shutil.rmtree(measurements_model_dir)
os.makedirs(measurements_model_dir, exist_ok=True)

file_key = 'contralateral-measurements/model.keras'

try:
    print(f"Downloading {file_key} to {model_path}")
    s3.download_file(bucket_name, file_key, model_path)
except Exception as e:
    print(f"Error downloading {file_key}: {e}")

file_key = 'contralateral-measurements/model.weights.h5'
weights_path = os.path.join(measurements_model_dir, 'model.weights.h5')
try:
    print(f"Downloading {file_key} to {weights_path}")
    s3.download_file(bucket_name, file_key, weights_path)
except Exception as e:
    print(f"Error downloading {file_key}: {e}")