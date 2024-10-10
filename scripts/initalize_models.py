import os
import boto3
import torch
import shutil

model_dir = '/code/files/vbc'
num_folds = 8

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

if not aws_access_key_id or not aws_secret_access_key:
    raise EnvironmentError("AWS credentials not found in environment variables.")

if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

os.makedirs(model_dir, exist_ok=True)

# Download the model files from S3
for i in range(num_folds):
    model_path = os.path.join(model_dir, f'model_fold_{i}.pt')
    
    bucket_name = 'ml-models-kino'
    file_key = f'vbc/model_fold_{i}.pt'
    local_file_name = model_path

    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    s3.download_file(bucket_name, file_key, local_file_name)

# Download the .keras model file from S3
model_dir = '/code/files/measurements'
model_path = os.path.join(model_dir, 'model.keras')

if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

os.makedirs(model_dir, exist_ok=True)

bucket_name = 'ml-models-kino'
file_key = 'contralateral-measurements/model.keras'
local_file_name = model_path

s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)
s3.download_file(bucket_name, file_key, local_file_name)
