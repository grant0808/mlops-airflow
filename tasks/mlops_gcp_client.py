from google.cloud import storage

import os

class MLOpsGCPClient(object):
    def __init__(self, bucket_name) -> None:
        self.client = storage.Client()
        self.bucket_name = bucket_name
        self.bucket = self.client.bucket(bucket_name)
    
    def upload_model(self, model_name, local_dir_path):
        try:
            blob = self.bucket.blob(f"{model_name}")
            blob.upload_from_filename(f"{local_dir_path}/{model_name}")

            print(f"model is uploaded. {model_name}")
        except Exception  as e:
            print(f"Failed to upload: {e}")
    
    def download_model(self, blob_name, dest_file_path):
        try:
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(dest_file_path)
            
            print(f"model is downloaded. {dest_file_path}")
        except Exception as e:
            print(f"Failed to download: {e}")