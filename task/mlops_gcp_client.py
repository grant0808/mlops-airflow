from google.cloud.storage import Client
from google.oauth2.service_account import Credentials

import os

class MLOpsGCPClient(object):
    def __init__(self, GCP_KEY_FILE) -> None:
        Credentials = Credentials.from_service_account_info(GCP_KEY_FILE)
        self.client = Client(Credentials=Credentials, project=Credentials.project_id)
    
    def upload_model(self,bucket_name, model_name, local_dir_path):
        try:
            bucket = self.client.get_bucket(bucket_name)
            file_names = [file for file in os.listdir(local_dir_path)]
            for file_name in file_names:
                blob = bucket.blob(f"{model_name}/{file_name}")
                blob.upload_from_filename(f"{local_dir_path}/{file_name}")

                print(f"model is uploaded. {blob.public_url}")
        except Exception  as e:
            print(f"Failed to upload: {e}")
    
    def download_model(self, bucket_name, blob_name, dest_file_path):
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(dest_file_path)
            
            print(f"model is downloaded. {dest_file_path}")
        except Exception as e:
            print(f"Failed to download: {e}")