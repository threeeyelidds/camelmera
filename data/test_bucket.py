import boto3

# Create an S3 client
s3_client = boto3.client('s3')

# Specify the name of the S3 bucket and the file to be uploaded
bucket_name = 'yadixawsbucket'
file_path = './README.md'  # Local file path
key_name = 'README.md'  # Name to be used for the file in the S3 bucket

# Upload the file to the S3 bucket
s3_client.upload_file(file_path, bucket_name, key_name)

print(f'Successfully uploaded {file_path} to {bucket_name}/{key_name}')
