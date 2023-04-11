import boto3

# Create an S3 client
s3_client = boto3.client('s3')

# Specify the name of the S3 bucket and the key of the file to be read
bucket_name = 'yadixawsbucket'
key_name = 'README.md'

# Use the get_object method to retrieve the S3 object
response = s3_client.get_object(Bucket=bucket_name, Key=key_name)

# Get the file content from the response
file_content = response['Body'].read().decode('utf-8')

# Print the file content
print(file_content)
