import sagemaker
from sagemaker.tensorflow import TensorFlow
import os
import time
from datetime import datetime
import boto3

# Get IAM role from environment variable
sagemaker_role = os.environ.get('SAGEMAKER_ROLE_ARN')
if not sagemaker_role:
    raise ValueError("SAGEMAKER_ROLE_ARN environment variable not set")

# Validate S3 data exists
def validate_s3_path(bucket, prefix):
    s3_client = boto3.client('s3')
    try:
        s3_client.head_object(Bucket=bucket, Key=prefix.strip('/'))
        return True
    except:
        return False

# Configure hyperparameters from environment or config file
hyperparameters = {
    'epochs': int(os.environ.get('TRAINING_EPOCHS', 10)),
    'batch-size': int(os.environ.get('BATCH_SIZE', 32)),
    'learning-rate': float(os.environ.get('LEARNING_RATE', 0.001)),
}

# Better job name
job_name = f'tensorflow-training-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

def cleanup_failed_jobs():
    sagemaker_client = boto3.client('sagemaker')
    response = sagemaker_client.list_training_jobs(
        StatusEquals='InProgress',
        MaxResults=10
    )
    for job in response['TrainingJobSummaries']:
        print(f"Stopping job: {job['TrainingJobName']}")
        try:
            sagemaker_client.stop_training_job(
                TrainingJobName=job['TrainingJobName']
            )
        except Exception as e:
            print(f"Error stopping job: {e}")

def train_on_sagemaker(bucket_name):
    print("Starting SageMaker training process...")
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Use existing S3 data
    training_data_s3 = f's3://{bucket_name}/training-data/'
    print(f"Using training data from: {training_data_s3}")
    
    # Generate unique job name
    print(f"Creating new training job: {job_name}")
    
    # Configure the training job
    tensorflow_estimator = TensorFlow(
        entry_point='train_model.py',
        source_dir='./',  # Include all local files
        role=sagemaker_role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        framework_version='2.11.0',
        py_version='py39',
        hyperparameters=hyperparameters,
        job_name=job_name,
        output_path=f's3://{bucket_name}/model-output'  # Add explicit output path
    )
    
    print("\nStarting training job...")
    print(f"Training with instance type: {tensorflow_estimator.instance_type}")
    
    try:
        # Start training with debug info
        print("Submitting training job to SageMaker...")
        tensorflow_estimator.fit(
            {'training': training_data_s3}, 
            wait=True,
            logs='All'  # Show all logs
        )
        print("\nTraining complete!")
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nChecking CloudWatch logs...")
        try:
            logs = sagemaker_session.logs_for_job(job_name)
            print(logs)
        except Exception as log_e:
            print(f"Error getting logs: {log_e}")
        raise
    
    return tensorflow_estimator

if __name__ == '__main__':
    bucket_name = "imagedatasetforgeoguessrai"
    estimator = train_on_sagemaker(bucket_name)