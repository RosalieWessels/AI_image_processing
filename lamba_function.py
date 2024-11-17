import boto3
import botocore
import os
import logging
import io
from PIL import Image
import uuid

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    # Initialize the Rekognition and S3 clients
    rekognition_client = boto3.client('rekognition')
    s3_client = boto3.client('s3')

    # Collection name
    collection_id = 'scu_faces_collection'

    def create_collection_if_not_exists(collection_id):
        try:
            # Check if the collection already exists
            response = rekognition_client.describe_collection(CollectionId=collection_id)
            logger.info(f"Collection '{collection_id}' already exists.")
        except rekognition_client.exceptions.ResourceNotFoundException:
            # If the collection does not exist, create it
            response = rekognition_client.create_collection(CollectionId=collection_id)
            logger.info(f"Collection '{collection_id}' created successfully.")
        except botocore.exceptions.ClientError as error:
            logger.error(f"An error occurred: {error}")

    def upload_new_face_to_aws(event, image_byte_arr, bucket_name):
        # Define the destination bucket and key
        logger.info("uploading new face to aws with id")
        random_id = str(uuid.uuid4())
        destination_key = 'faces/' + id
        
        # Upload the image to the destination bucket
        s3_client.put_object(Bucket=bucket_name, Key=destination_key, Body=image_byte_arr, ContentType='image/jpeg')
        s3_client.put_object_tagging(
                Bucket=bucket_name,
                Key=destination_key,
                Tagging={
                    'TagSet': [
                        {
                            'Key': 'id',
                            'Value': random_id
                        }
                    ]
                }
            )

    def run_model_on_image(event):
        try:
            # Get bucket name and object key from the event triggered by S3
            bucket_name = event['Records'][0]['s3']['bucket']['name']
            object_key = event['Records'][0]['s3']['object']['key']
            logger.info(f"Processing image from bucket: {bucket_name}, key: {object_key}")
            
            # Download the image from S3
            s3_object = s3_client.get_object(Bucket=bucket_name, Key=object_key)
            image_data = s3_object['Body'].read()
            group_image = Image.open(io.BytesIO(image_data))
            
            # Call Rekognition to detect faces in the image
            response = rekognition_client.detect_faces(
                Image={
                    'S3Object': {
                        'Bucket': bucket_name,
                        'Name': object_key,
                    }
                },
                Attributes=['ALL']
            )
            
            # Log the number of faces found
            face_count = len(response['FaceDetails'])
            logger.info(f"Number of faces detected: {face_count}")

            # Get bounding boxes of each detected face
            face_bounding_boxes = response['FaceDetails']

            # Iterate over each detected face and crop it for comparison
            for i, face in enumerate(face_bounding_boxes):
                # Bounding box coordinates
                box = face['BoundingBox']
                width, height = group_image.size
                left = int(box['Left'] * width)
                top = int(box['Top'] * height)
                right = int(left + (box['Width'] * width))
                bottom = int(top + (box['Height'] * height))

                # Crop the face area
                cropped_face = group_image.crop((left, top, right, bottom))
                
                # Convert cropped face image to bytes
                buffered = io.BytesIO()
                cropped_face.save(buffered, format="JPEG")
                face_image_bytes = buffered.getvalue()
                
                # Search for this cropped face in the collection
                search_response = rekognition_client.search_faces_by_image(
                    CollectionId=collection_id,
                    Image={'Bytes': face_image_bytes},
                    MaxFaces=1,                # Max number of matching faces to return
                    FaceMatchThreshold=90      # Adjust threshold as needed
                )

                # Check if there's a match
                if search_response['FaceMatches']:
                    for match in search_response['FaceMatches']:
                        logger.info(f"Face {i + 1} matches with {match['Face']['ExternalImageId']} at {match['Similarity']:.2f}% similarity.")
                else:
                    logger.info(f"Face {i + 1} has no match in the collection.")
                    upload_new_face_to_aws(event, face_image_bytes, bucket_name)

            
            logger.info("Model run successfully on the image.")
            return response
        except botocore.exceptions.ClientError as error:
            logger.error(f"An error occurred while running the model: {error}")
            return None

    # Run the function to create the collection if it doesn't exist
    create_collection_if_not_exists(collection_id)
    
    # Run the model on the image
    return run_model_on_image(event)
