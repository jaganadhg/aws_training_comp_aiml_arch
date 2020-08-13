import json
import boto3


ENDPOINT_NAME = os.environ['ENDPOINT_NAME']

runtime = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    
    data = json.loads(json.dumps(data))
    payload = data['data']
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
    ContentType='application/x-npy',
    Body=payload
    )
    
    
    print(response)
    
    
    return response
