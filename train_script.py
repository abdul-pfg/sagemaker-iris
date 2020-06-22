from sagemaker.sklearn.estimator import SKLearn
import boto3
import os


#USER VARIABLES
#role = os.environ['role']
#train_script = os.environ['train_script']
#source = os.environ['source']
#train_data = os.environ['train_data']


def train():

    try:

        #Create a sagemaker.sklearn.SKLearn Estimator

        #print('start')
        #print('create SKlearn obj')
        
        proxy = os.environ.get('PROXY')
        #os.environ['http_proxy'] = proxy 
        #os.environ['HTTP_PROXY'] = proxy
        #os.environ['https_proxy'] = proxy
        #os.environ['HTTPS_PROXY'] = proxy
        
        aws_sklearn = SKLearn(entry_point='sklearn-iris-main.py',
                            source_dir='s3://aws-sagemaker-iris-us-east-2-304472691870/sklearn-iris-main.py.tar.gz',
                            train_instance_type='ml.m4.xlarge',
                            role='aws-sagemaker-role-s843971')
                            
        #Call the fit method on SKlearn estimator which uses our python script to train the model
        
        print('run fit method')
       
        aws_sklearn.fit({'train':'s3://aws-sagemaker-iris-us-east-2-304472691870/train'})

        #Deploy the model created in previous step and create an endpoint
        print('deploy')
        aws_sklearn_predictor = aws_sklearn.deploy(instance_type='ml.m4.xlarge', 
                                                initial_instance_count=1)

    except Exception as e:
        return e
    else:
        return 'success'


if __name__ =='__main__':
    train()


    
