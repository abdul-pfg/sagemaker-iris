from sagemaker.sklearn.estimator import SKLearn
import boto3
import os


#ENV VARIABLES
ROLE = os.environ.get('ROLE')
TRAIN_SCRIPT = os.environ.get('TRAIN_SCRIPT')
SOURCE = os.environ.get('SOURCE')
TRAIN_DATA = os.environ.get('TRAIN_DATA')


def train():

    try:

        #Create a sagemaker.sklearn.SKLearn Estimator
    
        aws_sklearn = SKLearn(entry_point=TRAIN_SCRIPT,
                            source_dir=SOURCE,
                            train_instance_type='ml.m4.xlarge',
                            role=ROLE)
                            
        #Call the fit method on SKlearn estimator which uses our python script to train the model
        
        aws_sklearn.fit({'train':TRAIN_DATA})

        #Deploy the model created in previous step and create an endpoint
        
        aws_sklearn_predictor = aws_sklearn.deploy(instance_type='ml.m4.xlarge', 
                                                initial_instance_count=1)

    except Exception as e:
        return e
    else:
        return 'success'


if __name__ =='__main__':
    train()



    
