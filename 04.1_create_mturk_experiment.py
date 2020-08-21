# -*- coding: utf-8 -*-
from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv('https://gist.githubusercontent.com/bshmueli/c99fc0abf56460e644bd610bf3024e83/raw/720285d133c85d94e0aa3fe3edcc199f6d99e3f7/lab4-data.csv')
idx = df.idx.to_list()
tweets = df.text.to_list()
import nltk
from nltk.tokenize import TweetTokenizer

tweets_len = [len(TweetTokenizer().tokenize(i)) for i in tweets]
avg_len=int(sum(tweets_len)/len(tweets_len))

"""# Authenticate"""

import boto3
import pandas as pd

CREDENTIALS_FILE = 'credentials.csv'
credentials = pd.read_csv(CREDENTIALS_FILE).to_dict('records')[0]
aws_access_key_id = credentials['Access key ID']
aws_secret_access_key = credentials['Secret access key']

region_name = 'us-east-1'
endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
 
client = boto3.client(
    'mturk',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    endpoint_url=endpoint_url,
    region_name=region_name
)

"""#Get  Balance"""

print(client.get_account_balance())

"""#Create HIT Type"""

one_minute = 60 #seconds
one_hour = 60 * one_minute
one_day = 24 * one_hour
one_week = 7 * one_day
expire_time = 8 * one_week
hit_type_response = client.create_hit_type(
    AutoApprovalDelayInSeconds=60 * one_minute,
    AssignmentDurationInSeconds=30 * one_minute,
    Reward='0.50',
    Title='NCUTS_Team_Romeo Score the text (may include some offensive content)',
    Keywords='nctu,language,NCTU',
    Description='Choose the emotion mostly matching your feeling in the sentence',
    QualificationRequirements=[
        {
            'QualificationTypeId': '00000000000000000071',
            'Comparator': 'In',
            'LocaleValues': [
                {
                    'Country': 'TW'
                },
                {
                    'Country': 'CA'
                },
                {
                    'Country': 'US'
                },
            ],
            'RequiredToPreview': True,
            'ActionsGuarded': 'PreviewAndAccept'
        },
        {
            'QualificationTypeId': '00000000000000000060',
             'Comparator': 'EqualTo',
            'IntegerValues':[1],
        },
    ]
)

hit_type_response_long = client.create_hit_type(
    AutoApprovalDelayInSeconds=60 * one_minute,
    AssignmentDurationInSeconds=30 * one_minute,
    Reward='1.00',
    Title='NCUTS_Team_Romeo core the text (may include some offensive content)',
    Keywords='nctu,language',
    Description='Choose the emotion mostly matching your feeling in the sentence',
    QualificationRequirements=[
        {
            'QualificationTypeId': '00000000000000000071',
            'Comparator': 'In',
            'LocaleValues': [
                {
                    'Country': 'TW'
                },
                {
                    'Country': 'CA'
                },
                {
                    'Country': 'US'
                },
            ],
            'RequiredToPreview': True,
            'ActionsGuarded': 'PreviewAndAccept'
        },
        {
            'QualificationTypeId': '00000000000000000060',
         'Comparator': 'EqualTo',
            'IntegerValues':[1],
        },
    ]
)

hit_type_id = hit_type_response['HITTypeId']
hit_type_id_long = hit_type_response_long['HITTypeId']

for i,j in zip(idx,tweets_len):
    question='''<?xml version="1.0" encoding="UTF-8"?>
<ExternalQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd">
  <ExternalURL>https://walibox.bixone.com/nlp.html?idx='''+str(i)+'''</ExternalURL>
  <FrameHeight>800</FrameHeight>
</ExternalQuestion>'''
    if j>avg_len:
        response = client.create_hit_with_hit_type(
    HITTypeId=hit_type_id_long,
    MaxAssignments=3,
    LifetimeInSeconds=expire_time,
    Question=question,
    RequesterAnnotation=str(i),
)
    else:
        response = client.create_hit_with_hit_type(
    HITTypeId=hit_type_id,
    MaxAssignments=3,
    LifetimeInSeconds=expire_time,
    Question=question,
    RequesterAnnotation=str(i),
)
