# -*- coding: utf-8 -*-
from google.colab import files
uploaded = files.upload()

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

import time
import xml.etree.cElementTree as ET

hits_paginator = client.get_paginator('list_hits')
assignments_paginator = client.get_paginator('list_assignments_for_hit')

result = []
for hits in hits_paginator.paginate():
    for hit in hits['HITs']:
        if hit['HITTypeId'] != '31L8DLROGDI3XS722AGL94NCHENA0R' and hit['HITTypeId'] != '3YSXL10HW5YMYDCETN1QJUH57AE7BJ': continue
        res_idx = hit['RequesterAnnotation']
        res_assign = hit['NumberOfAssignmentsCompleted']
        isAppend = 1
        
        for assignments in assignments_paginator.paginate(HITId=hit['HITId']):
            time.sleep(5)
            raw_val = 0
            raw_aro = 0
            raw_dom = 0
            raw_time = 0
            
            for assignment in assignments['Assignments']:
                answer = ET.fromstring(assignment['Answer'])
                for ans in answer:
                  if ans[0].text == 'Valence':
                    raw_val += int(ans[1].text)
                  if ans[0].text == 'Arousal':
                    raw_aro += int(ans[1].text)
                  if ans[0].text == 'Dominance':
                    raw_dom += int(ans[1].text)
                raw_time += (assignment['SubmitTime'] - assignment['AcceptTime']).total_seconds()
            
            if res_assign > 0 and isAppend != 0:
              avg_val = raw_val / res_assign
              avg_aro = raw_aro / res_assign
              avg_dom = raw_dom / res_assign
              avg_time = raw_time / res_assign
              result.append([res_idx, avg_val, avg_aro, avg_dom, avg_time, res_assign])
              isAppend = 0

import csv

with open('result.csv', 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(['idx', 'avg_valence', 'avg_arousal', 'avg_dominance', 'avg_time(seconds)', 'assignments'])
  writer.writerows(result)

time.sleep(180)
files.download('result.csv')