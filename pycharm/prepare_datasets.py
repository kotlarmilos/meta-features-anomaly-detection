import json
import pandas as pd
import csv
import os

with open('../../Downloads/combined_labels.json') as f:
  data = json.load(f)

for key in data:
  df = pd.read_csv('../../Downloads/110_240_bundle_archive/'+key, error_bad_lines=False, warn_bad_lines=False, index_col=None)
  df.columns = ['timestamp', 'value']
  df['label'] = '\"n\"'
  for a in data[key]:
    df.loc[df['timestamp'] == a, 'label'] = '\"o\"'

  df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int)

  name = os.path.basename(key)
  anomaly_ratio = df.loc[df['label'] == '\"o\"'].shape[0]/df.shape[0]
  metadata = {
      "name": name,
      "type_of_data": ["temporal"],
      "domain": ["software"],
      "anomaly_types": ["single-point"],
      "anomaly_space": "univariate",
      "anomaly_entropy": anomaly_ratio,
      "label": "label",
      "files": [name]
    }

  os.mkdir('../../Downloads/more/'+name)
  df.to_csv('../../Downloads/more/'+name+'/'+name, index=False, quoting=csv.QUOTE_NONE)
  with open('../../Downloads/more/'+name+'/metadata.json', 'w') as f:
    json.dump(metadata, f)