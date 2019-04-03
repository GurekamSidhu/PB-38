# PB38

Price Setting API

## Model training

Data gathered from receipts.bson. Schema for most features (besides duration) is configurable in receipts-schema.json.
Features can be class-based ("featurename": "class") or numerical ("featurename": "number"). 

Duration is hard-coded as (endTime - startTime)

## Scalability testing

A "fake" dataset can be generated with receipts_fake_gen.py (default 500 new entries). This synthetic dataset will be placed
in the bin folder and automatically detected by receipts_model.py and used in training. Delete receipts_fake.json to remove

This can be used to improve model accuracy at the expense of overfitting (may be useful when dataset is still small), 
or test the model training time with larger data sets.

## Graphing app

The graphing app can be launched with receipts_graph.py. This will show the model accuracy, as well as the weight and distribution
of each feature.

### Server setup

See readme in server directory.
