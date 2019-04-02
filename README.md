# PB38

Price Setting API

## Model training

Data gathered from receipts.bson. Schema for most features (besides duration) is configurable in receipts-schema.json.
Features can be class-based ("featurename": "class") or numerical ("featurename": "number"). 

Duration is hard-coded as (endTime - startTime)

### Server setup

See readme in server directory.
