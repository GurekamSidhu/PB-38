# PB38

Price Setting API

## Model training

Data gathered from receipts.bson. Schema for most features (besides duration) is configurable in receipts-schema.json.
Features can be class-based ("featurename": "class") or numerical ("featurename": "number"). 

Duration is hard-coded as (endTime - startTime)

### Server setup

Setup and run server (on windows)

From /PB-38

Install dependencies

```
pip install -r requirements.txt
```

If you are adding more dependencies

```
pip freeze > requirements.txt
```

Linux and Mac:

```
export FLASK_APP=flaskr
export FLASK_ENV=development
flask run
```

Windows cmd:

```
set FLASK_APP=flaskr
set FLASK_ENV=development
flask run
```

Windows PowerShell:

```
$env:FLASK_APP = "flaskr"
$env:FLASK_ENV = "development"
flask run
```
