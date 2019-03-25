## Server

### To Run
To start server at `localhost:5000`
```
python manage.py run <dev|test|prod>
```


### Endpoints
#### Retrain Model
To retrain the price predicting model:
- NEED ADMIN ACCESS (see authorization below)

`GET /api/retrain`

**Response:**

Success:

Status Code:

	- 200 OK

Data:
```
{
	'status':'Success',
	'message': <message>
}
```

Error:

Status Code:

	- 401 UNAUTHORIZED
	- 501 INTERNAL FAILURE

Data:
```
{
	'status':'Failure',
	'message' <message>
}
```

#### Query price
- NEED USER ACCESS (see authorization below)

`GET /api/calculate?`

URL Parameters:
- `duration=[integer]` (required)
- `speciality=[integer]` (required)
- `eventType=[integer]` (required)
- `type=[integer]` (required)

**Response:**

Success:

Status Code:

	- 200 OK

Data:
```
{
	'status': 'Success',
	'message': <message>
}
```

Error:

Status Code:

	- 401 UNAUTHORIZED
	- 501 INTERNAL FAILURE

Data:
```
{
	'status': 'Failure',
	'message': <message>
}
```
#### Authentication
To access endpoints, use Basic HTTP Authentication and tokens below.

Users|Token
--- | ---
User | 6b93ccba414ac1d0ae1e77f3fac560c748a6701ed6946735a49d463351518e16
Admin | 240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9


### Testing
To run tests:
```
python manage.py test
```
