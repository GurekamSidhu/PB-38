## Server

### To Run
To start server at `localhost:3000`
```
python manage.py run <dev|test|prod>
```


### Endpoints
#### Retrain Model
Title | Retrain Model
---|---
URL | *GET* api/retrain
Authorization | Admin access only.
Success Response | Code: 200 OK </br> Content: ``` {'status': 'Success', 'message':"Successfully retrained model."} ```
Error Response | Code: 401 UNAUTHORIZED </br> Content: ``` { "status":"Failure", "message":<error message> }```

#### Query price
Title | Get Predicted Price
---|---
URL | *GET* api/calculate
Authorization | User/Admin access only
URL Params | Required: </br> `duration=[integer]` duration of visit in seconds </br> `speciality=[integer]` speciality of practitioner </br> `eventType=[integer]` Voice/Video/Report </br> `type=[integer]`Counseling/Consultation
Success Response | Code: 200 OK </br> Content: ``` {"status":"Success", "message":"Successful price generation.", "data":{"price":<price>}}```
Error Response | Code: 401 UNAUTHORIZED </br> Content: ``` { "status":"Failure", "message":<error message> }``` </br> Code: 400 BAD REQUEST </br> Content: ``` { "status":"Failure", "message":<error message> }```

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
