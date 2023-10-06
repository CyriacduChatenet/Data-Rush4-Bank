# Rush 4 Bank prediction
Epitech Digital : data piscine rush 4

[![python](https://img.shields.io/badge/Python-3572A5?style=for-the-badge&logo=python&logoColor=FFFFFF)](https://www.python.org/)

## CLI
run algorithm
```
python3 src/main.py arg
```
arg is in arg list

Arg list:
- one: run knn file
- two: run correlation matrix
- three: run logistic regression
- four run linear regression

## Packages :
- pandas
- sk learn

## CSV
csv file is located in data folder. this folder is located at root of project
example csv file :
```
month,credit_amount,credit_term,age,sex,education,product_type,having_children_flg,region,income,family_status,phone_operator,is_client,bad_client_target
```
## Warning
You can change manually k value in ```main.py``` file :
```
Knn.calculateKnn(StandardScaler(),data_encoded,data, 4)
```

By default third param is empty fixed_k value is optionnal param for test. the algorithme search automatically k value
