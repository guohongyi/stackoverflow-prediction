# Machine Learning Deployment Tutorials
Sample end to end projects from data collection to putting models into production using flask, docker etc.

## If this repository helps you in anyway, show your love :heart: by putting a :star: on this project :v:

# 1. Predict Sales

Check out the corresponding medium blog post [https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4).

## Environment and tools
1. scikit-learn
2. pandas
3. numpy
4. flask

## Installation

`pip install scikit-learn pandas numpy flask`

`python model.py`

`python app.py`

![Logo](i1.png)

# 2. Predict House Prices

Download the dataset from [here](https://www.kaggle.com/shivachandel/kc-house-data).

## Environment and tools
1. scikit-learn
2. pandas
3. numpy
4. flask
5. docker

## Installation

`curl -X POST -H "Content-Type: application/json" -d @to_predict_json.json http://localhost:8080/predict_price`

where `to_predict.json` contains:

`{"grade":9.0,"lat":37.45,"long":12.09,"sqft_living":1470.08,"waterfront":0.0,"yr_built":2008.0}`

Output:

```
{
  "predict cost": 1022545.34768284
}
```

## Citing

```
@misc{Abhinav:2019,
  Author = {Abhinav Sagar},
  Title = {Machine-Learning-Deployment-Tutorials},
  Year = {2019},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/abhinavsagar/Machine-Learning-Deployment-Tutorials}}
}
```

## Contacts

If you want to keep updated with my latest articles and projects follow me on Medium. These are some of my contacts details:

1. [Personal Website](https://abhinavsagar.github.io/)
2. [Linkedin](https://in.linkedin.com/in/abhinavsagar4)
3. [Medium](https://medium.com/@abhinav.sagar)
4. [GitHub](https://github.com/abhinavsagar)
5. [Kaggle](https://www.kaggle.com/abhinavsagar)

