# Census Data Project
Predicting population income is essential for estimation of labor and supplies in policy-making. To this end, employing a model that estimates the income of individuals, given standard questionares from census data would be beneficial. This project explores the use of a machine learning model using census income data.   
  
## Repository structure
```
.
├── .github               # github actions and CI/CD setup files
├── data                  # data folder
├── model                 # model directory
├── notebooks             # EDA and train baseline notebook
├── src                   # source code for the project
│   ├── modeling          # modeling module
│   ├── preprocessing     # data preprocessing module
│   ├── evaluate_model.py # script to evaluate model
│   └── train_model.py    # script to train baseline model
└── main.py               # file to run app
└── model_card.md         # model description
└── tests                 # pytest folder
└── requirements.txt      # pip install environment
└── environments.yml      # conda install environment
└── runtime.txt           # heroku req file
└── Procfile              # heroku req file
```
  
## Environment Setup
To prepare environment using conda

> conda env create -f environments.yml
  
To prepare environment using pip

>  pip install -r requirements.txt
    

### Useful scripts
- `train_model.py` training pipeline
- `evaluate_model.py` model evaluation script
### Links
- [Heroku APP](https://predict-income-by-census-data.herokuapp.com/)
    - [x] Add get-method for greetings  
    - [x] Add post-method to predict target (income) given new data  
    - [ ] Add front-end interface  