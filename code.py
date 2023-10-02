import numpy as np 
import pandas as pd 

!wget https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv

data = pd.read_csv("data.csv")

# Q1
data.columns = data.columns.str.replace(' ', '_').str.lower()
data.fillna(0, inplace=True)
data.rename(columns={'msrp': 'price'}, inplace=True)

mode_transmission_type = data['transmission_type'].mode()[0]
mode_transmission_type

# Q2
numerical_features = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'price']
corr_matrix = data[numerical_features].corr()

max_corr = corr_matrix.unstack().sort_values(ascending=False)
max_corr.head(2)

#Q3
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score

# Make price binary
data['above_average'] = (data['price'] > data['price'].mean()).astype(int)

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(data.drop(['price', 'above_average'], axis=1),
                                                    data['above_average'],
                                                    test_size=0.4,
                                                    random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Calculate mutual information scores
categorical_cols = ['make', 'model', 'transmission_type', 'vehicle_style']

mi_scores = {}
for col in categorical_cols:
    mi_scores[col] = round(mutual_info_score(X_train[col], y_train), 2)

lowest_mi_score = min(mi_scores, key=mi_scores.get)
lowest_mi_score

#Q4
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Perform one-hot encoding
dv = DictVectorizer(sparse=False)
train_dict = X_train[categorical_cols].to_dict(orient='records')
X_train_encoded = dv.fit_transform(train_dict)

val_dict = X_val[categorical_cols].to_dict(orient='records')
X_val_encoded = dv.transform(val_dict)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_features[0:4]])
X_val_scaled = scaler.transform(X_val[numerical_features[0:4]])

# Combine one-hot encoded categorical and scaled numerical features
X_train_final = pd.concat([pd.DataFrame(X_train_scaled, columns=numerical_features[0:4]), pd.DataFrame(X_train_encoded, columns=dv.get_feature_names_out(categorical_cols))], axis=1)
X_val_final = pd.concat([pd.DataFrame(X_val_scaled, columns=numerical_features[0:4]), pd.DataFrame(X_val_encoded, columns=dv.get_feature_names_out(categorical_cols))], axis=1)

# Train logistic regression model
model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
model.fit(X_train_final, y_train)

# Predict and calculate accuracy on the validation dataset
y_val_pred = model.predict(X_val_final)
accuracy = round(accuracy_score(y_val, y_val_pred), 2)
accuracy

#Q5
from sklearn.metrics import accuracy_score

model.fit(X_train_final, y_train)
accuracy_with_all_features = accuracy_score(y_val, model.predict(X_val_final))

least_useful_feature = None
smallest_difference = float('inf')

for feature in X_train_final.columns:
    X_train_subset = X_train_final.drop([feature], axis=1)
    X_val_subset = X_val_final.drop([feature], axis=1)
    
    model.fit(X_train_subset, y_train)
    
    y_val_pred = model.predict(X_val_subset)
    accuracy_without_feature = accuracy_score(y_val, y_val_pred)
    
    difference = accuracy_with_all_features - accuracy_without_feature
    
    if difference < smallest_difference:
        smallest_difference = difference
        least_useful_feature = feature

least_useful_feature

#Q6
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

alphas = [0, 0.01, 0.1, 1, 10]
best_rmse = float('inf')
best_alpha = None

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha, solver='sag', random_state=42)
    ridge_model.fit(X_train_final, y_train_log)
    y_val_pred_log = ridge_model.predict(X_val_final)
    rmse = np.sqrt(mean_squared_error(y_val_log, y_val_pred_log))
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha

round(best_alpha, 3)
