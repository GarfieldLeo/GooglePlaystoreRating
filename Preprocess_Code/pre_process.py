import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import re
import json

def convert_to_float(x):
    num_list = re.split(",", x)
    temp = ""
    for i in num_list:
        temp += i
    return float(temp)

data = pd.read_csv("googleplaystore.csv").dropna()
encoded_label = dict()

# process y into category and numeric
y_raw_data = data["Rating"]
y_cate_data = []
y_numr_data = []
for i in y_raw_data:
    y_cate_data.append(f"{i}")
    y_numr_data.append(i)
# yData has 2 cols of numerical and category
new_y_data = pd.DataFrame({"numerical": y_numr_data})

# Encode Category
le = preprocessing.LabelEncoder()
le.fit(data["Category"])
encoded_label["Category"] = tuple(le.classes_)
new_category = le.transform(data["Category"])
new_category = pd.get_dummies(data['Category'])
# Reviews
new_reviews = []
for i in data["Reviews"]:
    new_reviews.append(float(i))
new_reviews = np.array(new_reviews).reshape(-1,1)
scaler = preprocessing.StandardScaler()
scaler.fit(new_reviews)
new_reviews = scaler.transform(new_reviews)
new_reviews = [i[0] for i in new_reviews]

# Encode Size
size_dict = {"M": 1000000, "k":1000, "+":1}
new_size = []
for i in data["Size"]:
    if i == "Varies with device":
        new_size.append(0)
    else:
        num = convert_to_float(i[0:len(i)-1])
        new_size.append(num * size_dict[i[len(i)-1]])
new_size = np.array(new_size).reshape(-1,1)
scaler = preprocessing.StandardScaler()
scaler.fit(new_size)
new_size = scaler.transform(new_size)
new_size = [i[0] for i in new_size]

# Encode Type
le = preprocessing.LabelEncoder()
le.fit(data["Type"])
encoded_label["Type"] = tuple(le.classes_)
new_type = le.transform(data["Type"])
new_type = pd.get_dummies(data["Type"])
# Encode Installs
new_installs = []
for i in data["Installs"]:
    try:
        new_installs.append(convert_to_float(i[:len(i)-1]))
    except:
        new_installs.append(pd.NA)
new_installs = np.array(new_installs).reshape(-1,1)
scaler = preprocessing.StandardScaler()
scaler.fit(new_installs)
new_installs = scaler.transform(new_installs)
new_installs = [i[0] for i in new_installs]

# Encode price
new_price = []
for i in data["Price"]:
    try:
        if i == "0":
            new_price.append(0)
        else:
            new_price.append(float(i[1:len(i)]))
    except:
        new_price.append(pd.NA)

# Encode content rating
le = preprocessing.LabelEncoder()
le.fit(data["Content Rating"])
encoded_label["Content Rating"] = tuple(le.classes_)
new_content_rating = le.transform(data["Content Rating"])
new_content_rating = pd.get_dummies(data["Content Rating"])

# Encode Genres
le = preprocessing.LabelEncoder()
le.fit(data["Genres"])
encoded_label["Genres"] = tuple(le.classes_)
new_genres = le.transform(data["Genres"])
new_genres = pd.get_dummies(data["Genres"])

with open('encoded_label.json', 'w') as f:
    json.dump(encoded_label, f)

new_data = pd.DataFrame({"Reviews": new_reviews, "Size": new_size, "Installs":new_installs,"Price":new_price})
new_data = pd.concat([new_data, new_category, new_type, new_content_rating, new_genres, new_y_data], axis=1)
new_data = new_data.dropna()
new_y_data = new_data["numerical"]
new_data = new_data.drop("numerical",1)


X_train, X_test, y_train, y_test = train_test_split(new_data, new_y_data, test_size=0.3, random_state=10)
X_train.to_csv("xTrain.csv")
X_test.to_csv("xTest.csv")
y_train.to_csv("yTrain.csv")
y_test.to_csv("yTest.csv")

