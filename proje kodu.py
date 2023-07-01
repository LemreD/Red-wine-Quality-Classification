import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , GridSearchCV , cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("wine.csv")
df = data.copy()
df.head()
model_names = ["knn", "logistic", "navi","desicion","random forest","mlp"]
times = []
colors = ["red", "blue", "green","pink","purple","black"]
#%%eda
figure, ax = plt.subplots(1,5, figsize = (24,6))
sns.boxplot(data = df, x = "quality", y="fixed acidity", ax = ax[0])
sns.boxplot(data = df, x = "quality", y="volatile acidity", ax = ax[1])
sns.boxplot(data = df, x = "quality", y="citric acid", ax = ax[2])
sns.boxplot(data = df, x = "quality", y="residual sugar", ax = ax[3])
sns.boxplot(data = df, x = "quality", y="chlorides", ax = ax[4])
plt.show()


figure, ax = plt.subplots(1,6, figsize = (24,6))
sns.boxplot(data = df, x = "quality", y="free sulfur dioxide", ax = ax[0])
sns.boxplot(data = df, x = "quality", y="total sulfur dioxide", ax = ax[1])
sns.boxplot(data = df, x = "quality", y="density", ax = ax[2])
sns.boxplot(data = df, x = "quality", y="pH", ax = ax[3])
sns.boxplot(data = df, x = "quality", y="sulphates", ax = ax[4])
sns.boxplot(data = df, x = "quality", y="alcohol", ax = ax[5])
plt.show()

df["quality"].value_counts()
df["quality"] = df["quality"].apply(lambda value : 1 if value >= 7 else 0)
df["quality"].value_counts()
x = df[df.columns[:-1]]
y = df["quality"]

print(data['quality'].value_counts())
_ = sns.countplot(x='quality', data=data)


Numeric_cols = data.drop(columns=['quality']).columns

fig, ax = plt.subplots(4, 3, figsize=(15, 12))
for variable, subplot in zip(Numeric_cols, ax.flatten()):
    g=sns.histplot(data[variable],bins=30, kde=True, ax=subplot)
    g.lines[0].set_color('green')
    g.axvline(x=data[variable].mean(), color='m', label='Mean', linestyle='--', linewidth=2)
plt.tight_layout()



plt.figure(figsize=(12,12))
sns.heatmap(data.corr("pearson"),vmin=-1, vmax=1,cmap='coolwarm',annot=True, square=True)
#box
data_melted = pd.melt(data, id_vars = "quality",
                      var_name = "features",
                      value_name = "value")

plt.figure()
sns.boxplot(x = "features", y = "value", hue = "quality", data = data_melted)
plt.xticks(rotation = 90) # özellik isimlerinin dik hale getirilmesi
plt.show()

#pair
sns.pairplot(data, diag_kind = "kde", markers = "+",hue = "quality")
plt.show()


#%% bu kısım ön işlemeden önce alınan değerlerin yazdırıldığı kısım
def once():
    model_names = ["knn", "logistic", "navi","desicion","random forest","mlp"]
    times = []
    colors = ["red", "blue", "green","pink","purple","black"]

    data = pd.read_csv("wine.csv")
    x = data[data.columns[:-1]]
    y = data["quality"]

    x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.20)
    #%%
    print("KNN Classifier")
    start = time.time()
    knn = KNeighborsClassifier()
    knn_model = knn.fit(x_train , y_train)

    y_pred_knn = knn_model.predict(x_test)
    print(accuracy_score(y_test , y_pred_knn))

    knn_params = {"n_neighbors": np.arange(1,60)}
    knn_cv = GridSearchCV(knn , knn_params, cv = 10)
    knn_cv.fit(x_train , y_train)


    print("Best Parameters: " + str(knn_cv.best_params_))

    knn = KNeighborsClassifier(n_neighbors = 2)
    opt_knn = knn.fit(x_train , y_train)
    y_pred_knn = opt_knn.predict(x_test)
    print(accuracy_score(y_test , y_pred_knn))
    print(classification_report(y_test , y_pred_knn))

    end = time.time()
    times.append(end - start)

    print("-------------------------")

    #%%Logistic Regression
    start = time.time()
    print("Logistic Regression")
    logi = LogisticRegression(solver = "liblinear")
    log_model = logi.fit(x_train,y_train)
    log_model.predict_proba(x_test)[:5]

    y_pred_logi = log_model.predict(x_test)
    print(accuracy_score(y_test, y_pred_logi))
    print(  cross_val_score(log_model, x_test, y_test, cv = 10).mean()  )
    print(classification_report(y_test , y_pred_logi))


    end = time.time()
    times.append(end - start)

    print("-------------------------")


    #%%Naive Bayes
    start = time.time()
    print("Naive Bayes")
    nb = GaussianNB()
    nb_model = nb.fit(x_train , y_train)
    y_pred_nb = nb_model.predict(x_test)
    print(accuracy_score(y_test , y_pred_nb))
    cross_val_score(nb_model, x_test, y_test, cv = 10).mean()
    print(classification_report(y_test , y_pred_nb))

    end = time.time()
    times.append(end - start)
    print("-------------------------")


    #%% Desicion Tree Classifier
    start = time.time()
    print("Desicion Tree Classifier")
    clf = DecisionTreeClassifier()
    clf_model = clf.fit(x_train , y_train)
    y_pred_clf = clf_model.predict(x_test)
    accuracy_score(y_test , y_pred_clf)
    print(classification_report(y_test , y_pred_clf))


    end = time.time()
    times.append(end - start)
    print("-------------------------")

    #%% Random Forest Classifier
    start = time.time()
    print("Random Forest Classifier")
    rf = RandomForestClassifier()
    rf_model = rf.fit(x_train, y_train)
    y_pred_rf = rf_model.predict(x_test)
    accuracy_score(y_test, y_pred_rf)
    # Optimized Model
    rf_model = RandomForestClassifier(max_features = "auto" , max_depth = 19,
                                      random_state = 44 , n_estimators = 1000)

    opt_rf = rf_model.fit(x_train, y_train)
    y_pred_rf = opt_rf.predict(x_test)
    accuracy_score(y_test, y_pred_rf)
    print(classification_report(y_test , y_pred_rf))

    end = time.time()
    times.append(end - start)

    #%% neural
    print("MLP Classifier")
    # MLP Classifier (Multi-Layer Perceptron Classifier)
    start = time.time()
    mlpc = MLPClassifier()
    mlpc_model = mlpc.fit(x_train , y_train)
    y_pred_mlpc = mlpc.predict(x_test)
    accuracy_score(y_test, y_pred_mlpc)

    mlpc = MLPClassifier(activation = "relu" , alpha = 0.01,
                         hidden_layer_sizes = (100, 100, 100) , solver = "adam")

    opt_mlpc = mlpc.fit(x_train , y_train)

    y_pred_mlpc = opt_mlpc.predict(x_test)
    acc = accuracy_score(y_test, y_pred_mlpc)
    print("acc:",acc)

    print(classification_report(y_test , y_pred_mlpc))
    end = time.time()
    times.append(end - start)
    #%%süre
    plt.bar(model_names, times, color=colors)
    plt.xlabel("Model name")
    plt.ylabel("Training time (seconds)")
    plt.show()

#%%split - standardizasyon - smote
scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.20)


smote = SMOTE(k_neighbors = 4 )
x_train, y_train = smote.fit_resample(x_train, y_train)
y_train.value_counts()
#%%KNN Classifier (K-Nearest Neighbors Algorithm)
print("KNN Classifier")
start = time.time()
knn = KNeighborsClassifier()
knn_model = knn.fit(x_train , y_train)

y_pred_knn = knn_model.predict(x_test)
print(accuracy_score(y_test , y_pred_knn))

knn_params = {"n_neighbors": np.arange(1,60)}
knn_cv = GridSearchCV(knn , knn_params, cv = 10)
knn_cv.fit(x_train , y_train)


print("Best Parameters: " + str(knn_cv.best_params_))

knn = KNeighborsClassifier(n_neighbors = 2)
opt_knn = knn.fit(x_train , y_train)
y_pred_knn = opt_knn.predict(x_test)
print(accuracy_score(y_test , y_pred_knn))
print(classification_report(y_test , y_pred_knn))

score = round(accuracy_score(y_test, y_pred_knn), 6)
cm = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm, annot = True, fmt = ".0f")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Accuracy Score: {0}".format(score), size = 15)
plt.show()
end = time.time()
times.append(end - start)

print("-------------------------")

#%%Logistic Regression
start = time.time()
print("Logistic Regression")
logi = LogisticRegression(solver = "liblinear")
log_model = logi.fit(x_train,y_train)
log_model.predict_proba(x_test)[:5]

y_pred_logi = log_model.predict(x_test)
print(accuracy_score(y_test, y_pred_logi))
print(  cross_val_score(log_model, x_test, y_test, cv = 10).mean()  )
print(classification_report(y_test , y_pred_logi))

score = round(accuracy_score(y_test, y_pred_logi), 6)
cm = confusion_matrix(y_test, y_pred_logi)
sns.heatmap(cm, annot = True, fmt = ".0f")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Accuracy Score: {0}".format(score), size = 15)
plt.show()

end = time.time()
times.append(end - start)

print("-------------------------")


#%%Naive Bayes
start = time.time()
print("Naive Bayes")
nb = GaussianNB()
nb_model = nb.fit(x_train , y_train)
y_pred_nb = nb_model.predict(x_test)
print(accuracy_score(y_test , y_pred_nb))
cross_val_score(nb_model, x_test, y_test, cv = 10).mean()
print(classification_report(y_test , y_pred_nb))

score = round(accuracy_score(y_test, y_pred_nb), 6)
cm = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm, annot = True, fmt = ".0f")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Accuracy Score: {0}".format(score), size = 15)
plt.show()
end = time.time()
times.append(end - start)
print("-------------------------")


#%% Desicion Tree Classifier
start = time.time()
print("Desicion Tree Classifier")
clf = DecisionTreeClassifier()
clf_model = clf.fit(x_train , y_train)
y_pred_clf = clf_model.predict(x_test)
accuracy_score(y_test , y_pred_clf)
print(classification_report(y_test , y_pred_clf))

score = round(accuracy_score(y_test, y_pred_clf), 6)
cm = confusion_matrix(y_test, y_pred_clf)
sns.heatmap(cm, annot = True, fmt = ".0f")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Accuracy Score: {0}".format(score), size = 15)
plt.show()
end = time.time()
times.append(end - start)
print("-------------------------")

#%% Random Forest Classifier
start = time.time()
print("Random Forest Classifier")
rf = RandomForestClassifier()
rf_model = rf.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)
accuracy_score(y_test, y_pred_rf)
# Optimized Model
rf_model = RandomForestClassifier(max_features = "auto" , max_depth = 19,
                                  random_state = 44 , n_estimators = 1000)

opt_rf = rf_model.fit(x_train, y_train)
y_pred_rf = opt_rf.predict(x_test)
accuracy_score(y_test, y_pred_rf)
print(classification_report(y_test , y_pred_rf))

score = round(accuracy_score(y_test, y_pred_rf), 6)
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot = True, fmt = ".0f")
plt.xlabel("Predicted Values") 
plt.ylabel("Actual Values")
plt.title("Accuracy Score: {0}".format(score), size = 15)
plt.show()
end = time.time()
times.append(end - start)

#%% neural
print("MLP Classifier")
# MLP Classifier (Multi-Layer Perceptron Classifier)
start = time.time()
mlpc = MLPClassifier()
mlpc_model = mlpc.fit(x_train , y_train)
y_pred_mlpc = mlpc.predict(x_test)
accuracy_score(y_test, y_pred_mlpc)

mlpc = MLPClassifier(activation = "relu" , alpha = 0.01,
                     hidden_layer_sizes = (100, 100, 100) , solver = "adam",verbose=1)

opt_mlpc = mlpc.fit(x_train , y_train)

y_pred_mlpc = opt_mlpc.predict(x_test)
acc = accuracy_score(y_test, y_pred_mlpc)
print("acc:",acc)

print(classification_report(y_test , y_pred_mlpc))
score = round(accuracy_score(y_test, y_pred_mlpc), 6)
cm = confusion_matrix(y_test, y_pred_mlpc)
sns.heatmap(cm, annot = True, fmt = ".0f")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Accuracy Score: {0}".format(score), size = 15)
plt.show()
end = time.time()
times.append(end - start)

#%%çalışma süresi
plt.bar(model_names, times, color=colors)
plt.xlabel("Model name")
plt.ylabel("Training time (seconds)")
plt.show()






once()












































