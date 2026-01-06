import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, precision_score, \
    recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

#generowanie 1000 danych na temat wzrostu
#obliczanie średniej, mediany i odchylenia standardowego
height = np.random.normal(loc=175, scale=12, size=1000)
mean_height = np.mean(height)
median_height = np.median(height)
std_deviation_height = np.std(height)

print(f"Mean: {mean_height:.3f}")
print(f"Median: {median_height:.3f}")
print(f"Standard Deviation: {std_deviation_height:.3f}")

# https://www.geeksforgeeks.org/plotting-histogram-in-python-using-matplotlib/
# tworzenie histogramu dla wygenerowanych danych
plt.hist(height, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Wzrost')
plt.title('Histogram')
plt.show()

#print("Wylosowane wartości wzrostu:")
#for i, h in enumerate(height, 1):
#    print(f"{h:.2f}", end=", ")
#    if i % 10 == 0:
#        print()

# https://www.geeksforgeeks.org/numpy-percentile-in-python/
# obliczanie centyli dla wzrostu z bazy danych
p25 = np.percentile(height, 25)
p50 = np.percentile(height, 50)
p75 = np.percentile(height, 75)
print(f"25. percentyl: {p25:.2f} cm")
print(f"50. percentyl (mediana): {p50:.2f} cm")
print(f"75. percentyl: {p75:.2f} cm")

#obliczanie wartości odstających
# IQR - rozstęp międzykwartylowy - 50% środkowych danych
IQR = p75 - p25
lower_bound = p25 - 1.5 * IQR
upper_bound = p75 + 1.5 * IQR
outliers = height[(height < lower_bound) | (height > upper_bound)]

print(f"Liczba wartości odstających: {len(outliers)}")
print(f"Zakres wartości odstających: < {lower_bound:.2f} cm lub > {upper_bound:.2f} cm")

#test t-studenta, hipoteza H0
t_student, p_val = stats.ttest_1samp(height, popmean=170)
print(f"t-student: {t_student:.2f}")
print(f"p-wartość: {p_val}")

print(f"H0: Średni wzrost w populacji wynosi 170 cm")
if p_val < 0.05:
    print("Różnica jest statystycznie istotna (odrzucamy H0).")
else:
    print("Brak podstaw do odrzucenia H0.")

#Prawdopodobieństwo, że wzrost > 190 cm
more_than_190 = np.mean(height > 190)
print(f"Prawdopodobieństwo wzrostu > 190 cm: {more_than_190:.2%}")

#wczytanie danych z pliku .csv
file = pd.read_csv("heart_disease_datasetx.csv", sep =';')
#wczytanie kolumn dla upewnienia się czy w kodzie używamy dobrych nazw
print(file.columns.tolist())

#obliczanie zależności występowania choroby od płci
disease_by_sex = file[file['Disease'] == 1]['Sex'].value_counts()
total_by_sex = file['Sex'].value_counts()

percent_men = disease_by_sex.get('male', 1) / total_by_sex.get('male', 1) * 100
percent_women = disease_by_sex.get('female', 1) / total_by_sex.get('female', 1) * 100
diff_percent = percent_men - percent_women

print("1. Choroby serca – płeć:")
print(f"Mężczyźni: {percent_men:.2f}%, Kobiety: {percent_women:.2f}%")
print(f"Różnica: {abs(diff_percent):.2f}% więcej u {'mężczyzn' if diff_percent > 0 else 'kobiet'}\n")

print("2. Średni cholesterol w zależności od płci i występowania chorób serca:")
cholesterol_means = file.groupby(['Sex', 'Disease'])['Serum cholesterol in mg/dl'].mean()
print(cholesterol_means, "\n")

plt.figure(figsize=(8,5))
file[file['Disease'] == 1]['Age'].hist(bins=20, edgecolor='black', color='pink')
plt.title("Histogram wieku osób z chorobą serca")
plt.xlabel("Wiek")
plt.ylabel("Liczba osób")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(data=file, x='Disease', y='Maximum heart rate achieved')
plt.title("Maksymalne tętno vs choroba serca")
plt.xlabel("Choroba serca (false = brak, true = obecna)")
plt.ylabel("Maksymalne tętno")
plt.grid(True)
plt.show()

#crosstabs https://www.geeksforgeeks.org/pandas-crosstab-function-in-python/
pain_vs_disease = pd.crosstab(file['Exercise induced angina'], file['Disease'])
pain_vs_disease.plot(kind='bar', stacked=True, colormap='Set1', figsize=(10, 8))
plt.title("Częstość choroby serca a ból dławicowy przy wysiłku")
plt.xlabel("Ból dławicowy przy wysiłku (0 = nie, 1 = tak)")
plt.ylabel("Liczba osób")
plt.legend(title="Choroba serca")
plt.grid(True)
plt.show()

# do wykonania regresji logistycznej korzystano ze źródeł:
# https://www.geeksforgeeks.org/ml-logistic-regression-using-python/
# https://www.datacamp.com/tutorial/understanding-logistic-regression-python
# https://alfa.im.pwr.edu.pl/~sobczyk/dydaktyka/regresja_logistyczna_dane.pdf


file['Sex'] = LabelEncoder().fit_transform(file['Sex'])  # male=1, female=0, kodowanie tekstu do wykonania regresji
file['Fasting blood sugar > 120 mg/dl'] = file['Fasting blood sugar > 120 mg/dl'].astype(int) # true = 1, false = 0
file['Exercise induced angina'] = file['Exercise induced angina'].astype(int)

data = [
    'Age',
    'Sex',
    'Chest pain type',
    'Resting blood pressure',
    'Serum cholesterol in mg/dl',
    'Fasting blood sugar > 120 mg/dl',
    'Maximum heart rate achieved',
    'Exercise induced angina'
]
X = file[data] #dane wejściowe
y = file['Disease'].astype(int) # przewidywanie występowania disease 1 = true, 0 = false

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model regresji logistycznej
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predykcje, klasyfikacja danych testowych
y_pred = model.predict(X_test) # tablica z przewidzianymi wartościami [0, 0, 1, 0, 1 itp.]
y_prob = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# Współczynniki regresji
print("Współczynniki regresji:")
coefficients = pd.Series(model.coef_[0], index=data)
print(coefficients)

# Macierz pomyłek i klasyfikacja
print("\nKlasyfikacja:")
print(classification_report(y_test, y_pred))

# https://www.geeksforgeeks.org/evaluation-metrics-for-classification-model-in-python/
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f'Precision(test): {precision:.3f}')
print(f'Recall(test): {recall:.3f}')
print(f'F1 Score(test): {f1:.3f}')
# Krzywa ROC i AUC
# Receiver Operator Characteristic
# AUC oddalenie od prostej y = x (równoważnie pole pod wykresem)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# jupiter notebook - wrzucić przez collab