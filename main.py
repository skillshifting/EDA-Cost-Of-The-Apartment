# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# %% [markdown]
# ### №1 - Сбор датасета

# %%
np.random.seed(42)
n_samples = 200

area = np.random.normal(70, 20, n_samples).round(1)  # Площадь в кв.м.
area = np.clip(area, 25, 150)
dist_metro = np.random.exponential(1.5, n_samples).round(1)  # До метро в км
renovation = np.random.choice(["None", "Cosmetic", "Euro"], n_samples, p=[0.3, 0.5, 0.2])

# %%
price = (
    50
    + area * 2.5
    - dist_metro * 10
    + np.where(renovation == "Cosmetic", 15, np.where(renovation == "Euro", 40, 0))
    + np.random.normal(0, 15, n_samples)
)
price = price.round(1)

# %%
dist_metro[np.random.choice([True, False], n_samples, p=[0.05, 0.95])] = np.nan  # Пропуски
renovation = renovation.astype(object)
renovation[np.random.choice([True, False], n_samples, p=[0.04, 0.96])] = None  # Пропуски

price[10] = 650.0  # Выброс: огромная цена для маленькой квартиры
area[20] = -45.0  # Ошибка ввода: отрицательная площадь

# Собираем DataFrame
df = pd.DataFrame(
    {
        "Area_Sqm": area,
        "Dist_Metro_Km": dist_metro,
        "Renovation": renovation,
        "Price_K_USD": price,
    }
)

# Добавляем дубликаты строк
df = pd.concat([df, df.iloc[[5, 12]]], ignore_index=True)
print("Датасет готов к EDA и регрессии! Размер таблицы:", df.shape)

# %% [markdown]
# ### №2 - Начальная обработка датасета

# %%
print("Вид датасета")
print(df.tail())
print(df['Renovation'].unique())

print("="*100,"\n")
print("Краткая информация по датасету")
df.info()


# %%
print("Количество пустых ячеек")
print(df.isna().sum())

print("="*100,"\n")
print("Количество дубликатов")
print(df.duplicated().reset_index())

print("="*100,"\n")
print("Проверка на аномалии")
df.describe()

# %%
df_cleaned = df.drop_duplicates().reset_index(drop=True)
df_cleaned = df_cleaned[df_cleaned['Area_Sqm']>0]
df_cleaned = df_cleaned[df_cleaned['Price_K_USD']<500]


Median_metro = df_cleaned['Dist_Metro_Km'].median()

df_cleaned['Dist_Metro_Km'] = df_cleaned['Dist_Metro_Km'].fillna(Median_metro)

renovation_mode = df_cleaned['Renovation'].mode()[0]
df_cleaned['Renovation'] = df_cleaned['Renovation'].fillna(renovation_mode)

renovation_dict = {'None':0, 'Euro':1 , 'Cosmetic':2}

df_cleaned["Renovation"] = df_cleaned["Renovation"].astype("category")
df_cleaned["Renovation_code"] = df_cleaned["Renovation"].map(renovation_dict)
df_cleaned["Renovation_code"] = df_cleaned["Renovation_code"].astype("int")

print(df_cleaned.head())

print("Количество пропусков после обработки")
print(df_cleaned.isna().sum(),'\n')

print("Количество дубликатов после обработки")
print(df_cleaned.duplicated().sum())

df_cleaned.describe()

print(df_cleaned.head())

# %% [markdown]
# ## №3 - Построение базовых графиков

# %%
pocket = []

for k in range(int((min(df_cleaned['Price_K_USD'].unique().tolist())-(min(df_cleaned['Price_K_USD'].unique().tolist())%100))),int((max(df_cleaned['Price_K_USD'].unique().tolist())+(100-(max(df_cleaned['Price_K_USD'].unique().tolist())%100))))+1,100):
    if k not in pocket:
            pocket.append(k)

labels = ['0-100','100-200','200-300','300-400']

df_cleaned['Pocket_price'] = pd.cut(df_cleaned['Price_K_USD'],bins=pocket,labels=labels)

counts_price = df_cleaned['Pocket_price'].value_counts().sort_index()
counts_renovation = df_cleaned['Renovation'].value_counts().sort_index()
print(counts_renovation)

# %%
fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, figsize = (16,4))

ax1.bar(counts_price.index.astype(str),counts_price.values)
ax1.set_title('')
ax1.grid(True)

ax2.pie(counts_renovation.values,labels=counts_renovation.index.astype(str),autopct='%1.1f%%')

ax3.boxplot(df_cleaned['Dist_Metro_Km'].dropna(),patch_artist=True)

ax4.boxplot(df_cleaned['Price_K_USD'].dropna(),patch_artist=True)

plt.tight_layout()
plt.show()



# %% [markdown]
# ## №4 - Построение корреляционной матрицы и линейной регрессии

# %%
corr_matr = df_cleaned.select_dtypes(include=[np.number]).corr()
print(corr_matr)

# %%
df_linear = pd.DataFrame()
df_linear = df_cleaned[['Area_Sqm', 'Dist_Metro_Km', 'Price_K_USD', 'Renovation_code']].copy()
df_linear

# %%
x = df_linear.drop(columns=["Price_K_USD"])
y = df_linear["Price_K_USD"]

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test,y_pred)

mse = mean_squared_error(y_test,y_pred)

rmse = mean_squared_error(y_test,y_pred)**0.5

print(f"Коэффициент детерминации R2: {r2}")
print(f"Средняя квадратичная ошибка MSE: {mse}")
print(f"Корень средней квадратичной ошибки RMSE: {rmse}")

print("\n")

for feature, coef in zip(x.columns, model.coef_):
    print(f"Признак '{feature}': {coef:.2f}")

print(f"Свободный член (Intercept): {model.intercept_:.2f}")

print(f"Итоговая формула для линейной регрессии:\n Y = {x.columns[0]}*{model.coef_[0]:.2f} + {x.columns[1]}*({model.coef_[1]:.2f}) + {x.columns[2]}*{model.coef_[2]:.2f} + {model.intercept_:.2f}")

# %%



