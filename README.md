# Life Expectancy Prediction Model

This project aims to predict life expectancy using a machine learning model based on health-related and socio-economic data. The implementation involves extensive data preprocessing, visualization, and regression model building. Below is a comprehensive description of the methodologies, tools, and technologies used in this project.

---

## 📊 Features and Methodologies

### Data Preprocessing
- **Pandas**: Used for data loading, cleaning, and manipulation.
- **NumPy**: Utilized for numerical computations and data transformations.
- **Handling Non-Numeric Data**: Converted categorical variables into dummy variables using `pd.get_dummies`.
- **Missing Data Handling**:
  - Identified non-numeric columns.
  - Filled missing values with the column mean.
  - Verified the absence of missing values using heatmaps.

### Data Visualization
- **Matplotlib**: Used for visualizing data distributions and results.
- **Seaborn**:
  - Generated a heatmap to visualize missing data.
  - Created scatter plots to explore relationships between features and the target variable (`Life expectancy`).

### Machine Learning
- **XGBoost (XGBRegressor)**: A gradient boosting algorithm used for regression tasks.
  - Parameters:
    - Objective: `reg:squarederror` for regression.
    - Learning Rate: `0.1`.
    - Maximum Depth: `10`.
    - Number of Estimators: `100`.
- **Train-Test Split**: Split data into training (70%) and testing (30%) subsets using Scikit-learn.

### Model Evaluation
- **Scikit-learn Metrics**:
  - `mean_squared_error` (MSE)
  - `mean_absolute_error` (MAE)
  - `r2_score` (R²)
  - Calculated Root Mean Squared Error (RMSE) and Adjusted R² for deeper insights into model performance.

---

## ⚙️ Tools and Technologies

### Programming Languages
- **Python**: Used for data preprocessing, visualization, and machine learning.

### Libraries and Frameworks
- **Pandas**: For data handling and preprocessing.
- **NumPy**: For numerical operations.
- **Matplotlib** and **Seaborn**: For data visualization.
- **XGBoost**: For building the regression model.
- **Scikit-learn**: For model evaluation and splitting data.

---

## 💻 Project Workflow

### Preprocessing Steps
1. Identified non-numeric columns:
   ```python
   non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
   print("Non-numeric columns:", non_numeric_cols)
   ```
2. Converted categorical variables to numeric:
   ```python
   df = pd.get_dummies(df, columns=['Status'])
   ```
3. Handled missing values:
   ```python
   df = df.apply(lambda x: x.fillna(x.mean()), axis=0)
   ```
4. Verified the absence of missing data:
   ```python
   sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='Blues')
   ```

### Splitting Data
- Defined `X` (features) and `y` (target):
  ```python
  X = df.drop(columns=['Life expectancy '])
  y = df['Life expectancy ']
  ```
- Converted to NumPy arrays:
  ```python
  X = np.array(X).astype('float32')
  y = np.array(y).astype('float32')
  ```
- Split into training and testing sets:
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
  ```

### Model Training and Evaluation
- Trained the XGBoost regressor:
  ```python
  from xgboost import XGBRegressor
  model = XGBRegressor(objective='reg:squarederror', learning_rate=0.1, max_depth=10, n_estimators=100)
  model.fit(X_train, y_train)
  ```
- Evaluated the model:
  ```python
  from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
  from math import sqrt
  
  y_predict = model.predict(X_test)
  MSE = mean_squared_error(y_test, y_predict)
  RMSE = float(format(np.sqrt(MSE), '.3f'))
  MAE = mean_absolute_error(y_test, y_predict)
  r2 = r2_score(y_test, y_predict)
  
  k = X_test.shape[1]
  n = len(X_test)
  adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
  
  print(f'RMSE = {RMSE}\nMSE = {MSE}\nMAE = {MAE}\nR2 = {r2}\nAdjusted R2 = {adj_r2}')
  ```

---

## 📈 Results and Visualizations

| ![Missing Data Heatmap](images/missing_data_heatmap.png) | ![Scatter Plot Example](images/scatter_plot_example.png) |
|:--------------------------------------------------------:|:---------------------------------------------------------:|
| **Missing Data Heatmap**                                 | **Scatter Plot Example**                                  |

| ![Model Performance Metrics](images/performance_metrics.png) |
|:-----------------------------------------------------------:|
| **Model Performance Metrics**                              |

---

## 🛠 Future Enhancements
- Incorporate feature engineering to improve model accuracy.
- Explore hyperparameter tuning using GridSearchCV or Bayesian Optimization.
- Evaluate additional models for comparison, such as Random Forest or Neural Networks.

---

This project demonstrates the potential of machine learning in predicting life expectancy, providing valuable insights for health-related analytics.
