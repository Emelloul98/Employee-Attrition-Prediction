Here’s a README file for your Python project in the format you requested:

---

# Employee Attrition Prediction with Feature Selection and Logistic Regression

This project implements a machine learning pipeline that predicts employee attrition using various classifiers, feature selection, and logistic regression. The project processes a dataset containing employee information, applies feature selection methods, and evaluates the performance of different models using cross-validation and bagging techniques.

---

**Description:**

The project focuses on predicting employee attrition using a combination of machine learning algorithms, including logistic regression, decision trees, random forests, and gradient boosting classifiers. It also implements feature selection using mutual information and compares the performance of models based on the selected features. The project leverages K-Fold cross-validation and ensemble methods like bagging to improve the accuracy and robustness of the predictions.

---

**Project Structure:**

1. **Data Preprocessing:**
   - The dataset `data.csv` is preprocessed by removing irrelevant columns like 'Employee ID' and applying label encoding to binary and categorical columns.
   - Numerical features are scaled using `StandardScaler` to normalize the data.

2. **Feature Selection:**
   - Mutual Information (MI) is used to rank the features based on their importance. 
   - Features are selected based on the MI scores, and the model is trained and evaluated for different feature counts.

3. **Model Training:**
   - Logistic Regression is used as the primary model for prediction.
   - The model is evaluated using K-Fold cross-validation to assess its generalization performance across different feature sets.
   - Ensemble methods such as bagging are applied to create a more robust prediction model using multiple logistic regression models.

4. **Performance Evaluation:**
   - Model performance is evaluated using F1 score, accuracy, and confusion matrix.
   - The final model is trained on the selected features, and its performance is tested on a separate test set.

---

**Usage:**

1. **Data Preprocessing:**
   - The dataset is loaded, and binary, categorical, and numerical features are processed.
   - Features are encoded, and numerical values are scaled for optimal model performance.

   ```python
   df = pd.read_csv('data.csv')
   df = df.drop(columns=['Employee ID'])
   ```

2. **Feature Selection:**
   - Select the most informative features using mutual information scores:

   ```python
   selected_features_mi = mi_feature_selection(X, y, 15)
   X_selected = X[:, selected_features_mi]
   ```

3. **Model Training and Evaluation:**
   - Train a logistic regression model using the selected features and evaluate it using cross-validation:

   ```python
   model = LogisticRegression()
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   ```

4. **Ensemble Methods:**
   - Bagging is used to combine predictions from multiple models to improve robustness and accuracy.

   ```python
   final_model = LogisticRegression()
   final_model.fit(X_train, y_new)
   ```

5. **Confusion Matrix and Performance Metrics:**
   - Evaluate the model’s performance on the test set using accuracy, F1 score, and confusion matrix:

   ```python
   test_accuracy = accuracy_score(y_test, y_test_pred)
   conf_matrix = confusion_matrix(y_test, y_test_pred)
   ```

---

**Features:**

- **Data Preprocessing:**
  - Label encoding for binary and categorical features.
  - Standard scaling for numerical features.
  
- **Feature Selection:**
  - Mutual Information-based feature selection to rank and select the most important features for the model.
  
- **Machine Learning Models:**
  - Logistic Regression as the primary model.
  - K-Fold cross-validation for robust performance evaluation.
  - Bagging for ensemble learning and model improvement.
  
- **Performance Metrics:**
  - F1 score, accuracy, confusion matrix for model evaluation.
  
---

**Dependencies:**

The project uses the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `scipy`

To install the required dependencies, use the following command:

```bash
pip install pandas numpy scikit-learn matplotlib scipy
```

---

**Example Results:**

- **F1 Score vs. Number of Features:**
  - Visualization of model performance across different feature counts.

  ![F1 Score vs Features](example_plot.png)

- **Confusion Matrix:**
  - Displaying the performance of the final model on the test set.

  ![Confusion Matrix](confusion_matrix.png)

---
