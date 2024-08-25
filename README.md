# Cross-Border Transaction Optimization Using Machine Learning Models

## Project Overview

This project aims to develop an AI-based solution to optimize cross-border money transfers. The goal is to select the most optimal solution for each transaction by minimizing both time and cost. The project explores various machine learning models, including Gradient Boosting Machines (GBM), Deep Learning (DL), and Decision Trees, to predict the best solution option for each transaction based on relevant features such as transaction fees, time, and currency information. The dataset used for this project is synthetically generated but follows real-world patterns to simulate cross-border transaction scenarios.

## Dataset

The dataset contains the following features:
- **Solution option name:** The name of the solution option used for the transaction (e.g., SWIFT, SEPA, PayPal, Western Union).
- **Time taken to complete step:** The time taken to complete the transaction in minutes.
- **Cost of transaction fee:** The fee charged for the transaction.
- **Cost of additional fee:** Any additional fees incurred during the transaction.
- **Sender’s Currency:** The currency of the sender.
- **Sender’s country:** The country of the sender.
- **Recipient’s currency:** The currency of the recipient.
- **Recipient’s country:** The country of the recipient.
- **Transaction amount:** The amount of money sent.
- **Settled amount:** The amount of money received after deducting fees.
- **Cost Ratio, Time Cost Interaction, Transaction Efficiency:** Derived features to capture the relationship between time, cost, and efficiency.

## Project Structure

### 1. Data Preparation
The data is loaded from a CSV file, and categorical variables are encoded using `LabelEncoder`. Features are selected based on their relevance to the task, and the data is scaled using `StandardScaler`.

### 2. Model Implementation
Three primary models are implemented:
- **Gradient Boosting Machine (GBM) using XGBoost:**
  - This model is trained using XGBoost, which is an efficient and scalable implementation of gradient boosting. The model is evaluated for accuracy and its ability to predict the best solution option for each transaction.
  
- **Deep Learning Model using TensorFlow:**
  - A fully connected neural network is built using TensorFlow and Keras. The model is designed with three hidden layers, each followed by a dropout layer to prevent overfitting. The output layer uses the softmax activation function to classify the transactions.

- **Ensemble Model:**
  - A stacking ensemble model is implemented using XGBoost and a Multi-Layer Perceptron (MLP) as base learners. Logistic Regression is used as the final estimator to combine the predictions of the base models. This approach leverages the strengths of multiple models to improve overall performance.

### 3. Feature Engineering
Additional features are engineered to capture more complex relationships within the data:
- **Transaction Amount to Fee Ratio:** The ratio of the transaction amount to the total fees.
- **Time and Fee Product:** The product of time taken and total fees.
- **Log Transformed Features:** Logarithmic transformations of certain features to reduce skewness.
- **Interaction Features:** Interaction terms between sender and recipient currencies, as well as polynomial features of key variables.

### 4. Model Tuning and Evaluation
Grid search is used to tune the hyperparameters of the XGBoost model, ensuring the best performance. The models are evaluated using accuracy, precision, recall, and F1-score. The results are compared to determine the best-performing model for this task.

### 5. Model Interpretation using LIME
LIME (Local Interpretable Model-Agnostic Explanations) is used to explain the predictions of the best XGBoost model. This helps in understanding which features contribute the most to the decision-making process.

### 6. Decision Tree Implementation
A Decision Tree model is also implemented to provide an interpretable alternative to the more complex models. The Decision Tree model is evaluated using the same metrics.

## How to Run the Code

1. **Prerequisites:**
   - Python 3.7 or higher
   - Required Python libraries:
     - pandas
     - numpy
     - scikit-learn
     - xgboost
     - tensorflow
     - imbalanced-learn
     - lime
     - matplotlib

   Install the required libraries using the following command:
   ```bash
   pip install pandas numpy scikit-learn xgboost tensorflow imbalanced-learn lime matplotlib
   ```

2. **Running the Code:**
   - Load the dataset and run each section of the code sequentially.
   - Start with data preparation, followed by model implementation, feature engineering, and finally, model evaluation.

3. **Dataset:**
   - Ensure the dataset `Enhanced_Patterned_Cross_Border_Transactions.csv` is in the same directory as the code.

4. **Output:**
   - The models will output accuracy, classification reports, and feature importance plots. The LIME explanations will provide insights into how specific predictions are made.

## Model Performance

- **XGBoost Model:**
  - The XGBoost model achieved a high accuracy due to its ability to handle complex relationships in the data. After hyperparameter tuning, the model showed strong performance in predicting the optimal transaction solution.

- **Deep Learning Model:**
  - The deep learning model, while powerful, required more data and careful tuning to match the performance of XGBoost. However, it still provided a strong baseline for comparison.

- **Ensemble Model:**
  - The ensemble model, combining XGBoost and MLP, showed the best performance by leveraging the strengths of both models. It outperformed individual models in terms of accuracy and stability.

- **Decision Tree Model:**
  - The Decision Tree model provided a more interpretable but slightly less accurate alternative. It was useful for understanding the decision-making process at a granular level.

## Conclusion

The project demonstrates that the Gradient Boosting Machine (GBM) is the best fit for selecting the optimal cross-border transaction solution. Its ability to achieve high accuracy, handle complex feature interactions, and provide feature importance makes it ideal for minimizing both time and cost in financial transactions. The additional feature engineering and ensemble approaches further enhanced the model's performance, making it a robust solution for real-world applications.

### Future Work

- Explore additional feature engineering techniques to capture more complex patterns in the data.
- Test the models on real-world datasets to validate their performance in practical scenarios.
- Consider integrating the model into a production system for real-time transaction optimization.
