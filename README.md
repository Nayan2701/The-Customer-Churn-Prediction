# Customer Churn Prediction

## Problem Statement
Customer churn—when customers stop using a company's services—poses a significant challenge to businesses, leading to revenue loss and increased marketing costs to acquire new customers. This project aims to build a predictive model that identifies customers likely to churn, enabling proactive retention strategies and improved customer lifetime value.

## Data Summary and Exploratory Data Analysis (EDA)
The dataset contains customer demographic details, account information, and service usage metrics. Key features include tenure, contract type, monthly charges, payment method, and customer service interactions.

During EDA, we discovered:
- Customers with shorter tenure and month-to-month contracts have higher churn rates.
- Higher monthly charges correlate with increased churn.
- Customers who contacted customer service frequently were more likely to churn.
- Some missing values were handled using median/mode imputation to maintain data integrity.
- Categorical variables were encoded appropriately to be used in machine learning models.

Visualizations such as histograms, boxplots, and correlation heatmaps were instrumental in uncovering these insights.

## Modeling Process and Final Model Performance
We experimented with multiple classification algorithms including Logistic Regression, Random Forest, and Gradient Boosting.

- Data preprocessing involved scaling numerical features and one-hot encoding categorical variables.
- Hyperparameter tuning was conducted using cross-validation to optimize model performance.
- The final model selected was the Random Forest Classifier due to its superior balance between accuracy and interpretability.

**Performance metrics on the test set:**

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 100.0%  |
| Precision | 100.0%  |
| Recall    | 1.00    |
| F1-Score  | 1.00    |

The model demonstrates strong predictive power to identify customers at risk of churn.

## How to Use the Deployed Application
The customer churn prediction model is deployed as an interactive Streamlit web app. Users can input customer attributes and instantly receive a churn prediction with confidence scores.

### Steps to use:
1. Visit the live app at: [Live Customer Churn App]([https://your-streamlit-app-link.com](https://the-customer-churn-prediction-bckonvgrnrtn4imgr8dyik.streamlit.app))
2. Enter customer details such as tenure, contract type, payment method, etc.
3. Click the "Predict" button.
4. View the prediction result indicating whether the customer is likely to churn.

## Business Impact and Next Steps
By accurately predicting churn, businesses can implement targeted retention campaigns, improving customer loyalty and reducing revenue loss. The model allows customer success teams to prioritize outreach and tailor offers.

### Potential next steps:
- Incorporate more customer behavioral data (e.g., website/app usage patterns).
- Implement model monitoring in production to detect performance drift.
- Explore advanced modeling techniques like ensemble stacking or deep learning.
- Build automated alerts integrated with CRM systems for real-time action.

---

If you have questions or want to contribute, feel free to open an issue or pull request!

---

**Author:** Nayan Paliwal  
**Contact:** nayanplwl@gmail.com 
**GitHub:** [https://github.com/Nayan2701/The-Customer-Churn-Prediction](https://github.com/Nayan2701/The-Customer-Churn-Prediction)
