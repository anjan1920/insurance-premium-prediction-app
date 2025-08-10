
# Insurance Premium Prediction

A practical machine learning project to predict insurance premium amounts by applying regression techniques on real-world data. The goal was to convert theoretical knowledge of ML into hands-on experience.

---

## Tech Stack

* Python 3.12.1
* Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Joblib, Urllib3
* Streamlit for interactive UI

---

## Dataset

* **premiums.xlsx** (\~50,000 records) containing features such as age, income, smoking status, dependents, and more.

---

## Key Challenges & Learnings

* **Challenge:** High error rate despite good overall accuracy.
* **Solution:** Detailed error analysis and dataset segmentation by age improved accuracy substantially.
* **Learning:** Highlighted importance of feature relevance and that some data subsets require more or different features for reliable predictions.

---

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/insurance-premium-prediction.git
   cd insurance-premium-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app/main.py
   ```

---

## Live Demo

Try the app live here:
[ Click here ](https://insurance-premium-prediction-app-6rwuqxuzeau7laqentvpce.streamlit.app/)

---

## Requirements

```
python 3.12.1
streamlit
joblib==1.5.1
matplotlib==3.10.5
matplotlib-inline==0.1.7
numpy==2.3.2
pandas==2.3.1
scikit-learn==1.7.1
seaborn==0.13.2
urllib3==2.5.0
xgboost==3.0.3
```

---

## Notes

I have documented my entire workflow in projectFlow.txt. If you use this project, feel free to drop me a message on LinkedIn—I'd love to connect, improve this work together, and learn more. Also, don’t hesitate to try different models; with XGBoost, you can achieve nearly 99% accuracy

---

## Author

**Anjan Das**
[LinkedIn](www.linkedin.com/in/anjan-das-22b278236) 

---

