# Hotel Booking Cancellations – Exploratory Data Analysis & Business Insights

This project analyzes booking behavior and cancellation patterns in a large hotel dataset.  
The goal is to understand the key drivers of cancellations and extract actionable insights  
for pricing, policy design, and operational strategy.

---

## 📊 Dataset

**Source:**  
Kaggle – *Hotel Booking Demand*  
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

**Description & Motivation:**  
The dataset contains detailed booking information for city and resort hotels.  
It includes demographics, booking channels, lead time, deposit types, and cancellation outcomes.  
The motivation of the project is to identify structural cancellation drivers and propose  
data-driven strategies for revenue stability.

---

## 🧮 Dataset Size Used in This Project

After filtering out irrelevant columns and performing data cleaning:

- **Rows:** ~119,000  
- **Columns:** ~30–32 (depending on transformations)  
- A subset of the original dataset was used after removing non-informative or empty features.

---

## 🎯 Project Goals / Key Questions

This analysis focuses on two core hypotheses:

1. **H1:** Longer lead time increases the likelihood of cancellation.  
2. **H2:** Deposits reduce cancellations.

Additionally, the project investigates:

- Which market segments and channels drive the highest cancellation risk  
- How “Non-Refund” and deposit types behave in practice  
- Structural differences between Groups, TA/TO, and Online Travel Agencies  
- Operational and revenue impacts of long-lead and high-risk bookings

---

## 🗂 Project Structure

```
hotel-business-problems/
│
├── data/
│   ├── raw/          # Original dataset
│   └── processed/    # Cleaned & transformed datasets
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_transformation.ipynb
│   └── 04_analytics_and_conclusion.ipynb
│
├── reports/
│   └── figures/      # Exported plots for analysis & storytelling
│
├── src/
│   ├── cleaning.py
│   ├── transformation.py
│   └── utils.py      # Helper to load saved figures into notebooks
│
└── README.md
```

---

## ▶️ How to Run the Project

### **Environment Requirements**
- Python **3.10+**
- Jupyter Notebook / JupyterLab
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - pathlib

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Run the Analysis**
1. Open the repository in JupyterLab  
2. Run notebooks in this order:
   - `01_data_exploration.ipynb`
   - `02_cleaning.ipynb`
   - `03_transformation.ipynb`
   - `04_analytics_and_conclusion.ipynb`

3. Figures are automatically saved to `reports/figures/` and can be loaded with:
```python
from src.utils import show_saved_fig
show_saved_fig("your_figure.png")
```

---

## 🧠 Results & Key Insights

- **Lead time is the strongest driver of cancellations** — long-lead bookings cancel disproportionately often, confirming H1.  
- **Deposits do *not* reduce cancellations** — Non-Refund bookings have *higher* cancellation rates, disproving H2.  
- **Groups and Offline TA/TO are the main sources of non-refund cancellations**, not Online Travel Agencies.  
- **Occupancy forecasting must incorporate lead-time risk** to avoid overestimating future demand.  
- **Channel- and segment-specific policies are essential**: stricter contracts for Groups and performance-based agreements for TA/TO dramatically reduce volatility.

---

## 📌 Summary

This project provides a full descriptive analysis of hotel booking dynamics and reveals structural drivers of cancellations.  
The results offer clear business implications for revenue management, pricing, and channel strategy,  
making it a practical foundation for further modeling (e.g., cancellation prediction).

---
