# 📊 Occupational Value (OVP) Dashboard

The **Occupational Value (OVP) Dashboard** is an interactive Streamlit web app that helps users explore the motivational structure of work.
It identifies the **Occupational Value Profile (OVP)** for a single occupation and compares it against four benchmark occupations.

---

## 🔎 Purpose
Work is more than tasks — it is a structure that permits or restricts the satisfaction of core psychological needs.
The OVP Dashboard provides insight into how well a given occupation supports these six work values from the **Theory of Work Adjustment (TWA)**:

- **Achievement**
- **Independence**
- **Recognition**
- **Relationships**
- **Support**
- **Working Conditions**

---

## ⚙️ Features
- Enter a **job title** and match it to the closest O*NET occupation.
- Display the occupation’s **OVP scores** across six dimensions.
- Compare results to four benchmark occupations:
  - Chief Executives (11-1011.00)
  - Nearest Mid-High occupation
  - Participant (selected occupation)
  - Nearest Mid-Low occupation
  - Graders & Sorters, Agricultural Products (45-2041.00)
- Clean, interactive interface powered by **Streamlit** and **Altair**.

---

## 🚀 Getting Started

### 1) Clone this repository
```bash
git clone https://github.com/WorkLocomotion/ovp-dashboard.git
cd ovp-dashboard
```

### 2) Install the required packages
```bash
pip install -r requirements.txt
```

### 3) Run the dashboard
```bash
streamlit run ovp_dashboard.py
```

### 4) Open the app in your browser
```
http://localhost:8501
```

> Requires Python 3.10+ (3.11 recommended).

---

## 🌐 Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io
3. Create a **New app** and select:
   - **Repo**: `WorkLocomotion/ovp-dashboard`
   - **Branch**: `main`
   - **Main file path**: `ovp_dashboard.py`
4. Click **Deploy**.

The app will be available at a public URL (for example, `https://<your-handle>-ovp-dashboard.streamlit.app`).

---

## 📂 Repository Structure
```
ovp-dashboard/
├─ ovp_dashboard.py         # main Streamlit app
├─ requirements.txt         # dependencies
├─ README.md
├─ data/
│  ├─ onet_work_values.xlsx
│  └─ ovp_title_index.xlsx
└─ .streamlit/
   └─ config.toml           # optional theming
```

---

## 📘 Background
This dashboard is part of **Work Locomotion**, integrating:
- **Occupational Value Profiles (OVP)**
- The **Theory of Work Adjustment (TWA)**
- **Holland’s Interest Hexagon**

Aim: redesign work to improve motivation, retention, and flourishing, particularly in the skilled trades.

---

## 🔗 Connect
- Substack: https://worklocomotion.substack.com/
- LinkedIn: https://www.linkedin.com/in/YOUR-HANDLE/

---

## 📜 License
Licensed under the MIT License. See [LICENSE](LICENSE) for details.
