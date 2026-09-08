# 🎓 Intern Learning Path Recommendation System

An AI-powered machine learning recommendation system that generates personalized learning paths for interns based on their learning performance and historical interactions.

The system uses **Collaborative Filtering** and **Singular Value Decomposition (SVD)** to predict suitable learning modules for individual interns.

---

## 🚀 Project Overview

Interns have different learning histories and performance levels, so a single learning path may not be equally suitable for everyone.

This project provides personalized learning module recommendations by analyzing historical assessment data. The recommendation engine predicts the most relevant modules for an intern and generates a ranked learning roadmap.

---

## ✨ Features

- 🎯 Personalized learning path recommendations
- 🤖 Collaborative Filtering recommendation system
- 🧠 Singular Value Decomposition (SVD)
- 📊 Learning performance and interaction analysis
- ⭐ Predicted module compatibility scores
- 🗺️ Ranked learning roadmap generation
- 📚 Completed module history
- ⏱️ Estimated module duration
- 🌐 Interactive Streamlit web application

---

## 🧠 Machine Learning Approach

The recommendation system uses **Collaborative Filtering with Singular Value Decomposition (SVD)**.

The training process:

1. Loads assessment data.
2. Merges assessment and performance information.
3. Removes records with missing scores.
4. Calculates average scores for each intern and learning module.
5. Converts scores into ratings on a scale of 1 to 5.
6. Trains an SVD recommendation model.
7. Predicts suitable modules that an intern has not previously completed.

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Surprise
- Pillow

---

## 📂 Project Structure

```text
intern-learning-path-recommender/
│
├── assets/
│   └── logo.png
│
├── data/
│   ├── raw/
│   │   ├── assessments.csv
│   │   ├── courses.csv
│   │   └── studentAssessment.csv
│
│   └── processed/
│       └── processed_interactions.csv
│
├── models/
│   └── model.pkl
│
├── src/
│   └── train_model.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/talalkazi/intern-learning-path-recommender.git
```

Navigate to the project directory:

```bash
cd intern-learning-path-recommender
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Model

Run:

```bash
python src/train_model.py
```

This generates the trained recommendation model and processed interaction data.

---

## ▶️ Run the Application

Run:

```bash
streamlit run app.py
```

The application will open in your web browser.

---

## 📊 Recommendation Model

The project uses **Singular Value Decomposition (SVD)** to learn hidden relationships between interns and learning modules.

The model predicts a compatibility score for modules that an intern has not previously completed and ranks them to generate a personalized learning path.

---

## 🎯 Future Improvements

- Hybrid recommendation system
- Content-based recommendations
- Intern skill profiles
- Course skill and difficulty analysis
- Learning progress tracking
- User authentication
- Online deployment
- Improved recommendation accuracy using additional performance data

---

## 👨‍💻 Author

Developed as part of a Machine Learning Internship project.

---

## 📄 License

This project is created for educational and internship purposes.
