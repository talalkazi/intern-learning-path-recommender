import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from surprise import Dataset, Reader, SVD



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"


def prepare_data():
    """
    Load and preprocess student assessment data to create
    intern-course interaction ratings.
    """

    student_assess = pd.read_csv(
        RAW_DATA_DIR / "studentAssessment.csv"
    )

    assessments = pd.read_csv(
        RAW_DATA_DIR / "assessments.csv"
    )

    
    df = student_assess.merge(
        assessments,
        on="id_assessment"
    )

    
    df = df.dropna(subset=["score"])

    
    user_course_matrix = (
        df.groupby(["id_student", "code_module"])["score"]
        .mean()
        .reset_index()
    )


    user_course_matrix["rating"] = (
        np.ceil(user_course_matrix["score"] / 20.0)
        .clip(1, 5)
    )

    
    user_course_matrix = user_course_matrix.rename(
        columns={
            "id_student": "intern_id",
            "code_module": "course_id"
        }
    )

    return user_course_matrix[
        ["intern_id", "course_id", "rating"]
    ]


def train_and_save_model():
    """
    Train an SVD collaborative filtering model and save
    both the trained model and processed interaction data.
    """

    
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    
    data_df = prepare_data()

    
    reader = Reader(
        rating_scale=(1, 5)
    )

    
    data = Dataset.load_from_df(
        data_df[
            ["intern_id", "course_id", "rating"]
        ],
        reader
    )

    
    trainset = data.build_full_trainset()

    
    model = SVD(
        n_factors=20,
        n_epochs=20,
        random_state=42
    )

    
    model.fit(trainset)

    
    model_path = MODEL_DIR / "model.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    
    processed_path = (
        PROCESSED_DATA_DIR /
        "processed_interactions.csv"
    )

    data_df.to_csv(
        processed_path,
        index=False
    )

    print("Model training completed successfully!")
    print(f"Model saved to: {model_path}")
    print(
        f"Processed data saved to: "
        f"{processed_path}"
    )


if __name__ == "__main__":
    train_and_save_model()