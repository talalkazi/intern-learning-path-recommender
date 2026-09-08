import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "processed_interactions.csv"
COURSES_PATH = BASE_DIR / "data" / "raw" / "courses.csv"
LOGO_PATH = BASE_DIR / "assets" / "logo.png"

st.set_page_config(
    page_title="Intern Path Recommender",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.stApp {
    background-color: #F8FAFC;
}

[data-testid="stSidebar"] {
    display: none;
}

h1, h2, h3, h4 {
    color: #045A93 !important;
    font-weight: 700;
}

.stExpander {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
}

.stExpander details summary p {
    color: #045A93 !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
}

.stExpander details summary svg {
    fill: #045A93 !important;
}

.control-card {
    background-color: #FFFFFF;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
}

.stat-card {
    background-color: #FFFFFF;
    border-radius: 8px;
    padding: 12px 16px;
    border: 1px solid #E2E8F0;
    text-align: center;
}

.stat-label {
    color: #64748B;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
}

.stat-value {
    color: #045A93;
    font-size: 1.2rem;
    font-weight: 700;
}

.step-card {
    background-color: #FFFFFF;
    border-radius: 12px;
    padding: 20px;
    border-left: 6px solid #045A93;
    border-top: 1px solid #E2E8F0;
    border-right: 1px solid #E2E8F0;
    border-bottom: 1px solid #E2E8F0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    margin-bottom: 15px;
}

div.stButton > button:first-child {
    background-color: #045A93 !important;
    color: #FFFFFF !important;
    border-radius: 8px;
    border: none;
    width: 100%;
    font-weight: 600;
    padding: 0.6rem 1rem;
    margin-top: 25px;
}

div.stButton > button:first-child:hover {
    background-color: #03436E !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    courses_df = pd.read_csv(COURSES_PATH)
    return df, courses_df


try:
    model = load_model()
    df, courses_df = load_data()

    all_interns = sorted(df["intern_id"].unique())
    all_courses = sorted(df["course_id"].unique())

    col_title_1, col_title_2 = st.columns([1, 15])

    with col_title_1:
        if LOGO_PATH.exists():
            st.image(
                Image.open(LOGO_PATH),
                width=65
            )

    with col_title_2:
        st.markdown(
            "<h1 style='margin:0; padding-top:2px;'>Intern Learning Path Recommendation System</h1>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p style='color: #64748B; margin:0;'>AI-Driven Collaborative Filtering Engine powered by Singular Value Decomposition (SVD)</p>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container():
        st.markdown(
            "<div class='control-card'>",
            unsafe_allow_html=True
        )

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(
            [2, 2, 1.5]
        )

        with ctrl_col1:
            selected_intern = st.selectbox(
                "Select Intern ID:",
                all_interns
            )

        with ctrl_col2:
            top_n = st.selectbox(
                "Max Recommendations:",
                options=[1, 2, 3, 4, 5],
                index=2
            )

        with ctrl_col3:
            generate_btn = st.button(
                "Generate Learning Path",
                type="primary"
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.markdown(
            f"""
            <div class='stat-card'>
                <div class='stat-label'>Total Interns</div>
                <div class='stat-value'>{len(all_interns):,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with stat_col2:
        st.markdown(
            f"""
            <div class='stat-card'>
                <div class='stat-label'>Available Modules</div>
                <div class='stat-value'>{len(all_courses)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with stat_col3:
        st.markdown(
            """
            <div class='stat-card'>
                <div class='stat-label'>Model Type</div>
                <div class='stat-value'>Matrix Factorization (SVD)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    completed_count = len(
        df[df["intern_id"] == selected_intern]
    )

    with stat_col4:
        st.markdown(
            f"""
            <div class='stat-card'>
                <div class='stat-label'>Completed Modules</div>
                <div class='stat-value'>{completed_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if generate_btn:

        completed = df[
            df["intern_id"] == selected_intern
        ]

        completed_list = completed[
            "course_id"
        ].tolist()

        unseen_courses = [
            course
            for course in all_courses
            if course not in completed_list
        ]

        predictions = []

        for course_id in unseen_courses:

            prediction = model.predict(
                uid=selected_intern,
                iid=course_id
            )

            predictions.append(
                {
                    "Module Code": course_id,
                    "Predicted Match Score": round(
                        prediction.est,
                        2
                    )
                }
            )

        pred_df = pd.DataFrame(
            predictions
        ).sort_values(
            by="Predicted Match Score",
            ascending=False
        ).head(top_n)

        course_summary = (
            courses_df
            .groupby("code_module")[
                "module_presentation_length"
            ]
            .mean()
            .round()
            .astype(int)
            .reset_index()
        )

        pred_df = pred_df.merge(
            course_summary,
            left_on="Module Code",
            right_on="code_module",
            how="left"
        )

        pred_df = pred_df.rename(
            columns={
                "module_presentation_length":
                "Avg Duration (Days)"
            }
        ).drop(
            columns=["code_module"]
        )

        st.markdown(
            f"## Recommended Roadmap for Intern #{selected_intern}"
        )

        st.markdown(
            "<p style='color: #64748B;'>Sequence optimized by predicted compatibility and engagement performance.</p>",
            unsafe_allow_html=True
        )

        for index, row in pred_df.iterrows():

            step_number = (
                pred_df.index.get_loc(index) + 1
            )

            module_code = row[
                "Module Code"
            ]

            predicted_score = row[
                "Predicted Match Score"
            ]

            duration = row[
                "Avg Duration (Days)"
            ]

            st.markdown(
                f"""
                <div class='step-card'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <span style='background-color: #045A93; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 0.85rem;'>
                                STEP {step_number}
                            </span>
                            <span style='font-size: 1.4rem; font-weight: bold; color: #1E293B; margin-left: 12px;'>
                                Module {module_code}
                            </span>
                        </div>
                        <div style='text-align: right;'>
                            <span style='font-size: 1.1rem; font-weight: bold; color: #045A93;'>
                                Predicted Score: {predicted_score} / 5.0
                            </span>
                            <span style='color: #64748B; margin-left: 15px;'>
                                Est. Duration: {duration} Days
                            </span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if not completed.empty:

            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )

            with st.expander(
                "View Intern Completed Module History"
            ):

                history_df = completed[
                    ["course_id", "rating"]
                ].rename(
                    columns={
                        "course_id":
                        "Module Code",

                        "rating":
                        "Past Score (1-5)"
                    }
                )

                st.dataframe(
                    history_df,
                    hide_index=True,
                    use_container_width=True
                )

    else:

        st.info(
            "Select an Intern ID and configuration above, then click 'Generate Learning Path' to render the personalized roadmap."
        )

except FileNotFoundError:

    st.error(
        "Required project files were not found. Please make sure the model, datasets, and logo are placed in the correct folders."
    )

except Exception as error:

    st.error(
        f"An unexpected error occurred: {error}"
    )