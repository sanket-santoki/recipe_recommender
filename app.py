import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------- Utility Functions ----------

def tokenize_text(text):
    """
    Tokenizes a text string into a set of lowercase word tokens.
    E.g., "gallon whole milk" -> {"gallon", "whole", "milk"}
    """
    return set(re.findall(r'\w+', text.lower()))

def get_recipe_ingredients(ing):
    """
    Convert the ingredients field to a list.
    First tries ast.literal_eval; if that fails, splits by comma.
    """
    import ast
    if isinstance(ing, list):
        return ing
    if isinstance(ing, str):
        try:
            parsed = ast.literal_eval(ing)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
            else:
                return [str(parsed)]
        except Exception:
            return [item.strip() for item in ing.split(",") if item.strip()]
    return []

def predict_recipes(ingredients_input, recipe_model, df):
    """
    Given an ingredients string (comma-separated), use the model to find candidate recipes.
    Returns a list of candidate dictionaries.
    """
    # Process user input.
    user_list = [s.strip().lower() for s in ingredients_input.split(",") if s.strip()]
    if not user_list:
        return []
    user_set = set(user_list)
    user_text = " ".join(user_list)
    
    try:
        # Increase candidate pool for better filtering.
        distances, indices = recipe_model.kneighbors([user_text], n_neighbors=20)
    except Exception as e:
        logger.error("Error during model inference: %s", e)
        return []
    
    candidates = []
    distances = distances[0]
    indices = indices[0]
    for dist, idx in zip(distances, indices):
        row = df.iloc[idx]
        recipe_ingredients = get_recipe_ingredients(row["ingredients_list"])
        normalized_recipe_ings = [ing.strip().lower() for ing in recipe_ingredients]
        
        # First, try for an exact match.
        exact_matches = [ing for ing in normalized_recipe_ings if ing in user_set]
        # If no exact match, try substring match.
        similar_matches = [ing for ing in normalized_recipe_ings if any(token in ing for token in user_set)]
        
        if exact_matches:
            available = exact_matches
            is_exact = True
        elif similar_matches:
            available = similar_matches
            is_exact = False
        else:
            continue
        
        missing = list(set(normalized_recipe_ings) - set(available))
        similarity = round(1 - dist, 2)
        candidate = {
            "recipe_id": int(row["recipe_id"]),
            "recipe_name": row["recipe_name"],
            "aver_rate": float(row["aver_rate"]),
            "image_url": row["image_url"],
            "review_nums": int(row["review_nums"]),
            "calories": float(row["calories"]),
            "fat": float(row["fat"]),
            "carbohydrates": float(row["carbohydrates"]),
            "protein": float(row["protein"]),
            "cholesterol": float(row["cholesterol"]),
            "sodium": float(row["sodium"]),
            "fiber": float(row["fiber"]),
            "ingredients_list": recipe_ingredients,
            "available_ingredients": available,
            "missing_ingredients": missing,
            "similarity": similarity,
            "missing_count": len(missing),
            "available_count": len(available),
            "is_exact": is_exact
        }
        candidates.append(candidate)
    
    # If any candidate has an exact match, filter out others.
    if any(cand["is_exact"] for cand in candidates):
        candidates = [cand for cand in candidates if cand["is_exact"]]
    
    # Sort candidates by descending similarity, then by fewer missing ingredients.
    candidates.sort(key=lambda c: (-c["similarity"], c["missing_count"]))
    # Return only the top 5 results.
    return candidates[:5]

# ---------- Global Loading of Model and Data ----------
@st.cache_resource(show_spinner=False)
def load_model_and_data():
    try:
        model = joblib.load("recipe_recommender_model.pkl")
        data = joblib.load("recipe_data.pkl")
        logger.info("Model and data loaded successfully.")
        return model, data
    except Exception as e:
        logger.error("Error loading model or data: %s", e)
        return None, None

recipe_model, df = load_model_and_data()

# ---------- Streamlit Page Layout ----------

# Set page config.
st.set_page_config(page_title="MyRecipe", page_icon="🍲", layout="wide")

# Sidebar navigation.
page = st.sidebar.selectbox("Navigation", ["Home", "Prediction", "About"])

# ---------- Home Page ----------
if page == "Home":
    st.title("Welcome to MyRecipe")
    st.image("static/logo.png", width=100)
    st.markdown("""
    ### About MyRecipe
    MyRecipe is a cutting-edge recipe recommendation system that leverages advanced machine learning techniques to suggest delicious recipes based on the ingredients you have.

    **Key Features:**
    - **Data:** Trained on a dataset of over **48,000 recipes** with detailed nutritional information, ingredient lists, ratings, and reviews.
    - **Model:** Utilizes advanced methods like TF‑IDF with SVD and modern embedding techniques. Hyperparameter tuning and cross‑validation ensure robust, stable performance.
    - **User-Friendly:** Easily enter your available ingredients to get personalized recipe recommendations.
    
    Explore the app by selecting **Prediction** from the sidebar to get started, or learn more about the developer in the **About** section.
    """)
    
    st.markdown("---")
    st.subheader("System Details")
    st.markdown("""
    - **Repository:** [GitHub Repository](https://github.com/sanket-santoki/recipe_recommender)
    - **Model Training:** The model was trained using a custom pipeline that includes data cleaning, feature extraction with TF‑IDF (and optional SVD), and K‑nearest neighbors for recommendation.
    - **Dataset:** Over 48,000 recipes with detailed information.
    - **Deployment:** This app is deployed using a production‑ready WSGI server.
    """)
    
# ---------- Prediction Page ----------
elif page == "Prediction":
    st.title("Recipe Predictor")
    st.markdown("Enter the ingredients you have (separated by commas) to get personalized recipe recommendations.")
    
    # Option: Use a text input (or you could also use st.multiselect with available ingredients from /ingredients)
    user_input = st.text_input("Ingredients", placeholder="e.g. milk, cheese, tomato")
    
    if st.button("Get Recommendations"):
        if not user_input:
            st.error("Please enter at least one ingredient.")
        elif recipe_model is None or df is None:
            st.error("Model or data not loaded.")
        else:
            with st.spinner("Fetching recommendations..."):
                results = predict_recipes(user_input, recipe_model, df)
            if not results:
                st.warning("No recommendations found.")
            else:
                st.success("Top recommendations:")
                for rec in results:
                    st.markdown(f"**{rec['recipe_name']}**  \n"
                                f"Rating: {rec['aver_rate']} ({rec['review_nums']} reviews)  \n"
                                f"Calories: {rec['calories']} | Fat: {rec['fat']}g | Carbs: {rec['carbohydrates']}g | Protein: {rec['protein']}g  \n"
                                f"**Available Ingredients:** {', '.join(rec['available_ingredients'])}  \n"
                                f"**Missing Ingredients:** {', '.join(rec['missing_ingredients'])}  \n"
                                f"Similarity Score: {rec['similarity']}  \n"
                                "___")
                    
# ---------- About Page ----------
elif page == "About":
    st.title("About the Developer")
    st.image("static/dev_photo.jpg", width=150)
    st.markdown("""
    **Hello! I'm [Your Name].**

    I am a passionate developer specializing in machine learning and data science. I built **MyRecipe** from scratch using Python, Flask, and advanced ML libraries. This system processes a dataset of over 48,000 recipes and leverages state-of-the-art techniques (such as TF‑IDF with SVD and embedding-based methods) to deliver personalized recipe recommendations.

    **What I did:**
    - Data cleaning and processing of 48,000+ recipes.
    - Designed and implemented a recommendation pipeline with hyperparameter tuning and cross‑validation.
    - Developed and deployed a production‑ready web application for recipe recommendations.

    I love combining technology and culinary creativity to make food discovery fun and easy. Thanks for checking out MyRecipe!
    """)

