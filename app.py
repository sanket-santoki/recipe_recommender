import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------- Utility Functions ----------------

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

def get_unique_ingredients(df):
    """
    Get a sorted list of unique ingredient tokens from the DataFrame.
    """
    unique_tokens = set()
    for ing_field in df["ingredients_list"]:
        for ing in get_recipe_ingredients(ing_field):
            unique_tokens.update(tokenize_text(ing))
    return sorted(unique_tokens)

def predict_recipes(ingredients_input, model, df):
    """
    Given a comma-separated ingredients input, return top candidate recipes.
    """
    # Process user input.
    user_list = [s.strip().lower() for s in ingredients_input.split(",") if s.strip()]
    if not user_list:
        return []
    user_set = set(user_list)
    user_text = " ".join(user_list)
    
    try:
        distances, indices = model.kneighbors([user_text], n_neighbors=20)
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
        # Try for exact matches first.
        exact_matches = [ing for ing in normalized_recipe_ings if ing in user_set]
        # Else, try substring match.
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
    
    # Sort candidates by descending similarity and fewest missing ingredients.
    candidates.sort(key=lambda c: (-c["similarity"], c["missing_count"]))
    return candidates[:5]

def render_recipe_card(recipe):
    """
    Returns HTML for a styled recipe card.
    """
    available = ", ".join(recipe["available_ingredients"]) if recipe["available_ingredients"] else "None"
    missing = ", ".join(recipe["missing_ingredients"]) if recipe["missing_ingredients"] else "None"
    card_html = f"""
    <div class="card">
      <div class="card-header">
        <h3>{recipe['recipe_name']}</h3>
        <p class="rating">Rating: {recipe['aver_rate']} ({recipe['review_nums']} reviews)</p>
      </div>
      <div class="card-body">
        <p><strong>Nutrition:</strong> Calories: {recipe['calories']}, Fat: {recipe['fat']}g, Carbs: {recipe['carbohydrates']}g, Protein: {recipe['protein']}g</p>
        <p><strong>Available:</strong> {available}</p>
        <p><strong>Missing:</strong> {missing}</p>
        <p><strong>Similarity:</strong> {recipe['similarity']}</p>
      </div>
    </div>
    """
    return card_html

# ---------------- Global Loading of Model and Data ----------------
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

# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="MyRecipe", page_icon="🍲", layout="wide")

# Inject CSS for styling cards and animations.
st.markdown("""
<style>
/* Overall styling */
body { background-color: #f1f4f8; }
/* Recipe card styling */
.card {
  background-color: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 16px;
  margin: 16px 0;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 12px rgba(0,0,0,0.15);
}
.card-header h3 {
  margin-bottom: 4px;
  font-size: 22px;
  color: #333;
}
.rating {
  color: #f39c12;
  font-size: 14px;
}
.card-body p {
  margin: 6px 0;
  font-size: 16px;
  color: #555;
}
/* Navigation Sidebar styling */
.sidebar .stRadio label {
  font-size: 16px;
}
/* Page Header */
.page-title {
  font-size: 32px;
  margin-bottom: 20px;
  color: #333;
}
/* About page styling */
.about-content {
  font-size: 18px;
  line-height: 1.6;
  color: #333;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Home", "Prediction", "About"])

# ---------------- Home Page ----------------
if page == "Home":
    st.title("Welcome to MyRecipe")
    st.markdown("""
    MyRecipe is a state-of-the-art recipe recommendation system that leverages advanced machine learning techniques to suggest delicious recipes based on the ingredients you have.  
    **Key Features:**  
    - Over **48,000 recipes** with detailed nutritional info and ingredient lists.
    - Uses advanced methods like TF‑IDF with SVD and embedding techniques.
    - Hyperparameter tuning and cross‑validation ensure robust and stable recommendations.
    """)
    st.markdown("---")
    st.subheader("System Details")
    st.markdown("""
    **Repository:** [GitHub Repository](https://github.com/sanket-santoki/recipe_recommender)  
    **Model Training:** The model was built using Python, Flask, and advanced ML libraries. It includes data cleaning, feature extraction, and K‑nearest neighbors for recommendation.  
    **Dataset:** Over 48,000 recipes.  
    **Deployment:** This app is deployed using a production‑ready WSGI server.
    """)

# ---------------- Prediction Page ----------------
elif page == "Prediction":
    st.title("Recipe Predictor")
    st.markdown("Enter the ingredients you have to get personalized recipe recommendations.")
    
    # Create a multiselect dropdown from available ingredients.
    if df is not None:
        available_ings = get_unique_ingredients(df)
    else:
        available_ings = []
    
    selected_ingredients = st.multiselect("Select Ingredients", options=available_ings)
    
    if st.button("Get Recommendations"):
        if not selected_ingredients:
            st.error("Please select at least one ingredient.")
        elif recipe_model is None or df is None:
            st.error("Model or data not loaded.")
        else:
            with st.spinner("Fetching recommendations..."):
                results = predict_recipes(", ".join(selected_ingredients), recipe_model, df)
            if not results:
                st.warning("No recommendations found.")
            else:
                st.success("Top recommendations:")
                # Render each recipe card using columns for a responsive layout.
                for rec in results:
                    st.markdown(render_recipe_card(rec), unsafe_allow_html=True)

# ---------------- About Page ----------------
elif page == "About":
    st.title("About the Developer")
    st.markdown("""
    **Developer:** [Nidhi Pokiya]

    I am a passionate developer specializing in machine learning and data science. I built **MyRecipe** from scratch using Python, Streamlit, and a suite of advanced ML libraries.  
    This system was developed by processing a dataset of over **48,000 recipes**, and building a robust recommendation pipeline using techniques like TF‑IDF with SVD and embedding-based models. Hyperparameter tuning and cross‑validation ensured a stable, high‑performance model.  
    I enjoy combining technology with culinary creativity to help users discover new recipes. Thank you for using MyRecipe!
    """)

