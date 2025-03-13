import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import logging
import base64
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------- Utility Functions ----------------

def tokenize_text(text):
    """
    Tokenizes a text string into a set of lowercase word tokens.
    Example: "gallon whole milk" -> {"gallon", "whole", "milk"}
    """
    return set(re.findall(r'\w+', text.lower()))

def get_recipe_ingredients(ing):
    """
    Converts the ingredients field into a list.
    Tries ast.literal_eval first; if that fails, splits by comma.
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
    Returns a sorted list of unique ingredient tokens from the DataFrame.
    Filters out purely numeric tokens.
    """
    unique_set = set()
    for _, row in df.iterrows():
        ings = get_recipe_ingredients(row.get("ingredients_list", []))
        for ing in ings:
            token = ing.strip().lower()
            if token and not token.isdigit():
                unique_set.add(token)
    return sorted(list(unique_set))

def predict_recipes(ingredients_input, model, df):
    """
    Given a comma-separated string of ingredients, uses the loaded model and DataFrame
    to generate a list of recommended recipes.
    Returns a list of candidate recipe dictionaries (top 5).
    """
    user_list = [s.strip().lower() for s in ingredients_input.split(",") if s.strip()]
    if not user_list:
        return []
    user_set = set(user_list)
    user_text = " ".join(user_list)
    
    try:
        # Get a larger candidate pool for better filtering.
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
        # Try for an exact match.
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
    
    # If any candidate has an exact match, filter to keep only exact matches.
    if any(cand["is_exact"] for cand in candidates):
        candidates = [cand for cand in candidates if cand["is_exact"]]
    
    # Sort by descending similarity and then by fewer missing ingredients.
    candidates.sort(key=lambda c: (-c["similarity"], c["missing_count"]))
    return candidates[:5]

def render_recipe_card(recipe):
    """
    Returns a string of HTML representing a styled recipe card.
    Uses inline HTML/CSS for display.
    """
    available = ", ".join(recipe["available_ingredients"]) if recipe["available_ingredients"] else "None"
    missing = ", ".join(recipe["missing_ingredients"]) if recipe["missing_ingredients"] else "None"
    # If image_url is empty, use a placeholder image.
    img_url = recipe["image_url"] if recipe["image_url"] else "https://via.placeholder.com/300x160.png?text=No+Image"
    card_html = f"""
    <div class="card">
      <div class="card-img">
        <img src="{img_url}" alt="{recipe['recipe_name']}">
      </div>
      <div class="card-content">
        <h3>{recipe['recipe_name']}</h3>
        <p class="rating">Rating: {recipe['aver_rate']} ({recipe['review_nums']} reviews)</p>
        <p><strong>Nutrition:</strong> Calories: {recipe['calories']}, Fat: {recipe['fat']}g, Carbs: {recipe['carbohydrates']}g, Protein: {recipe['protein']}g</p>
        <p><strong>Available Ingredients:</strong> {available}</p>
        <p><strong>Missing Ingredients:</strong> {missing}</p>
        <p><strong>Similarity Score:</strong> {recipe['similarity']}</p>
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

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Home", "Prediction", "About"])

# Custom CSS for enhanced design and animations.
st.markdown("""
<style>
/* Overall styling */
body { background-color: #f1f4f8; }
header { background: linear-gradient(135deg, #5a67d8, #4299e1); color: #fff; padding: 20px; text-align: center; }
.header-top { display: flex; align-items: center; justify-content: center; gap: 10px; }
.logo-img { width: 60px; height: 60px; border-radius: 50%; border: 2px solid #fff; }
.sidebar .stRadio label { font-size: 16px; }

/* Card styles for recipe */
.card {
  background-color: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  margin-bottom: 20px;
}
.card:hover { transform: translateY(-5px); box-shadow: 0 8px 12px rgba(0,0,0,0.15); }
.card-img img { width: 100%; height: 160px; object-fit: cover; }
.card-content { padding: 15px; }
.card-content h3 { margin-bottom: 8px; font-size: 22px; color: #333; }
.rating { color: #f39c12; font-size: 14px; margin-bottom: 8px; }
.card-content p { margin-bottom: 8px; font-size: 16px; color: #555; }

/* Home page styles */
.home-section { background: #fff; padding: 30px; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px; }
.home-section h2 { margin-bottom: 20px; font-size: 32px; text-align: center; color: #333; }
.home-section p { margin-bottom: 15px; font-size: 16px; text-align: justify; }

/* Predictor page styles */
.input-container { display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 10px; margin-bottom: 30px; }
#ingredients { width: 100%; max-width: 500px; padding: 10px; font-size: 16px; border: 1px solid #ccc; border-radius: 4px; }
#get-rec-btn { background-color: #4299e1; color: #fff; border: none; padding: 10px 20px; font-size: 16px; border-radius: 4px; cursor: pointer; transition: background-color 0.3s ease; }
#get-rec-btn:hover { background-color: #2b6cb0; }

/* About page styles */
.about-section { background: #fff; padding: 30px; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px; }
.about-section h2 { margin-bottom: 20px; font-size: 32px; text-align: center; color: #333; }
.about-section p { margin-bottom: 15px; font-size: 18px; text-align: justify; color: #333; }

/* Iframe styling for additional pages */
iframe { width: 100%; border: none; }
</style>
""", unsafe_allow_html=True)

# ---------------- Home Page ----------------
if page == "Home":
    st.title("Welcome to MyRecipe")
    st.image("static/logo.png", width=120)
    st.markdown("""
    **MyRecipe** is a state-of-the-art recipe recommendation system that leverages advanced machine learning techniques to suggest delicious recipes based on the ingredients you have.  
    **Key Features:**
    - Over **48,000 recipes** with detailed nutritional and ingredient information.
    - Advanced models built using TF‑IDF with SVD and modern embedding techniques.
    - Hyperparameter tuning and cross‑validation ensure robust, stable recommendations.
    """)
    st.markdown("---")
    st.subheader("System Details")
    st.markdown("""
    - **Repository:** [GitHub Repository](https://github.com/sanket-santoki/recipe_recommender)  
    - **Model Training:** Developed with Python, Flask, and advanced ML libraries; the pipeline includes data cleaning, feature extraction, and a K‑nearest neighbors recommendation engine.  
    - **Dataset:** Over 48,000 recipes.  
    - **Deployment:** Deployed using a production‑ready WSGI server.
    """)
    st.button("Get Started", on_click=lambda: st.session_state.update(page="Prediction"))

# ---------------- Prediction Page ----------------
elif page == "Prediction":
    st.title("Recipe Predictor")
    st.markdown("Select the ingredients you have to get personalized recipe recommendations.")
    
    # Populate ingredients dropdown from the DataFrame.
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
                st.success("Top Recommendations:")
                for rec in results:
                    st.markdown(render_recipe_card(rec), unsafe_allow_html=True)

# ---------------- About Page ----------------
elif page == "About":
    st.title("About the Developer")
    st.markdown("""
    **Developer:** Sanket Santoki

    I am a passionate developer specializing in machine learning and data science. I built **MyRecipe** from scratch using Python, Streamlit, and advanced ML libraries.  
    This system was developed by processing a dataset of over **48,000 recipes** and designing a robust recommendation pipeline using techniques such as TF‑IDF with SVD and embedding-based models. Hyperparameter tuning and cross‑validation ensured high performance and stability.  
    I enjoy blending technology with culinary creativity to help users discover new recipes. Thank you for using **MyRecipe**!
    """)
