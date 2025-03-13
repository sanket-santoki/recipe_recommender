import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend.

import logging
import ast
import os
import io
import base64
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
from recipe_model import RecipeRecommender  # Ensure the model class is available

app = Flask(__name__, static_folder=".", template_folder="templates", static_url_path="")
CORS(app)

# Production logging configuration.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global loading of model artifacts.
try:
    recipe_model = joblib.load("recipe_recommender_model.pkl")
    df = joblib.load("recipe_data.pkl")
    logger.info("Model artifacts loaded successfully.")
except Exception as e:
    logger.error("Error loading model artifacts: %s", e)
    # Set globals to None to avoid undefined errors; production should not run if these fail.
    recipe_model = None
    df = None

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

def exact_match(recipe_set, user_set):
    """
    Returns the set of ingredients that exactly match between the recipe and the user's input.
    """
    return recipe_set.intersection(user_set)

@app.route("/favicon.ico")
def favicon():
    if os.path.exists("favicon.ico"):
        return send_from_directory(".", "favicon.ico", mimetype="image/vnd.microsoft.icon")
    return "", 204

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ingredients", methods=["GET"])
def ingredients():
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500
    ingredients_set = set()
    for _, row in df.iterrows():
        ing_list = get_recipe_ingredients(row.get("ingredients_list", []))
        for ing in ing_list:
            ingredients_set.add(ing.strip().lower())
    return jsonify(sorted(list(ingredients_set)))

@app.route("/recommend", methods=["POST"])
def recommend():
    if recipe_model is None or df is None:
        return jsonify({"error": "Model or data not loaded"}), 500

    data = request.get_json(silent=True)
    if not data or "ingredients" not in data:
        return jsonify({"error": "No ingredients provided"}), 400

    user_input = data["ingredients"].strip().lower()
    if not user_input:
        return jsonify({"error": "No ingredients provided"}), 400

    # Process user input.
    user_list = [s.strip().lower() for s in user_input.split(",") if s.strip()]
    user_set = set(user_list)
    user_text = " ".join(user_list)

    try:
        # Increase candidate pool for better filtering.
        distances, indices = recipe_model.kneighbors([user_text], n_neighbors=20)
    except Exception as e:
        logger.error("Error during model inference: %s", e)
        return jsonify({"error": "Model inference failed"}), 500

    candidates = []
    distances = distances[0]
    indices = indices[0]
    for dist, idx in zip(distances, indices):
        row = df.iloc[idx]
        recipe_ingredients = list(get_recipe_ingredients(row["ingredients_list"]))
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
    
    # Sort by descending similarity and then by fewest missing ingredients.
    candidates.sort(key=lambda c: (-c["similarity"], c["missing_count"]))
    # Return top 5 results.
    candidates = candidates[:5]
    
    return jsonify(candidates)

@app.route("/results")
def results():
    try:
        with open("cv_results.pkl", "rb") as f:
            cv_results = pickle.load(f)
    except Exception as e:
        logger.error("Error loading cv_results: %s", e)
        return "Error loading cv_results"
    
    df_cv = pd.DataFrame(cv_results)
    df_cv.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Graph 1: Bar Plot with Error Bars.
    fig1, ax1 = plt.subplots(figsize=(10,6))
    ax1.errorbar(df_cv.index, df_cv['mean_test_score'], yerr=df_cv['std_test_score'],
                 fmt="", marker="o", ecolor='red', capsize=5, linestyle='-', markersize=8)
    ax1.set_title("Mean CV Score per Candidate with Error Bars")
    ax1.set_xlabel("Candidate Index")
    ax1.set_ylabel("Mean CV Score")
    ax1.grid(True)
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    graph1 = base64.b64encode(buf1.getvalue()).decode('utf-8')
    plt.close(fig1)
    
    # Graph 2: Histogram of Mean CV Scores.
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.histplot(df_cv['mean_test_score'], kde=True, bins=10, color='skyblue', ax=ax2)
    ax2.set_title("Distribution of Mean CV Scores")
    ax2.set_xlabel("Mean CV Score")
    ax2.set_ylabel("Frequency")
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png')
    buf2.seek(0)
    graph2 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close(fig2)
    
    # Graph 3: Scatter Plot of Mean CV Score vs. Mean Fit Time.
    fig3, ax3 = plt.subplots(figsize=(10,6))
    ax3.scatter(df_cv['mean_test_score'], df_cv['mean_fit_time'], color='green', s=100)
    ax3.set_title("Mean CV Score vs. Mean Fit Time")
    ax3.set_xlabel("Mean CV Score")
    ax3.set_ylabel("Mean Fit Time (seconds)")
    ax3.grid(True)
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format='png')
    buf3.seek(0)
    graph3 = base64.b64encode(buf3.getvalue()).decode('utf-8')
    plt.close(fig3)
    
    # Graph 4: Box Plot of CV Scores per Fold (if available).
    if 'split0_test_score' in df_cv.columns:
        folds = []
        num_folds = 0
        for col in df_cv.columns:
            if col.startswith("split") and col.endswith("test_score"):
                num_folds += 1
                folds.append(df_cv[col])
        if num_folds > 0:
            fold_scores_df = pd.DataFrame(folds).T
            fold_scores_df.columns = [f"Fold {i+1}" for i in range(num_folds)]
            fig4, ax4 = plt.subplots(figsize=(10,6))
            sns.boxplot(data=fold_scores_df, ax=ax4)
            ax4.set_title("Box Plot of CV Scores per Candidate (Across Folds)")
            ax4.set_xlabel("Fold")
            ax4.set_ylabel("CV Score")
            buf4 = io.BytesIO()
            fig4.savefig(buf4, format='png')
            buf4.seek(0)
            graph4 = base64.b64encode(buf4.getvalue()).decode('utf-8')
            plt.close(fig4)
        else:
            graph4 = None
    else:
        graph4 = None
        
    # Graph 5: Scatter Plot of SVD Components vs. Mean CV Score (if applicable).
    svd_components = []
    cv_scores = []
    for i, params in enumerate(df_cv['params']):
        if params.get("use_svd", False) and "svd_params" in params:
            svd_components.append(params["svd_params"].get("n_components", None))
            cv_scores.append(df_cv['mean_test_score'][i])
    if svd_components:
        fig5, ax5 = plt.subplots(figsize=(10,6))
        ax5.scatter(svd_components, cv_scores, color='purple', s=100)
        ax5.set_title("SVD Components vs. Mean CV Score")
        ax5.set_xlabel("Number of SVD Components")
        ax5.set_ylabel("Mean CV Score")
        ax5.grid(True)
        buf5 = io.BytesIO()
        fig5.savefig(buf5, format='png')
        buf5.seek(0)
        graph5 = base64.b64encode(buf5.getvalue()).decode('utf-8')
        plt.close(fig5)
    else:
        graph5 = None

    descriptions = {
        "graph1": "Bar Plot: Displays the mean CV score per candidate with error bars representing the standard deviation across folds. It shows both the average performance and the stability of each candidate model.",
        "graph2": "Histogram: Shows the distribution of the mean CV scores across candidate models, indicating how the performance scores are spread out.",
        "graph3": "Scatter Plot: Compares the mean CV score with the mean fit time for each candidate model, highlighting any trade-offs between performance and training speed.",
        "graph4": "Box Plot: (If available) Displays the distribution of CV scores for each candidate across folds, illustrating model stability and variability.",
        "graph5": "Scatter Plot: (Optional) Visualizes the relationship between the number of SVD components and the mean CV score, helping to determine the optimal dimensionality if SVD is used."
    }

    return render_template("results.html", 
                           graph1=graph1, graph2=graph2, graph3=graph3, 
                           graph4=graph4, graph5=graph5,
                           descriptions=descriptions)

if __name__ == "__main__":
    try:
        if os.path.exists("recipe_recommender_model.pkl"):
            recipe_model = joblib.load("recipe_recommender_model.pkl")
        else:
            logger.error("recipe_recommender_model.pkl not found. Exiting.")
            raise FileNotFoundError("recipe_recommender_model.pkl not found")
        df = joblib.load("recipe_data.pkl")
        logger.info("Model artifacts loaded successfully.")
    except Exception as e:
        logger.error("Error loading model artifacts: %s", e)
        raise e
    # Production: debug should be False.
    app.run(host="0.0.0.0", port=5000, debug=False)
