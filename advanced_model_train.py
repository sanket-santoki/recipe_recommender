import pandas as pd
import joblib
import logging
import time
import pickle
from sklearn.model_selection import RandomizedSearchCV, KFold
from joblib import parallel_backend
from recipe_model import RecipeRecommender, preprocess_ingredients

# Set up logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def train_recipe_recommender():
    start_time = time.time()
    logger.info("Loading cleaned data...")
    df = pd.read_csv("cleaned_recipe_data.csv")
    
    # Preprocess ingredients.
    df['ingredients_text'] = df['ingredients_list'].apply(preprocess_ingredients)
    logger.info("Example processed ingredients: %s", df.loc[0, 'ingredients_text'])
    X = df['ingredients_text']

    # Define parameter grid for candidate pipelines.
    param_grid = [
        # TF-IDF candidate.
        {
            "representation": ["tfidf"],
            "vectorizer_params": [{"ngram_range": (1, 2), "min_df": 1}],
            "use_svd": [True, False],
            "svd_params": [{"n_components": 100}],
            "nn_params": [{"n_neighbors": 5, "metric": "cosine"}],
            "use_approx_nn": [False]
        },
        # Embedding candidate.
        {
            "representation": ["embedding"],
            "embedding_model_name": ["all-MiniLM-L6-v2"],
            "nn_params": [{"n_neighbors": 5}],
            "use_approx_nn": [True]
        }
    ]

    # Use 2-fold CV for speed.
    cv = KFold(n_splits=2, shuffle=True, random_state=42)
    base_model = RecipeRecommender()
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=2,  # Two iterations: one candidate from each grid.
        scoring=None,  # Uses estimator.score()
        cv=cv,
        n_jobs=1,
        verbose=1,
        random_state=42
    )

    logger.info("Starting hyperparameter tuning and cross-validation (RandomizedSearchCV)...")
    with parallel_backend('threading', n_jobs=1):
        random_search.fit(X)
    
    # Log detailed candidate performance.
    cv_results = random_search.cv_results_
    df_cv = pd.DataFrame(cv_results)
    num_candidates = len(df_cv)
    logger.info("Hyperparameter tuning complete. Displaying candidate scores:")
    for i in range(num_candidates):
        mean_score = df_cv.loc[i, "mean_test_score"]
        std_score = df_cv.loc[i, "std_test_score"]
        params = df_cv.loc[i, "params"]
        logger.info("Candidate %d: Mean CV Score = %.4f (Std = %.4f), Parameters = %s", 
                    i, mean_score, std_score, params)
    
    # Log best candidate information.
    best_score = random_search.best_score_
    best_params = random_search.best_params_
    logger.info("Best CV Score: %.4f", best_score)
    logger.info("Best Parameters: %s", best_params)
    
    # Refit the best estimator on the full dataset.
    logger.info("Refitting the best model on the full dataset...")
    best_model = random_search.best_estimator_.fit(X)
    
    # Save the best model, data, and cv_results for later visualization.
    logger.info("Saving the best model and data...")
    joblib.dump(best_model, "recipe_recommender_model.pkl")
    joblib.dump(df, "recipe_data.pkl")
    with open("cv_results.pkl", "wb") as f:
        pickle.dump(cv_results, f)
    
    total_time = time.time() - start_time
    logger.info("All components saved successfully. Total training time: %.2f seconds", total_time)

if __name__ == "__main__":
    train_recipe_recommender()
