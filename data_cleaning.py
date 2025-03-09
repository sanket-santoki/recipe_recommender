import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import ast

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Data Cleaning & Preprocessing
def clean_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    
    # Handle missing values
    df = df.dropna().reset_index(drop=True)
    
    # Convert 'ingredients_list' from string to list safely
    def parse_ingredients(ing):
        try:
            return ast.literal_eval(ing)
        except (ValueError, SyntaxError):
            return []
    df['ingredients_list'] = df['ingredients_list'].apply(parse_ingredients)
    
    # Remove rows with empty ingredient lists
    df = df[df['ingredients_list'].apply(lambda x: len(x) > 0)].reset_index(drop=True)
    
    # Convert multiword ingredients: replace spaces with hyphens in each ingredient string.
    df['ingredients_list'] = df['ingredients_list'].apply(lambda lst: [x.replace(" ", "-") for x in lst])
    
    return df

# Feature Extraction
def extract_features(df):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, token_pattern=None)
    ingredient_vectors = vectorizer.fit_transform(df['ingredients_list'])
    return ingredient_vectors, vectorizer

# Train-Test Split
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    file_path = "recipe_details.csv"
    df = load_data(file_path)
    df = clean_data(df)
    X, vectorizer = extract_features(df)
    y = df['recipe_name']  # Target variable for recommendations
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Save processed data
    df.to_csv("cleaned_recipe_data.csv", index=False)
    print("Data preprocessing completed and saved.")
