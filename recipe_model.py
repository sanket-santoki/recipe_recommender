import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator
from scipy.sparse import issparse

# Try to import SentenceTransformer for embeddings.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Try to import Annoy for approximate nearest neighbor search.
try:
    from annoy import AnnoyIndex
except ImportError:
    AnnoyIndex = None

def preprocess_ingredients(ing_str):
    """
    Convert a string representation of a list into a plain text string.
    For example:
      "['pork belly', 'smoked paprika', 'kosher salt', 'ground black pepper']"
    becomes:
      "pork belly smoked paprika kosher salt ground black pepper"
    """
    try:
        ing_list = ast.literal_eval(ing_str)
        if isinstance(ing_list, list):
            return " ".join(ing_list)
        else:
            return ""
    except Exception:
        return ""

class RecipeRecommender(BaseEstimator):
    """
    A unified recommendation estimator that supports two representations:
      - "tfidf": Uses a TF‑IDF vectorizer (with optional SVD)
      - "embedding": Uses a pre‑trained SentenceTransformer model.
      
    It supports exact NN search (using scikit‑learn’s NearestNeighbors) or
    approximate search (via Annoy).
    """
    def __init__(self,
                 representation="tfidf",
                 vectorizer_params=None,    # For TF-IDF
                 use_svd=False,             # For TF-IDF
                 svd_params=None,           # For SVD (if use_svd is True)
                 embedding_model_name="all-MiniLM-L6-v2",  # For embeddings
                 use_approx_nn=False,
                 nn_params=None):           # For NN search
        self.representation = representation
        self.vectorizer_params = vectorizer_params if vectorizer_params is not None else {}
        self.use_svd = use_svd
        self.svd_params = svd_params if svd_params is not None else {}
        self.embedding_model_name = embedding_model_name
        self.use_approx_nn = use_approx_nn
        self.nn_params = nn_params if nn_params is not None else {}

        # Delay creation of heavy components until fit()
        self.vectorizer = None
        self.svd = None
        self.embedding_model = None
        self.nn_model = None
        self.annoy_index_ = None
        self.features_ = None
        self.dimension_ = None

    def fit(self, X, y=None):
        if self.representation == "tfidf":
            self.vectorizer = TfidfVectorizer(**self.vectorizer_params)
            X_vec = self.vectorizer.fit_transform(X)
            if self.use_svd:
                if not self.svd_params:
                    self.svd_params = {"n_components": 100}
                self.svd = TruncatedSVD(**self.svd_params)
                X_vec = self.svd.fit_transform(X_vec)
            else:
                # For exact NN search, if not using approximate NN, keep data sparse.
                if self.use_approx_nn:
                    X_vec = X_vec.toarray()
            self.features_ = X_vec
        elif self.representation == "embedding":
            if SentenceTransformer is None:
                raise ImportError("Please install sentence-transformers for embedding representation.")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            X_vec = self.embedding_model.encode(X, show_progress_bar=True)
            X_vec = np.array(X_vec)
            self.features_ = X_vec
        else:
            raise ValueError("Unknown representation type.")

        # Build the nearest neighbor model.
        if self.use_approx_nn:
            if AnnoyIndex is None:
                raise ImportError("Please install annoy for approximate nearest neighbor search.")
            self.dimension_ = self.features_.shape[1]
            self.annoy_index_ = AnnoyIndex(self.dimension_, metric="angular")
            for i, vec in enumerate(self.features_):
                self.annoy_index_.add_item(i, vec)
            n_trees = self.nn_params.get("n_trees", 10)
            self.annoy_index_.build(n_trees)
        else:
            if issparse(self.features_):
                self.nn_params.setdefault("algorithm", "brute")
            self.nn_model = NearestNeighbors(**self.nn_params)
            self.nn_model.fit(self.features_)
        return self

    def kneighbors(self, X, n_neighbors=5):
        if self.representation == "tfidf":
            X_vec = self.vectorizer.transform(X)
            if self.use_svd:
                X_vec = self.svd.transform(X_vec)
            else:
                if self.use_approx_nn:
                    X_vec = X_vec.toarray()
        elif self.representation == "embedding":
            X_vec = self.embedding_model.encode(X, show_progress_bar=False)
            X_vec = np.array(X_vec)
        else:
            raise ValueError("Unknown representation type.")

        if self.use_approx_nn:
            distances = []
            indices = []
            for vec in X_vec:
                idxs, dists = self.annoy_index_.get_nns_by_vector(vec, n_neighbors, include_distances=True)
                indices.append(idxs)
                distances.append(dists)
            return np.array(distances), np.array(indices)
        else:
            return self.nn_model.kneighbors(X_vec, n_neighbors=n_neighbors)

    def score(self, X, y=None):
        distances, _ = self.kneighbors(X, n_neighbors=2)
        # Use the second neighbor's distance (first is the sample itself).
        avg_similarity = (1 - distances[:, 1]).mean()
        return avg_similarity
