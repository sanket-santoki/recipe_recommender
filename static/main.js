document.addEventListener('DOMContentLoaded', function() {
  // Fetch dynamic ingredients list from the backend.
  fetch('/ingredients')
    .then(response => response.json())
    .then(data => {
      initializeTagify(data);
    })
    .catch(error => {
      console.error("Error fetching ingredients:", error);
      // If fetching fails, initialize Tagify with an empty whitelist.
      initializeTagify([]);
    });

  function initializeTagify(whitelist) {
    var input = document.getElementById('ingredients');
    var tagify = new Tagify(input, {
      whitelist: whitelist,
      dropdown: {
        enabled: 1,      // Show suggestions when the input is focused.
        closeOnSelect: false
      }
    });

    // Set up the "Get Recommendations" button.
    const getRecBtn = document.getElementById('get-rec-btn');
    getRecBtn.addEventListener('click', function() {
      const ingredientsArr = tagify.value.map(item => item.value);
      if (ingredientsArr.length === 0) {
        alert("Please enter at least one ingredient.");
        return;
      }
      const ingredients = ingredientsArr.join(", ");
      fetchRecommendations(ingredients);
    });
  }

  // Refresh the Performance Metrics iframe when its tab is clicked.
  const resultsTabLink = document.querySelector('.tab-link[data-tab="results-tab"]');
  if (resultsTabLink) {
    resultsTabLink.addEventListener('click', function() {
      const iframe = document.getElementById('metrics-frame');
      if (iframe) {
        iframe.contentWindow.location.reload();
      }
    });
  }
});

function fetchRecommendations(ingredients) {
  const recipesContainer = document.getElementById('recipes-container');
  recipesContainer.innerHTML = "<p>Loading recommendations...</p>";

  fetch('/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ingredients: ingredients })
  })
  .then(response => {
    if (!response.ok) {
      throw new Error("Network error: " + response.statusText);
    }
    return response.json();
  })
  .then(data => {
    displayRecipes(data);
  })
  .catch(error => {
    recipesContainer.innerHTML = `<p>Error: ${error.message}</p>`;
    console.error("Error fetching recommendations:", error);
  });
}

function displayRecipes(recipes) {
  const recipesContainer = document.getElementById('recipes-container');
  recipesContainer.innerHTML = "";

  if (!recipes || recipes.length === 0) {
    recipesContainer.innerHTML = "<p>No recommendations found.</p>";
    return;
  }

  recipes.forEach(recipe => {
    // Create the recipe card.
    const card = document.createElement('div');
    card.className = 'recipe-card';

    // Recipe image.
    const img = document.createElement('img');
    img.src = recipe.image_url || "";
    img.alt = recipe.recipe_name || "Recipe Image";
    card.appendChild(img);

    // Recipe info container.
    const info = document.createElement('div');
    info.className = 'recipe-info';

    // Recipe header: name, rating, and review count.
    const headerDiv = document.createElement('div');
    headerDiv.className = 'recipe-header';

    const title = document.createElement('h2');
    title.textContent = recipe.recipe_name || "Unknown Recipe";
    headerDiv.appendChild(title);

    const ratingDiv = document.createElement('div');
    ratingDiv.className = 'rating';
    const rate = (typeof recipe.aver_rate === 'number') ? recipe.aver_rate : 0;
    ratingDiv.innerHTML = `<span class="rating-value">${rate.toFixed(1)}</span>
                           <span class="review-count">(${recipe.review_nums || 0} reviews)</span>`;
    headerDiv.appendChild(ratingDiv);
    info.appendChild(headerDiv);

    // Recipe details: calories and macros.
    const details = document.createElement('p');
    details.innerHTML = `<strong>Calories:</strong> ${recipe.calories || 0}<br>
                         <strong>Macros:</strong> Fat: ${recipe.fat || 0}g, Carbs: ${recipe.carbohydrates || 0}g, Protein: ${recipe.protein || 0}g`;
    info.appendChild(details);
    card.appendChild(info);

    // Available ingredients chips.
    const availContainer = document.createElement('div');
    availContainer.className = 'chips-container';
    const availLabel = document.createElement('p');
    availLabel.textContent = "Available Ingredients:";
    availContainer.appendChild(availLabel);
    if (Array.isArray(recipe.available_ingredients) && recipe.available_ingredients.length > 0) {
      recipe.available_ingredients.forEach(ing => {
        const chip = document.createElement('span');
        chip.className = 'chip available';
        chip.textContent = ing;
        availContainer.appendChild(chip);
      });
    } else {
      const chip = document.createElement('span');
      chip.className = 'chip available';
      chip.textContent = "None";
      availContainer.appendChild(chip);
    }
    card.appendChild(availContainer);

    // Missing ingredients chips.
    const missingContainer = document.createElement('div');
    missingContainer.className = 'chips-container';
    const missingLabel = document.createElement('p');
    missingLabel.textContent = "Missing Ingredients:";
    missingContainer.appendChild(missingLabel);
    if (Array.isArray(recipe.missing_ingredients) && recipe.missing_ingredients.length > 0) {
      recipe.missing_ingredients.forEach(ing => {
        const chip = document.createElement('span');
        chip.className = 'chip missing';
        chip.textContent = ing;
        missingContainer.appendChild(chip);
      });
    } else {
      const chip = document.createElement('span');
      chip.className = 'chip missing';
      chip.textContent = "None";
      missingContainer.appendChild(chip);
    }
    card.appendChild(missingContainer);

    // Similarity score.
    const similarityEl = document.createElement('p');
    similarityEl.className = 'similarity';
    let simValue = (typeof recipe.similarity === 'number') ? recipe.similarity : 0;
    similarityEl.innerHTML = `<strong>Similarity Score:</strong> ${simValue.toFixed(2)}`;
    card.appendChild(similarityEl);

    recipesContainer.appendChild(card);
  });
}
