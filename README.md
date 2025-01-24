# Boston Housing Prices Prediction

![Boston Housing Prices Prediction](https://github.com/vincenzomaltese/Boston-House-Prices/blob/main/Images/Boston-Housing.jpg)

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation](#running-the-project)
- [Dataset Description](#dataset-description)
- [Key Steps](#key-steps)
- [Results](#results)
- [Usage](#usage)

---

## Project Overview

The goal of this project is to analyze and predict housing prices in Boston based on several factors such as the number of rooms, location, and socio-economic conditions. By utilizing the dataset's features, a Linear Regression model is developed and evaluated to provide accurate price predictions.

## Technologies Used

- **Python**: Programming language used for data analysis and modeling.
- **Libraries**:
  - `pandas`: Data manipulation and analysis.
  - `numpy`: Numerical computations.
  - `matplotlib` and `seaborn`: Data visualization.
  - `scikit-learn`: Machine learning algorithms and preprocessing tools.

## Running the Project
1. Clone this repository.
2. Open `Project-Case-Boston-House.ipynb` in Jupyter Notebook or JupyterLab.
3. Ensure the following libraries are installed: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.


## Dataset Description

The Boston Housing dataset contains 506 samples and 14 features, including:

- **CRIM**: Per capita crime rate by town.
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq. ft.
- **INDUS**: Proportion of non-retail business acres per town.
- **RM**: Average number of rooms per dwelling.
- **LSTAT**: Percentage of lower-status population.
- **MEDV**: Median value of owner-occupied homes (target variable).

> **Note**: The dataset is included in the `sklearn.datasets` module.

## Key Steps

1. **Exploratory Data Analysis (EDA):**
   - Examined data distribution and outliers.
   - Visualized correlations between features using heatmaps and scatter plots.

2. **Data Preprocessing:**
   - Normalized features using StandardScaler.
   - Split data into training and testing sets (80/20 split).

3. **Model Building:**
   - Developed a Linear Regression model using `scikit-learn`.
   - Trained the model on the training set and evaluated its performance on the test set.

4. **Evaluation Metrics:**
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - R-squared (R²)

## Results

- The model achieved a strong correlation between predicted and actual prices.
- **Evaluation Metrics:**
  - MAE: 3.42
  - MSE: 24.56
  - R²: 0.72

Visualizations include:
- Heatmaps of feature correlations.
- Comparison of predicted vs. actual prices.

## Usage

1. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Project-Case-Boston-House.ipynb
   ```

2. Follow the steps outlined in the notebook to explore the dataset, preprocess the data, and train the model.

3. Modify or extend the notebook to experiment with different algorithms or hyperparameters.

---

Feel free to contribute to this project or suggest improvements by creating a pull request or submitting an issue!
