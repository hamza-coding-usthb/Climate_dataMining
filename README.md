# Climate and Soil Data Mining Tool

This project is a comprehensive desktop application developed in Python for data mining and analysis of climate and soil datasets. It provides a user-friendly graphical interface built with PyQt5 that integrates various data preprocessing, cleaning, visualization, and machine learning functionalities.

A key feature of this project is that the core machine learning algorithms (Decision Tree, Random Forest, CLARANS, DBSCAN) are implemented from scratch, offering a deep dive into their mechanics.

## Features

The application provides a rich set of features to handle data from import to analysis:

### 1. Data Management & Exploration
- **Import/Export**: Load datasets from CSV files and save the processed data back to a CSV file.
- **Paginated View**: View large datasets efficiently through a paginated table.
- **Data Description**: Get a detailed statistical summary of the dataset, including mean, median, standard deviation, quartiles, and outlier bounds.
- **Instance Manipulation**: Directly update or delete individual rows in the dataset.
- **Undo**: Revert the last data modification.

### 2. Data Preprocessing & Cleaning
- **Handling Missing Values**: Remove rows with NaN values.
- **Outlier Detection & Treatment**:
  - Remove outliers based on the Interquartile Range (IQR).
  - Replace outliers with the column's mean, median, or a contextual mean (based on season and location).
  - Cap outliers at the 5th/95th percentiles or IQR limits.
- **Redundancy Elimination**:
  - **Vertical**: Remove duplicate rows.
  - **Horizontal**: Remove highly correlated features to reduce multicollinearity.
- **Data Transformation**:
  - **Normalization**: Apply Min-Max or Z-Score normalization.
  - **Aggregation**: Aggregate data on a monthly or seasonal basis.
  - **Discretization**: Convert continuous numerical data into categorical bins using Equal Width or Equal Frequency methods.
- **Data Merging**:
  - Merge two datasets based on common columns (e.g., latitude and longitude).
  - Perform spatial merges to aggregate climate data within soil polygons.

### 3. Data Visualization
- **Basic Plots**:
  - **Boxplot**: To visualize data distribution and identify outliers.
  - **Scatter Plot**: To explore relationships between two variables, annotated with the Pearson correlation coefficient.
  - **Histogram**: To view the frequency distribution of a variable.
  - **Bar Plot**: For visualizing categorical or interval data frequencies.
- **Advanced Plots**:
  - **Correlation Heatmap**: To visualize the correlation matrix of all numerical features.
  - **Parallel Coordinates**: For visualizing high-dimensional data.
  - **Pairwise Scatterplot Matrix**: To see relationships across multiple variable pairs at once.
  - **Bubble Chart**: To visualize the relationship between three numerical variables.

### 4. Machine Learning Algorithms (From Scratch)
The application includes custom implementations of several key data mining algorithms.

- **Clustering**:
  - **CLARANS**: A medoid-based clustering algorithm efficient for large datasets.
  - **DBSCAN**: A density-based clustering algorithm to find arbitrarily shaped clusters and identify noise.
  - Both clustering algorithms can be run with or without Principal Component Analysis (PCA) for dimensionality reduction.

- **Regression & Prediction**:
  - **Decision Tree Regressor**: A tree-based model for prediction tasks. The resulting tree can be visualized.
  - **Random Forest Regressor**: An ensemble model built from multiple decision trees for improved accuracy and robustness.

## Project Structure

The project is organized into several Python files:

```
part2_final2/
├── main.py                     # Main entry point to launch the application
├── csv_viewer.py               # Core GUI class, manages the main window and integrates all modules
├── data_manipulation_dialog.py # Handles data import, export, and description
├── data_cleaning_dialog.py     # Implements all data cleaning functionalities
├── data_normalization_dialog.py# Implements normalization techniques
├── data_aggregation_dialog.py  # Implements data aggregation
├── data_discretization_dialog.py# Implements data discretization
├── data_merger_dialog.py       # Implements data merging functionalities
├── data_visualization_dialog.py# Implements all data visualization plots
├── algorithms_clarans.py       # Custom implementation of the CLARANS algorithm
├── algorithms_dbscan.py        # Custom implementation of the DBSCAN algorithm
├── algorithms_decision_tree.py # Custom implementation of the Decision Tree Regressor
└── algorithms_random_forest.py # Custom implementation of the Random Forest Regressor
```

## Prerequisites

Before running the application, ensure you have Python and the following libraries installed:

- PyQt5
- pandas
- numpy
- scikit-learn (for metrics, PCA, and data splitting)
- matplotlib
- seaborn
- geopandas (for spatial merging)
- graphviz (to visualize decision trees)

You can install them using pip:
```bash
pip install PyQt5 pandas numpy scikit-learn matplotlib seaborn geopandas graphviz
```

## How to Run

1.  Navigate to the project directory:
    ```bash
    cd path/to/part2_final2
    ```
2.  Run the `main.py` script:
    ```bash
    python main.py
    ```
3.  The main application window will open. Start by importing a CSV dataset using the **Data Manipulation** -> **Import Dataset** button.

## How to Use

1.  **Load Data**: Use the "Data Manipulation" panel to import your CSV file.
2.  **Explore**: Use "Dataset Description" to understand your data's statistics.
3.  **Clean & Preprocess**: Use the various dialogs (Data Cleaning, Normalization, etc.) to prepare your data for analysis. Each operation will update the data displayed in the main table.
4.  **Visualize**: Use the "Data Visualization" panel to create plots and gain insights.
5.  **Apply Algorithms**:
    - Use the right-hand sidebar to run the machine learning algorithms.
    - A dialog will pop up asking for the required parameters (e.g., number of clusters for CLARANS, target column for Decision Tree).
    - The results, including performance metrics like training time and MSE, will be displayed. For clustering, the clusters will be plotted. For decision trees, the tree structure can be visualized.
6.  **Save Data**: Once you are done, you can save the cleaned and transformed dataset to a new CSV file.

---

This tool is designed to be an educational and practical asset for anyone interested in the fundamentals of data mining and machine learning, providing both a hands-on application and clear, from-scratch algorithm implementations.