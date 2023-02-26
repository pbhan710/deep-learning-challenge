# deep-learning-challenge
Repository of Python code in Jupyter notebook that preprocesses data; compile, train, and evaluate a neural network (i.e., deep learning model); and optimizes the model that predicts if an Alphabet Soup-funded organization will be successful based on features in the dataset.

## Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

- **EIN** and **NAME**—Identification columns
- **APPLICATION_TYPE**—Alphabet Soup application type
- **AFFILIATION**—Affiliated sector of industry
- **CLASSIFICATION**—Government organization classification
- **USE_CASE**—Use case for funding
- **ORGANIZATION**—Organization type
- **STATUS**—Active status
- **INCOME_AMT**—Income classification
- **SPECIAL_CONSIDERATIONS**—Special considerations for application
- **ASK_AMT**—Funding amount requested
- **IS_SUCCESSFUL**—Was the money used effectively

## Instructions
### Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s `StandardScaler()`, you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

1. Read in `charity_data.csv` to a Pandas DataFrame, and be sure to identify the following in your dataset:
- What variable(s) are the target(s) for your model?
- What variable(s) are the feature(s) for your model?

2. Drop the `EIN` and `NAME` columns.

3. Determine the number of unique values for each column.

4. For columns that have more than 10 unique values, determine the number of data points for each unique value.

5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.

6. Use `pd.get_dummies()` to encode categorical variables.

7. Split the preprocessed data into a features array, `X`, and a target array, `y`. Use these arrays and the `train_test_split` function to split the data into training and testing datasets.

8. Scale the training and testing features datasets by creating a `StandardScaler` instance, fitting it to the training data, then using the `transform` function.