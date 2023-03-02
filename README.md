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
    - What variable(s) are the target(s) for the model?
    - What variable(s) are the feature(s) for the model?

2. Drop the `EIN` and `NAME` columns.

3. Determine the number of unique values for each column.

4. For columns that have more than 10 unique values, determine the number of data points for each unique value.

5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.

6. Use `pd.get_dummies()` to encode categorical variables.

7. Split the preprocessed data into a features array, `X`, and a target array, `y`. Use these arrays and the `train_test_split` function to split the data into training and testing datasets.

8. Scale the training and testing features datasets by creating a `StandardScaler` instance, fitting it to the training data, then using the `transform` function.

### Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

2. Create the first hidden layer and choose an appropriate activation function.

3. If necessary, add a second hidden layer with an appropriate activation function.

4. Create an output layer with an appropriate activation function.

5. Check the structure of the model.

6. Compile and train the model.

7. Create a callback that saves the model's weights every five epochs.

8. Evaluate the model using the test data to determine the loss and accuracy.

9. Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity.h5`.

### Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

- Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
    - Dropping more or fewer columns.
    - Creating more bins for rare occurrences in columns.
    - Increasing or decreasing the number of values for each bin.

- Add more neurons to a hidden layer.

- Add more hidden layers.

- Use different activation functions for the hidden layers.

- Add or reduce the number of epochs to the training regimen.

**Note**: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

1. Create a new Jupyter Notebook file and name it `AlphabetSoupCharity_Optimization.ipynb`.

2. Import your dependencies and read in the `charity_data.csv` to a Pandas DataFrame.

3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

5. Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity_Optimization.h5`.

### Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

1. **Overview**: Explain the purpose of this analysis.

2. **Results**: Using bulleted lists and images to support your answers, address the following questions:

    - Data Preprocessing
        - What variable(s) are the target(s) for your model?
        - What variable(s) are the features for your model?
        - What variable(s) should be removed from the input data because they are neither targets nor features?

    - Compiling, Training, and Evaluating the Model
        - How many neurons, layers, and activation functions did you select for your neural network model, and why?
        - Were you able to achieve the target model performance?
        - What steps did you take in your attempts to increase model performance?

3. **Summary**: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

## Report
### Overview
The purpose of this analysis was to create a binary classifier that can help Alphabet Soup predict which applicants will be successful in their ventures if funded by Alphabet Soup. This analysis utilized past organizations that have received funding from Alphabet Soup over the years, with a number of features and determination of whether the funding was used effectively.

### Results
#### Data Preprocessing
1. *What variable(s) are the target(s) for your model?* 'IS_SUCCESSFUL'

2. *What variable(s) are the features for your model?* 'APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION', 'STATUS', 'INCOME_AMT', 'SPECIAL_CONSIDERATIONS', 'ASK_AMT'

3. *What variable(s) should be removed from the input data because they are neither targets nor features?* 'EIN', 'NAME'

#### Compiling, Training, and Evaluating the Model
1. *How many neurons, layers, and activation functions did you select for your neural network model, and why?* 

2. *Were you able to achieve the target model performance?*

3. *What steps did you take in your attempts to increase model performance?* 
    - Re-binned "Other" categories for 'APPLICATION_TYPE' and 'CLASSIFICATION' features.
    - Added additional hidden layer.
    - Increased number of neurons in both hidden layers.

### Summary
Given that this is ultimately a classification problem, perhaps a different logistical model could have performed better than this deep learning model.