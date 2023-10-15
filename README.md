# Parkinson-s-Disease-Detection

## Prerequisites
Before using the Parkinson's disease detection model, you need to have the following dependencies installed:
- Python (>=3.6)
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy

## Code Overview
- Import the necessary modules and libraries.
- Load the Parkinson's disease dataset from a CSV file (parkinsons.csv).
- Display basic information about the dataset, such as its shape and statistical summary.
- Check for missing values in the dataset.
- Visualize the distribution of the 'status' column using a catplot.
- Split the data into input features (X) and labels (Y), and apply standard scaling to the input features.
- Split the data into training and testing sets.
- Create an SVM (Support Vector Machine) model and train it using the training data.
- Make predictions on both the training and testing data.
- Calculate and print the accuracy of the model on both sets.

## Accuracy
The model achieves an accuracy of 88% on testing data, making it a reliable tool for detecting Parkinson's disease based on the provided features.

## Usage
- Make sure you have installed the necessary dependencies mentioned in the prerequisites section.
- Clone this repository or download the script to your local environment.
- Run the script.
- Use the predictive system by providing input data as a tuple.

## Contributions 
Contributions to this project are welcome. If you have ideas to improve the model's accuracy, efficiency, or expand its capabilities, feel free to make a pull request or open an issue. 
Your contributions can help in the early detection of Parkinson's disease, which is essential for timely treatment and support for affected individuals.

