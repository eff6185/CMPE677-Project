# Import necessary libraries
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations and random generation
import matplotlib.pyplot as plt  # For plotting, though not used in this code
from sklearn.svm import OneClassSVM  # One-Class Support Vector Machine (SVM) for anomaly detection
from tqdm.auto import tqdm  # Progress bar for loops
import time

# Random seed for reproducibility
np.random.seed(0)

# Parameters
# n_training_samples = 16  # Number of training samples (unused in this code)
# nums_features = [1, 4, 8, 16]  # List of feature counts to iterate over
# nums_samples = [1, 2, 3, 4, 5]  # List of sample sizes to iterate over
percentages = [50] #percentages = [10, 20, 30, 40, 50]  Different percentages of Gaussian noise
ratios = [0.9] #ratios = [0.8, 0.9]  Data split ratios for train/test (80-20% or 90-10%)
ratios_text = ['9-1'] #ratios_text = ['8-2', '9-1'] Text labels for the split ratios
# List of different `nu` (a hyperparameter of OneClassSVM, controlling support vectors)
NUs = [0.05, 1./16., 0.1, 0.2, 0.5, 0.7, 0.9, 0.99]
# List of different `gamma` values (another hyperparameter controlling kernel width in SVM)
GAMMAs = [0.05, 1./16., 0.1, 0.2, 0.5, 0.7, 0.9, 0.99]

OptimizedNU = [0.99, 0.99, 0.99, 0.05, 0.99, 0.99, 0.05, 0.99, 0.99, 0.05, 0.99, 0.99, 0.99]
OptimizedGamma = [0.99, 0.99, 0.99, 0.05, 0.99, 0.99, 0.05, 0.99, 0.99, 0.05, 0.99, 0.99, 0.99]
OptimizedModels = []
MultiTrainData = []
SelfClassificaitons = []
SharedIndices = {}





# Iterate through different configurations
for percentage in tqdm(percentages, desc="Processing percentages"):
    for ratio, ratio_text in tqdm(zip(ratios, ratios_text), desc="Processing ratios", leave=True):  # Iterate over train-test split ratios
        output = []  # Placeholder for storing results before writing to CSV
        # print(f'No. of features: {n_features}, No. of generated samples: {n_samples}, Split ratio: {ratio_text}, Gaussian percentage: {percentage}')
        
        # Read data from CSV based on current configuration (generated beforehand)
        df = pd.read_csv("New/Data-traffic-distribution-giga.csv")
        
        # Column indicating the class labels
        class_column = 'Applications (Label Classes)'
        
        # Get unique class labels in the dataset
        labels = np.unique(df[class_column].to_numpy())
        
        # Convert class labels into numeric values (e.g., categorical to integer)
        df[class_column] = pd.factorize(df[class_column])[0]
        label_indices = np.unique(df[class_column].to_numpy())  # Get unique numeric labels
        
        tmp_out = []  # Temporary output for current configuration
        tmp_out.append(['Features: 16', '', '', '', '', '', ''])  # Formatting for CSV output

        # Iterate through each class (target class)
        i = 0
        for index in tqdm(label_indices, desc="Processing classes", leave=True):
            tqdm.write(f'Target class: {labels[index]}')

            #Grab Optimized Parameters
            nu = OptimizedNU[i]
            gamma = OptimizedGamma[i]
            i += 1

            # Select data points for the current class (target class) excluding the class label column
            X_train = df[df[class_column] == index].to_numpy()[:, :16]
            np.random.shuffle(X_train)  # Shuffle the rows to randomize

            # Split the data into training and testing based on the given ratio (80-20% or 90-10%)
            split_index = int(X_train.shape[0] * ratio)
            
            # Training data (target class)
            X_train_target = X_train[:split_index, :]
            
            # Test data from the target class (benign cases)
            X_test_target = X_train[split_index:, :]

            #Add Test Data to list of data to be used for multi

            MultiTrainData.extend(X_test_target)
            
            # Test data from other classes (anomalous cases)
            X_test_others = df[df[class_column] != index].to_numpy()[:, :16]

            # Add target class header to the temporary output
            tmp_out.append([f'Target Class: {labels[index]}', 'TP', 'FP', 'FN', 'TN', 'ACC (%)', ''])  # TP, FP, etc. are placeholders for output
 
            # Initialize OneClassSVM with current parameters
            clf = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
            clf.fit(X_train_target)  # Train the model on the target class
            
            #Append the trained model to the list of optimized models
            OptimizedModels.append(clf)


#MultiClassificaiton attempt
print("Testing MultiClass identification now")
classindex = 0 #For tracking which model to currently predict on
for index in tqdm(label_indices, desc="Processing classes", leave=True):
    tqdm.write(f'Target class: {labels[index]}')
    currentmodel = OptimizedModels[classindex]

    classindex = classindex + 1
    
    #Makes the prediciton on the model
    predictions = currentmodel.predict(MultiTrainData)

    #Tracked the indices that were selected as apart of the model
    inlier_indices = [i for i, pred in enumerate(predictions) if pred == 1]

    SelfClassificaitons.append(inlier_indices)

for index, sublist in enumerate(SelfClassificaitons):
    for number in sublist:
        # Add or update the number in the dictionary
        if number not in SharedIndices:
            SharedIndices[number] = []
        SharedIndices[number].append(index)

SharedIndices = {num: indices for num, indices in SharedIndices.items() if len(indices) > 1}

print(SharedIndices)





        

