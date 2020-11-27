# Pandas is used for data manipulation
import pandas as pd

import pickle

# Read in data 
features = pd.read_csv('ourDataset.csv')

#############################################################

# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels = np.array(features[['Trg','Density_composition_average','IsBoron_composition_average','IsDBlock_composition_average','IsTransitionMetal_composition_average','NdValence_composition_average','NValance_composition_average','HeatVaporization_max_value']])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop(['Trg','Density_composition_average','IsBoron_composition_average','IsDBlock_composition_average','IsTransitionMetal_composition_average','NdValence_composition_average','NValance_composition_average','HeatVaporization_max_value'], axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

#############################################################
# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels,test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 42 )
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
 
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 700 decision trees
rf = RandomForestRegressor(n_estimators = 700, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels);

pickle.dump(rf, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
print( predictions )
#Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

########################################################################

#Import tools needed for visualization
#from sklearn.tree import export_graphviz
#import pydot
#import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# Pull out one tree from the forest
#tree = rf.estimators_[5]
# Export the image to a dot file
#export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
#(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
#graph.write_png('tree.png')

