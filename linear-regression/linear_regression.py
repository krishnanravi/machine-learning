
# coding: utf-8

# In[26]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[20]:


# Step 1: Prepare data
data_df = pd.read_csv("./data/ex1data1.txt")
data_arry = data_df.values
training_set_size = data_arry.size #size of the training set

features_n_output = np.split(data_arry, 2, 1)
raw_features = features_n_output[0] #raw features
ones_arry = np.ones((np.size(raw_features, 0), 1)) #to account for theta0
features = np.concatenate((ones_arry, raw_features), axis=1)
expected_op = features_n_output[1] #expected output


# In[21]:


# Step 2: Visualize data
plt.scatter(raw_features, expected_op)


# In[47]:


# Step 3: Define hypothesis and cost function

#initialize model params to all zeros
mdl_params = np.zeros((np.size(features,0),1))

#predict the output using the params without any training
hypothesis = features * mdl_params 

#visualize the training data and model
plt.scatter(raw_features, expected_op)
plt.scatter(features, hypothesis)


# In[74]:


# Step 4: Train model
def compute_cost(mdl_params, features, expected_op): 
    size = np.size(features, 1)
    return (1/size) * np.square(mdl_params * features - expected_op).sum()

print(compute_cost(mdl_params, features, expected_op))

def train_gradient_descent(mdl_params, features, expected_op, learning_rate, max_iterations): 
    prev_cost = -1
    for count in range(1, max_iterations) :
        cost = compute_cost(mdl_params, features, expected_op)
        if cost > prev_cost:
            print("iteration #{}. Cost is {}".format(count, cost))
        else:
            break       
    return mdl_params


#Train model using gradient descent
max_iterations=2000 #define max iterations
learning_rate = 0.1 #define learning rate
mdl_params = train_gradient_descent(mdl_params, features, expected_op, learning_rate, max_iterations)

#visualize the training data and model
plt.scatter(raw_features, expected_op)
plt.scatter(features, hypothesis)

