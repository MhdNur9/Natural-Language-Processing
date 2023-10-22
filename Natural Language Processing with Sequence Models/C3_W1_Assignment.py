#!/usr/bin/env python
# coding: utf-8

# # Assignment 1:  Sentiment with Deep Neural Networks
# 
# Welcome to the first assignment of course 3. In this assignment, you will explore sentiment analysis using deep neural networks. 
# 
# ## Important Note on Submission to the AutoGrader
# 
# Before submitting your assignment to the AutoGrader, please make sure you are not doing the following:
# 
# 1. You have not added any _extra_ `print` statement(s) in the assignment.
# 2. You have not added any _extra_ code cell(s) in the assignment.
# 3. You have not changed any of the function parameters.
# 4. You are not using any global variables inside your graded exercises. Unless specifically instructed to do so, please refrain from it and use the local variables instead.
# 5. You are not changing the assignment code where it is not required, like creating _extra_ variables.
# 
# If you do any of the following, you will get something like, `Grader Error: Grader feedback not found` (or similarly unexpected) error upon submitting your assignment. Before asking for help/debugging the errors in your assignment, check for these first. If this is the case, and you don't remember the changes you have made, you can get a fresh copy of the assignment by following these [instructions](https://www.coursera.org/learn/sequence-models-in-nlp/supplement/6ThZO/how-to-refresh-your-workspace).

# ## Table of Contents
# - [1 - Import Libraries and try out Trax](#1)
# - [2 - Importing the Data](#2)
#     - [2.1 - Loading in the Data](#2-1)
#     - [2.2 - Building the Vocabulary](#2-2)
#     - [2.3 - Converting a Tweet to a Tensor](#2-3)
#         - [Exercise 1 - tweet_to_tensor (UNQ_C1)](#ex-1)
#     - [2.4 - Creating a Batch Generator](#2-4)
#         - [Exercise 2 - data_generator (UNQ_C2)](#ex-2)
# - [3 - Defining Classes](#3)
#     - [3.1 - ReLU Class](#3-1)
#         - [Exercise 3 - Relu (UNQ_C3)](#ex-3)
#     - [3.2 - Dense Class](#3.2)
#         - [Exercise 4 - Dense (UNQ_C4)](#ex-4)
#     - [3.3 - Model](#3-3)
#         - [Exercise 5 - classifier (UNQ_C5)](#ex-5)
# - [4 - Training](#4)
#     - [4.1 Training the Model](#4-1)
#         - [Exercise 6 - train_model (UNQ_C6)](#ex-6)
#     - [4.2 - Practice Making a Prediction](#4-2)
# - [5 - Evaluation](#5)
#     - [5.1 - Computing the Accuracy on a Batch](#5-1)
#         - [Exercise 7 - compute_accuracy (UNQ_C7)](#ex-7)
#     - [5.2 - Testing your Model on Validation Data](#5-2)
#         - [Exercise 8 - test_model (UNQ_C8)](#ex-8)
# - [6 - Testing with your Own Input](#6)
# - [7 - Word Embeddings](#7)

# In course 1, you implemented Logistic regression and Naive Bayes for sentiment analysis. However if you were to give your old models an example like:
# 
# <center> <span style='color:blue'> <b>This movie was almost good.</b> </span> </center>
# 
# Your model would have predicted a positive sentiment for that review. However, that sentence has a negative sentiment and indicates that the movie was not good. To solve those kinds of misclassifications, you will write a program that uses deep neural networks to identify sentiment in text. By completing this assignment, you will: 
# 
# - Understand how you can build/design a model using layers
# - Train a model using a training loop
# - Use a binary cross-entropy loss function
# - Compute the accuracy of your model
# - Predict using your own input
# 
# As you can tell, this model follows a similar structure to the one you previously implemented in the second course of this specialization. 
# - Indeed most of the deep nets you will be implementing will have a similar structure. The only thing that changes is the model architecture, the inputs, and the outputs. Before starting the assignment, we will introduce you to the Google library `trax` that we use for building and training models.
# 
# 
# Now we will show you how to compute the gradient of a certain function `f` by just using `  .grad(f)`. 
# 
# - Trax source code can be found on Github: [Trax](https://github.com/google/trax)
# - The Trax code also uses the JAX library: [JAX](https://jax.readthedocs.io/en/latest/index.html)

# <a name="1"></a>
# ## 1 - Import Libraries and try out Trax
# 
# - Let's import libraries and look at an example of using the Trax library.

# In[81]:


import os 
import shutil
import random as rnd

# import relevant libraries
import trax
import trax.fastmath.numpy as np
from trax import layers as tl
from trax import fastmath

# import Layer from the utils.py file
from utils import Layer, load_tweets, process_tweet
import w1_unittest


# In[82]:


# Create an array using trax.fastmath.numpy
a = np.array(5.0)

# View the returned array
display(a)

print(type(a))


# Notice that trax.fastmath.numpy returns a DeviceArray from the jax library.

# In[83]:


# Define a function that will use the trax.fastmath.numpy array
def f(x):
    
    # f = x^2
    return (x**2)


# In[84]:


# Call the function
print(f"f(a) for a={a} is {f(a)}")


# The gradient (derivative) of function `f` with respect to its input `x` is the derivative of $x^2$.
# - The derivative of $x^2$ is $2x$.  
# - When x is 5, then $2x=10$.
# 
# You can calculate the gradient of a function by using `trax.fastmath.grad(fun=)` and passing in the name of the function.
# - In this case the function you want to take the gradient of is `f`.
# - The object returned (saved in `grad_f` in this example) is a function that can calculate the gradient of f for a given trax.fastmath.numpy array.

# In[85]:


# Directly use trax.fastmath.grad to calculate the gradient (derivative) of the function
grad_f = trax.fastmath.grad(fun=f)  # df / dx - Gradient of function f(x) with respect to x

# View the type of the retuned object (it's a function)
type(grad_f)


# In[86]:


# Call the newly created function and pass in a value for x (the DeviceArray stored in 'a')
grad_calculation = grad_f(a)

# View the result of calling the grad_f function
display(grad_calculation)


# The function returned by trax.fastmath.grad takes in x=5 and calculates the gradient of f, which is 2*x, which is 10. The value is also stored as a DeviceArray from the jax library.

# <a name="2"></a>
# ## 2 - Importing the Data
# 
# <a name="2-1"></a>
# ### 2.1 - Loading in the Data
# 
# Import the data set.  
# - You may recognize this from earlier assignments in the specialization.
# - Details of process_tweet function are available in utils.py file

# In[87]:


## DO NOT EDIT THIS CELL

# Import functions from the utils.py file

def train_val_split():
    # Load positive and negative tweets
    all_positive_tweets, all_negative_tweets = load_tweets()

    # View the total number of positive and negative tweets.
    print(f"The number of positive tweets: {len(all_positive_tweets)}")
    print(f"The number of negative tweets: {len(all_negative_tweets)}")

    # Split positive set into validation and training
    val_pos   = all_positive_tweets[4000:] # generating validation set for positive tweets
    train_pos  = all_positive_tweets[:4000]# generating training set for positive tweets

    # Split negative set into validation and training
    val_neg   = all_negative_tweets[4000:] # generating validation set for negative tweets
    train_neg  = all_negative_tweets[:4000] # generating training set for nagative tweets
    
    # Combine training data into one set
    train_x = train_pos + train_neg 

    # Combine validation data into one set
    val_x  = val_pos + val_neg

    # Set the labels for the training set (1 for positive, 0 for negative)
    train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))

    # Set the labels for the validation set (1 for positive, 0 for negative)
    val_y  = np.append(np.ones(len(val_pos)), np.zeros(len(val_neg)))


    return train_pos, train_neg, train_x, train_y, val_pos, val_neg, val_x, val_y


# In[88]:


train_pos, train_neg, train_x, train_y, val_pos, val_neg, val_x, val_y = train_val_split()

print(f"length of train_x {len(train_x)}")
print(f"length of val_x {len(val_x)}")


# Now import a function that processes tweets (we've provided this in the utils.py file).
# - `process_tweets` removes unwanted characters e.g. hashtag, hyperlinks, stock tickers from a tweet.
# - It also returns a list of words (it tokenizes the original string).

# In[89]:


# Try out function that processes tweets
print("original tweet at training position 0")
print(train_pos[0])

print("Tweet at training position 0 after processing:")
process_tweet(train_pos[0])


# Notice that the function `process_tweet` keeps key words, removes the hash # symbol, and ignores usernames (words that begin with '@').  It also returns a list of the words.

# <a name="2-2"></a>
# ### 2.2 - Building the Vocabulary
# 
# Now build the vocabulary.
# - Map each word in each tweet to an integer (an "index"). 
# - The following code does this for you, but please read it and understand what it's doing.
# - Note that you will build the vocabulary based on the training data. 
# - To do so, you will assign an index to everyword by iterating over your training set.
# 
# The vocabulary will also include some special tokens
# - `__PAD__`: padding
# - `</e>`: end of line
# - `__UNK__`: a token representing any word that is not in the vocabulary.

# In[90]:


# Build the vocabulary
# Unit Test Note - There is no test set here only train/val
def get_vocab(train_x):

    # Include special tokens 
    # started with pad, end of line and unk tokens
    Vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2} 

    # Note that we build vocab using training data
    for tweet in train_x: 
        processed_tweet = process_tweet(tweet)
        for word in processed_tweet:
            if word not in Vocab: 
                Vocab[word] = len(Vocab)
    
    return Vocab

Vocab = get_vocab(train_x)

print("Total words in vocab are",len(Vocab))
display(Vocab)


# The dictionary `Vocab` will look like this:
# ```CPP
# {'__PAD__': 0,
#  '__</e>__': 1,
#  '__UNK__': 2,
#  'followfriday': 3,
#  'top': 4,
#  'engag': 5,
#  ...
# ```
# 
# - Each unique word has a unique integer associated with it.
# - The total number of words in Vocab: 9088

# <a name="2-3"></a>
# ## 2.3 - Converting a Tweet to a Tensor
# 
# Write a function that will convert each tweet to a tensor (a list of unique integer IDs representing the processed tweet).
# - Note, the returned data type will be a **regular Python `list()`**
#     - You won't use TensorFlow in this function
#     - You also won't use a numpy array
#     - You also won't use trax.fastmath.numpy array
# - For words in the tweet that are not in the vocabulary, set them to the unique ID for the token `__UNK__`.
# 
# ##### Example
# Input a tweet:
# ```CPP
# '@happypuppy, is Maria happy?'
# ```
# 
# The tweet_to_tensor will first conver the tweet into a list of tokens (including only relevant words)
# ```CPP
# ['maria', 'happi']
# ```
# 
# Then it will convert each word into its unique integer
# 
# ```CPP
# [2, 56]
# ```
# - Notice that the word "maria" is not in the vocabulary, so it is assigned the unique integer associated with the `__UNK__` token, because it is considered "unknown."
# 
# 

# <a name="ex-1"></a>
# ### Exercise 1 - tweet_to_tensor
# **Instructions:** Write a program `tweet_to_tensor` that takes in a tweet and converts it to an array of numbers. You can use the `Vocab` dictionary you just found to help create the tensor. 
# 
# - Use the vocab_dict parameter and not a global variable.
# - Do not hard code the integer value for the `__UNK__` token.

# <details>    
# <summary>
#     <font size="3" color="darkgreen"><b>Hints</b></font>
# </summary>
# <p>
# <ul>
#     <li>Map each word in tweet to corresponding token in 'Vocab'</li>
#     <li>Use Python's Dictionary.get(key,value) so that the function returns a default value if the key is not found in the dictionary.</li>
# </ul>
# </p>
# 

# In[91]:


# CANDIDATE FOR TABLE TEST - If a student forgets to check for unk, there might be errors or just wrong values in the list.
# We can add those errors to check in autograder through tabled test or here student facing user test.

# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT) 
# GRADED FUNCTION: tweet_to_tensor
def tweet_to_tensor(tweet, vocab_dict, unk_token='__UNK__', verbose=False):
    '''
    Input: 
        tweet - A string containing a tweet
        vocab_dict - The words dictionary
        unk_token - The special string for unknown tokens
        verbose - Print info durign runtime
    Output:
        tensor_l - A python list with
        
    '''     
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    # Process the tweet into a list of words
    # where only important words are kept (stop words removed)
    word_l = process_tweet(tweet)
    
    if verbose:
        print("List of words from the processed tweet:")
        print(word_l)
        
    # Initialize the list that will contain the unique integer IDs of each word
    tensor_l = [] 
    
    # Get the unique integer ID of the __UNK__ token
    unk_ID = vocab_dict[unk_token]
    
    if verbose:
        print(f"The unique integer ID for the unk_token is {unk_ID}")
        
    # for each word in the list:
    for word in word_l:
        
        # Get the unique integer ID.
        # If the word doesn't exist in the vocab dictionary,
        # use the unique ID for __UNK__ instead.        
        if word in vocab_dict:
            word_ID = vocab_dict[word]
        else:
            word_ID = unk_ID
            
        # Append the unique integer ID to the tensor list.
        tensor_l.append(word_ID)
    ### END CODE HERE ###
    
    return tensor_l


# In[92]:


print("Actual tweet is\n", val_pos[0])
print("\nTensor of tweet:\n", tweet_to_tensor(val_pos[0], vocab_dict=Vocab))


# ##### Expected output
# 
# ```CPP
# Actual tweet is
#  Bro:U wan cut hair anot,ur hair long Liao bo
# Me:since ord liao,take it easy lor treat as save $ leave it longer :)
# Bro:LOL Sibei xialan
# 
# Tensor of tweet:
#  [1065, 136, 479, 2351, 745, 8148, 1123, 745, 53, 2, 2672, 791, 2, 2, 349, 601, 2, 3489, 1017, 597, 4559, 9, 1065, 157, 2, 2]
# ```

# In[93]:


# Test your function
w1_unittest.test_tweet_to_tensor(tweet_to_tensor, Vocab)


# <a name="2-4"></a>
# ### 2.4 - Creating a Batch Generator
# 
# Most of the time in Natural Language Processing, and AI in general we use batches when training our data sets. 
# - If instead of training with batches of examples, you were to train a model with one example at a time, it would take a very long time to train the model. 
# - You will now build a data generator that takes in the positive/negative tweets and returns a batch of training examples. It returns the model inputs, the targets (positive or negative labels) and the weight for each target (ex: this allows us to can treat some examples as more important to get right than others, but commonly this will all be 1.0). 
# 
# Once you create the generator, you could include it in a for loop
# 
# ```CPP
# for batch_inputs, batch_targets, batch_example_weights in data_generator:
#     ...
# ```
# 
# You can also get a single batch like this:
# 
# ```CPP
# batch_inputs, batch_targets, batch_example_weights = next(data_generator)
# ```
# The generator returns the next batch each time it's called. 
# - This generator returns the data in a format (tensors) that you could directly use in your model.
# - It returns a triplet: the inputs, targets, and loss weights:
#     - Inputs is a tensor that contains the batch of tweets we put into the model.
#     - Targets is the corresponding batch of labels that we train to generate.
#     - Loss weights here are just 1s with same shape as targets. Next week, you will use it to mask input padding.

# <a name="ex-2"></a>
# ### Exercise 2 - data_generator
# Implement `data_generator`.

# In[94]:


# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED: Data generator
def data_generator(data_pos, data_neg, batch_size, loop, vocab_dict, shuffle=False):
    '''
    Input: 
        data_pos - Set of positive examples
        data_neg - Set of negative examples
        batch_size - number of samples per batch. Must be even
        loop - True or False
        vocab_dict - The words dictionary
        shuffle - Shuffle the data order
    Yield:
        inputs - Subset of positive and negative examples
        targets - The corresponding labels for the subset
        example_weights - A numpy array specifying the importance of each example
        
    '''     

    # make sure the batch size is an even number
    # to allow an equal number of positive and negative samples    
    assert batch_size % 2 == 0
    
    # Number of positive examples in each batch is half of the batch size
    # same with number of negative examples in each batch
    n_to_take = batch_size // 2
    
    # Use pos_index to walk through the data_pos array
    # same with neg_index and data_neg
    pos_index = 0
    neg_index = 0
    
    len_data_pos = len(data_pos)
    len_data_neg = len(data_neg)
    
    # Get and array with the data indexes
    pos_index_lines = list(range(len_data_pos))
    neg_index_lines = list(range(len_data_neg))
    
    # shuffle lines if shuffle is set to True
    if shuffle:
        rnd.shuffle(pos_index_lines)
        rnd.shuffle(neg_index_lines)
        
    stop = False
    
    # Loop indefinitely
    while not stop:  
        
        # create a batch with positive and negative examples
        batch = []
        
        # First part: Pack n_to_take positive examples
        
        # Start from 0 and increment i up to n_to_take
        for i in range(n_to_take):
                    
            # If the positive index goes past the positive dataset,
            if pos_index >= len_data_pos: 
                
                # If loop is set to False, break once we reach the end of the dataset
                if not loop:
                    stop = True;
                    break;
                # If user wants to keep re-using the data, reset the index
                pos_index = 0
                if shuffle:
                    # Shuffle the index of the positive sample
                    rnd.shuffle(pos_index_lines)
                    
            # get the tweet as pos_index
            tweet = data_pos[pos_index_lines[pos_index]]
            
            # convert the tweet into tensors of integers representing the processed words
            tensor = tweet_to_tensor(tweet, vocab_dict)
            
            # append the tensor to the batch list
            batch.append(tensor)
            
            # Increment pos_index by one
            pos_index = pos_index + 1


            
        ### START CODE HERE (Replace instances of 'None' with your code) ###

        # Second part: Pack n_to_take negative examples

        # Using the same batch list, start from 0 and increment i up to n_to_take
        for i in range(n_to_take):
            
            # If the negative index goes past the negative dataset,
            if not loop:
                
                # If loop is set to False, break once we reach the end of the dataset
                if not loop:
                    stop = True 
                    break 
                    
                # If user wants to keep re-using the data, reset the index
                neg_index = 0
                
                if shuffle:
                    # Shuffle the index of the negative sample
                    rnd.shuffle(neg_index_lines)
                    
            # get the tweet as neg_index
            tweet = data_neg[neg_index_lines[neg_index]]
            
            # convert the tweet into tensors of integers representing the processed words
            tensor = tweet_to_tensor(tweet, vocab_dict)
            
            # append the tensor to the batch list
            batch.append(tensor)
            
            # Increment neg_index by one
            neg_index += 1

        ### END CODE HERE ###        

        if stop:
            break;

        # Get the max tweet length (the length of the longest tweet) 
        # (you will pad all shorter tweets to have this length)
        max_len = max([len(t) for t in batch]) 
        
        
        # Initialize the input_l, which will 
        # store the padded versions of the tensors
        tensor_pad_l = []
        # Pad shorter tweets with zeros
        for tensor in batch:


        ### START CODE HERE (Replace instances of 'None' with your code) ###
            # Get the number of positions to pad for this tensor so that it will be max_len long
            n_pad = max_len - len(tensor)
            
            # Generate a list of zeros, with length n_pad
            pad_l = [0] * n_pad
            
            # concatenate the tensor and the list of padded zeros
            tensor_pad = tensor + pad_l
            
            # append the padded tensor to the list of padded tensors
            tensor_pad_l.append(tensor_pad)

        # convert the list of padded tensors to a numpy array
        # and store this as the model inputs
        inputs = np.array(tensor_pad_l)
  
        # Generate the list of targets for the positive examples (a list of ones)
        # The length is the number of positive examples in the batch
        target_pos = [1] * len(batch[:n_to_take])
        
        # Generate the list of targets for the negative examples (a list of zeros)
        # The length is the number of negative examples in the batch
        target_neg = [0] * len(batch[n_to_take:])
        
        # Concatenate the positve and negative targets
        target_l = target_pos + target_neg
        
        # Convert the target list into a numpy array
        targets = np.array(target_l)

        # Example weights: Treat all examples equally importantly.
        example_weights = np.ones_like(targets)
        

        ### END CODE HERE ###

        # note we use yield and not return
        yield inputs, targets, example_weights


# Now you can use your data generator to create a data generator for the training data, and another data generator for the validation data.
# 
# We will create a third data generator that does not loop, for testing the final accuracy of the model.

# In[95]:


# Set the random number generator for the shuffle procedure
rnd.seed(30) 

# Create the training data generator

def train_generator(batch_size, train_pos
                    , train_neg, vocab_dict, loop=True
                    , shuffle = False):
    return data_generator(train_pos, train_neg, batch_size, loop, vocab_dict, shuffle)

# Create the validation data generator
def val_generator(batch_size, val_pos
                    , val_neg, vocab_dict, loop=True
                    , shuffle = False):
    return data_generator(val_pos, val_neg, batch_size, loop, vocab_dict, shuffle)

# Create the validation data generator
def test_generator(batch_size, val_pos
                    , val_neg, vocab_dict, loop=False
                    , shuffle = False):
    return data_generator(val_pos, val_neg, batch_size, loop, vocab_dict, shuffle)

# Get a batch from the train_generator and inspect.
inputs, targets, example_weights = next(train_generator(4, train_pos, train_neg, Vocab, shuffle=True))

# this will print a list of 4 tensors padded with zeros
print(f'Inputs: {inputs}')
print(f'Targets: {targets}')
print(f'Example Weights: {example_weights}')


# In[96]:


# Test the train_generator

# Create a data generator for training data,
# which produces batches of size 4 (for tensors and their respective targets)
tmp_data_gen = train_generator(batch_size = 4, train_pos=train_pos, train_neg=train_neg, vocab_dict=Vocab)

# Call the data generator to get one batch and its targets
tmp_inputs, tmp_targets, tmp_example_weights = next(tmp_data_gen)

print(f"The inputs shape is {tmp_inputs.shape}")
for i,t in enumerate(tmp_inputs):
    print(f"input tensor: {t}; target {tmp_targets[i]}; example weights {tmp_example_weights[i]}")


# ##### Expected output
# 
# ```CPP
# The inputs shape is (4, 14)
# input tensor: [3 4 5 6 7 8 9 0 0 0 0 0 0 0]; target 1; example weights 1
# input tensor: [10 11 12 13 14 15 16 17 18 19 20  9 21 22]; target 1; example weights 1
# input tensor: [5738 2901 3761    0    0    0    0    0    0    0    0    0    0    0]; target 0; example weights 1
# input tensor: [ 858  256 3652 5739  307 4458  567 1230 2767  328 1202 3761    0    0]; target 0; example weights 1
# ```

# In[97]:


# Test your function
w1_unittest.test_data_generator(data_generator(data_pos=train_pos, data_neg=train_neg, batch_size=4, loop=True, vocab_dict=Vocab, shuffle = False))


# Now that you have your train/val generators, you can just call them and they will return tensors which correspond to your tweets in the first column and their corresponding labels in the second column. Now you can go ahead and start building your neural network. 

# <a name="3"></a>
# ## 3 - Defining Classes
# 
# In this part, you will write your own library of layers. It will be very similar
# to the one used in Trax and also in Keras and PyTorch. Writing your own small
# framework will help you understand how they all work and use them effectively
# in the future.
# 
# Your framework will be based on the following `Layer` class from utils.py.
# 
# ```CPP
# class Layer(object):
#     """ Base class for layers.
#     """
#       
#     # Constructor
#     def __init__(self):
#         # set weights to None
#         self.weights = None
# 
#     # The forward propagation should be implemented
#     # by subclasses of this Layer class
#     def forward(self, x):
#         raise NotImplementedError
# 
#     # This function initializes the weights
#     # based on the input signature and random key,
#     # should be implemented by subclasses of this Layer class
#     def init_weights_and_state(self, input_signature, random_key):
#         pass
# 
#     # This initializes and returns the weights, do not override.
#     def init(self, input_signature, random_key):
#         self.init_weights_and_state(input_signature, random_key)
#         return self.weights
#  
#     # __call__ allows an object of this class
#     # to be called like it's a function.
#     def __call__(self, x):
#         # When this layer object is called, 
#         # it calls its forward propagation function
#         return self.forward(x)
# ```

# <a name="3-1"></a>
# ### 3.1 - ReLU Class
# You will now implement the ReLU activation function in a class below. The ReLU function looks as follows: 
# <img src = "images/relu.jpg" style="width:300px;height:150px;"/>
# 
# $$ \mathrm{ReLU}(x) = \mathrm{max}(0,x) $$
# 

# <a name="ex-3"></a>
# ### Exercise 3 - Relu
# **Instructions:** Implement the ReLU activation function below. Your function should take in a matrix or vector and it should transform all the negative numbers into 0 while keeping all the positive numbers intact. 

# <details>    
# <summary>
#     <font size="3" color="darkgreen"><b>Hints</b></font>
# </summary>
# <p>
# <ul>
#     <li>Please use numpy.maximum(A,k) to find the maximum between each element in A and a scalar k</li>
# </ul>
# </p>
# 

# In[98]:


# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: Relu
class Relu(Layer):
    """Relu activation function implementation"""
    def forward(self, x):
        '''
        Input: 
            - x (a numpy array): the input
        Output:
            - activation (numpy array): all positive or 0 version of x
        '''
        ### START CODE HERE (Replace instances of 'None' with your code) ###
        
        activation = np.maximum(x, 0)

        ### END CODE HERE ###
        
        return activation


# In[99]:


# Test your relu function
x = np.array([[-2.0, -1.0, 0.0], [0.0, 1.0, 2.0]], dtype=float)
relu_layer = Relu()
print("Test data is:")
print(x)
print("Output of Relu is:")
print(relu_layer(x))


# ##### Expected Outout
# ```CPP
# Test data is:
# [[-2. -1.  0.]
#  [ 0.  1.  2.]]
# Output of Relu is:
# [[0. 0. 0.]
#  [0. 1. 2.]]
# ```

# In[100]:


w1_unittest.test_Relu(Relu)


# <a name="3.2"></a>
# ### 3.2 - Dense Class 
# 
# Implement the forward function of the Dense class. 
# - The forward function multiplies the input to the layer (`x`) by the weight matrix (`W`)
# 
# $$\mathrm{forward}(\mathbf{x},\mathbf{W}) = \mathbf{xW} $$
# 
# - You can use `numpy.dot` to perform the matrix multiplication.
# 
# Note that for more efficient code execution, you will use the trax version of `math`, which includes a trax version of `numpy` and also `random`.
# 
# Implement the weight initializer `new_weights` function
# - Weights are initialized with a random key.
# - The second parameter is a tuple for the desired shape of the weights (num_rows, num_cols)
# - The num of rows for weights should equal the number of columns in x, because for forward propagation, you will multiply x times weights.
# 
# Please use `trax.fastmath.random.normal(key, shape, dtype=tf.float32)` to generate random values for the weight matrix. The key difference between this function
# and the standard `numpy` randomness is the explicit use of random keys, which
# need to be passed. While it can look tedious at the first sight to pass the random key everywhere, you will learn in Course 4 why this is very helpful when
# implementing some advanced models.
# - `key` can be generated by calling `random.get_prng(seed=)` and passing in a number for the `seed`.
# - `shape` is a tuple with the desired shape of the weight matrix.
#     - The number of rows in the weight matrix should equal the number of columns in the variable `x`.  Since `x` may have 2 dimensions if it represents a single training example (row, col), or three dimensions (batch_size, row, col), get the last dimension from the tuple that holds the dimensions of x.
#     - The number of columns in the weight matrix is the number of units chosen for that dense layer.  Look at the `__init__` function to see which variable stores the number of units.
# - `dtype` is the data type of the values in the generated matrix; keep the default of `tf.float32`. In this case, don't explicitly set the dtype (just let it use the default value).
# 
# Set the standard deviation of the random values to 0.1
# - The values generated have a mean of 0 and standard deviation of 1.
# - Set the default standard deviation `stdev` to be 0.1 by multiplying the standard deviation to each of the values in the weight matrix.

# In[101]:


# See how the trax.fastmath.random.normal function works
tmp_key = trax.fastmath.random.get_prng(seed=1)
print("The random seed generated by random.get_prng")
display(tmp_key)

print("choose a matrix with 2 rows and 3 columns")
tmp_shape=(2,3)
display(tmp_shape)

# Generate a weight matrix
# Note that you'll get an error if you try to set dtype to tf.float32, where tf is tensorflow
# Just avoid setting the dtype and allow it to use the default data type
tmp_weight = trax.fastmath.random.normal(key=tmp_key, shape=tmp_shape)

print("Weight matrix generated with a normal distribution with mean 0 and stdev of 1")
display(tmp_weight)


# <a name="ex-4"></a>
# ### Exercise 4 - Dense
# 
# Implement the `Dense` class.

# In[102]:


# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: Dense
class Dense(Layer):
    """
    A dense (fully-connected) layer.
    """

    # __init__ is implemented for you
    def __init__(self, n_units, init_stdev=0.1):
        
        # Set the number of units in this layer
        self._n_units = n_units
        self._init_stdev = init_stdev

    # Please implement 'forward()'
    def forward(self, x):

        ### START CODE HERE (Replace instances of 'None' with your code) ###

        # Matrix multiply x and the weight matrix
        dense = np.dot(x, self.weights)
        
        ### END CODE HERE ###
        return dense

    # init_weights
    def init_weights_and_state(self, input_signature, random_key):
        
        ### START CODE HERE (Replace instances of 'None' with your code) ###
        # The input_signature has a .shape attribute that gives the shape as a tuple
        input_shape = input_signature.shape

        # Generate the weight matrix from a normal distribution, 
        # and standard deviation of 'stdev'        
        w = self._init_stdev * trax.fastmath.random.normal(key = random_key, shape = (input_shape[-1], self._n_units))
                
        ### END CODE HERE ###   
        self.weights = w
        return self.weights


# In[103]:


# Testing your Dense layer 
dense_layer = Dense(n_units=10)  #sets  number of units in dense layer
random_key = trax.fastmath.random.get_prng(seed=0)  # sets random seed
z = np.array([[2.0, 7.0, 25.0]]) # input array 

dense_layer.init(z, random_key)
print("Weights are\n ",dense_layer.weights) #Returns randomly generated weights
print("Foward function output is ", dense_layer(z)) # Returns multiplied values of units and weights


# ##### Expected Outout
# ```CPP
# Weights are
#   [[-0.02837108  0.09368162 -0.10050076  0.14165013  0.10543301  0.09108126
#   -0.04265672  0.0986188  -0.05575325  0.00153249]
#  [-0.20785688  0.0554837   0.09142365  0.05744595  0.07227863  0.01210617
#   -0.03237354  0.16234995  0.02450038 -0.13809784]
#  [-0.06111237  0.01403724  0.08410042 -0.1094358  -0.10775021 -0.11396459
#   -0.05933381 -0.01557652 -0.03832145 -0.11144515]]
# Foward function output is  [[-3.0395496   0.9266802   2.5414743  -2.050473   -1.9769388  -2.582209
#   -1.7952735   0.94427425 -0.8980402  -3.7497487 ]]
# ```

# In[104]:


# Testing your Dense layer 
dense_layer = Dense(n_units=5)  #sets  number of units in dense layer
random_key = trax.fastmath.random.get_prng(seed=0)  # sets random seed
z = np.array([[-1.0, 10.0, 0.0, 5.0]]) # input array 

dense_layer.init(z, random_key)
print("Weights are\n ",dense_layer.weights) #Returns randomly generated weights
print("Foward function output is ", dense_layer(z)) # Returns multiplied values of units and weights


# In[105]:


w1_unittest.test_Dense(Dense)


# <a name="3-3"></a>
# ### 3.3 - Model
# 
# Now you will implement a classifier using neural networks. Here is the model architecture you will be implementing. 
# 
# <img src = "images/nn.jpg" style="width:400px;height:250px;"/>
# 
# For the model implementation, you will use the Trax `layers` module, imported as `tl`.
# Note that the second character of `tl` is the lowercase of letter `L`, not the number 1. Trax layers are very similar to the ones you implemented above,
# but in addition to trainable weights also have a non-trainable state.
# State is used in layers like batch normalization and for inference, you will learn more about it in course 4.
# 
# First, look at the code of the Trax Dense layer and compare to your implementation above.
# - [tl.Dense](https://github.com/google/trax/blob/master/trax/layers/core.py#L29): Trax Dense layer implementation
# 
# One other important layer that you will use a lot is one that allows to execute one layer after another in sequence.
# - [tl.Serial](https://github.com/google/trax/blob/master/trax/layers/combinators.py#L26): Combinator that applies layers serially.  
#     - You can pass in the layers as arguments to `Serial`, separated by commas. 
#     - For example: `tl.Serial(tl.Embeddings(...), tl.Mean(...), tl.Dense(...), tl.LogSoftmax(...))`
# 
# Please use the `help` function to view documentation for each layer.

# In[106]:


# View documentation on tl.Dense
help(tl.Dense)


# In[107]:


# View documentation on tl.Serial
help(tl.Serial)


# - [tl.Embedding](https://github.com/google/trax/blob/1372b903bb66b0daccee19fd0b1fdf44f659330b/trax/layers/core.py#L113): Layer constructor function for an embedding layer.  
#     - `tl.Embedding(vocab_size, d_feature)`.
#     - `vocab_size` is the number of unique words in the given vocabulary.
#     - `d_feature` is the number of elements in the word embedding (some choices for a word embedding size range from 150 to 300, for example).    

# In[108]:


# View documentation for tl.Embedding
help(tl.Embedding)


# In[109]:


# An example of and embedding layer
rnd.seed(31)
tmp_embed = tl.Embedding(d_feature=2, vocab_size=3)
display(tmp_embed)


# In[110]:


# Let's assume as an example, a batch of two lists
# each list represents a set of tokenized words.
tmp_in_arr = np.array([[0,1,2],
                    [3,2,0]
                   ])

# In order to use the layer, we need to initialize its signature
tmp_embed.init(trax.shapes.signature(tmp_in_arr))

# Embedding layer will return an array of shape (batch size, sequence length, d_feature)
tmp_embedded_arr = tmp_embed(tmp_in_arr)

print(f"Shape of returned array is {tmp_embedded_arr.shape}")
display(tmp_embedded_arr)


# - [tl.Mean](https://github.com/google/trax/blob/1372b903bb66b0daccee19fd0b1fdf44f659330b/trax/layers/core.py#L276): Calculates means across an axis.  In this case, please choose axis = 1 to get an average embedding vector (an embedding vector that is an average of all words in the sentence).  
# - For example, if the embedding matrix is 300 elements and vocab size is 10,000 words, taking the mean of the embedding matrix along axis=1 will yield a vector of 300 elements.

# In[111]:


# view the documentation for tl.mean
help(tl.Mean)


# In[112]:


# Pretend the embedding matrix uses 
# 2 features for embedding the meaning of a word
# and you have a sentence of 3 words
# So the output of the embedding layer has shape (3,2), (sentence length, d_feature)
tmp_embeded = np.array([[1,2],
                        [3,4],
                        [5,6]])

# take the mean along axis 0
print("The mean along axis 0 creates a vector whose length equals the number of features in a word embedding")
display(np.mean(tmp_embeded,axis=0))

print("The mean along axis 1 creates a vector whose length equals the number of words in a sentence")
display(np.mean(tmp_embeded,axis=1))


# **Online documentation**
# 
# - [tl.Dense](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.Dense)
# 
# - [tl.Serial](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#module-trax.layers.combinators)
# 
# - [tl.Embedding](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.Embedding)
# 
# - [tl.Mean](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.Mean)

# <a name="ex-5"></a>
# ### Exercise 5 - classifier
# Implement the classifier function. 

# In[113]:


# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: classifier
def classifier(vocab_size=9088, embedding_dim=256, output_dim=2, mode='train'):
    
    ### START CODE HERE (Replace instances of 'None' with your code) ###
        
    # create embedding layer
    embed_layer = tl.Embedding( 
        vocab_size=vocab_size, # Size of the vocabulary
        d_feature=embedding_dim # Embedding dimension
    )
    
    # Create a mean layer, to create an "average" word embedding
    mean_layer = tl.Mean(1)
    
    # Create a dense layer, one unit for each output
    dense_output_layer = tl.Dense(n_units = output_dim)
    
    # Use tl.Serial to combine all layers
    # and create the classifier
    # of type trax.layers.combinators.Serial
    model = tl.Serial( 
      embed_layer, # embedding layer
      mean_layer, # mean layer
      dense_output_layer
    ) 
    ### END CODE HERE ###
    
    # return the model of type
    return model


# In[114]:


tmp_model = classifier(vocab_size=len(Vocab))


# In[115]:


print(type(tmp_model))
display(tmp_model)


# ##### Expected Outout
# ```python
# <class 'trax.layers.combinators.Serial'>
# Serial[
#   Embedding_9088_256
#   Mean
#   Dense_2
# ]
# ```

# In[116]:


w1_unittest.test_classifier(classifier)


# <a name="4"></a>
# ## 4 - Training
# 
# To train a model on a task, Trax defines an abstraction [`trax.supervised.training.TrainTask`](https://trax-ml.readthedocs.io/en/latest/trax.supervised.html#trax.supervised.training.TrainTask) which packages the train data, loss and optimizer (among other things) together into an object.
# 
# Similarly to evaluate a model, Trax defines an abstraction [`trax.supervised.training.EvalTask`](https://trax-ml.readthedocs.io/en/latest/trax.supervised.html#trax.supervised.training.EvalTask) which packages the eval data and metrics (among other things) into another object.
# 
# The final piece tying things together is the [`trax.supervised.training.Loop`](https://trax-ml.readthedocs.io/en/latest/trax.supervised.html#trax.supervised.training.Loop) abstraction that is a very simple and flexible way to put everything together and train the model, all the while evaluating it and saving checkpoints.
# Using `Loop` will save you a lot of code compared to always writing the training loop by hand, like you did in courses 1 and 2. More importantly, you are less likely to have a bug in that code that would ruin your training.

# In[117]:


# View documentation for trax.supervised.training.TrainTask
help(trax.supervised.training.TrainTask)


# In[118]:


# View documentation for trax.supervised.training.EvalTask
help(trax.supervised.training.EvalTask)


# In[119]:


# View documentation for trax.supervised.training.Loop
help(trax.supervised.training.Loop)


# In[120]:


# View optimizers that you could choose from
help(trax.optimizers)


# Notice some available optimizers include:
# ```CPP
#     adafactor
#     adam
#     momentum
#     rms_prop
#     sm3
# ```

# <a name="4-1"></a>
# ### 4.1  Training the Model
# 
# Now you are going to train your model. 
# 
# Let's define the `TrainTask`, `EvalTask` and `Loop` in preparation to train the model.

# In[121]:


# PLEASE, DO NOT MODIFY OR DELETE THIS CELL
from trax.supervised import training

def get_train_eval_tasks(train_pos, train_neg, val_pos, val_neg, vocab_dict, loop, batch_size = 16):
    
    rnd.seed(271)

    train_task = training.TrainTask(
        labeled_data=train_generator(batch_size, train_pos
                    , train_neg, vocab_dict, loop
                    , shuffle = True),
        loss_layer=tl.WeightedCategoryCrossEntropy(),
        optimizer=trax.optimizers.Adam(0.01),
        n_steps_per_checkpoint=10,
    )

    eval_task = training.EvalTask(
        labeled_data=val_generator(batch_size, val_pos
                    , val_neg, vocab_dict, loop
                    , shuffle = True),        
        metrics=[tl.WeightedCategoryCrossEntropy(), tl.WeightedCategoryAccuracy()],
    )
    
    return train_task, eval_task
    

train_task, eval_task = get_train_eval_tasks(train_pos, train_neg, val_pos, val_neg, Vocab, True, batch_size = 16)
model = classifier()


# In[122]:


model


# This defines a model trained using [`tl.WeightedCategoryCrossEntropy`](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.metrics.WeightedCategoryCrossEntropy) optimized with the [`trax.optimizers.Adam`](https://trax-ml.readthedocs.io/en/latest/trax.optimizers.html#trax.optimizers.adam.Adam) optimizer, all the while tracking the accuracy using [`tl.WeightedCategoryAccuracy`](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.metrics.WeightedCategoryAccuracy) metric. We also track `tl.WeightedCategoryCrossEntropy` on the validation set.

# Now let's make an output directory and train the model.

# In[123]:


dir_path = './model/'

try:
    shutil.rmtree(dir_path)
except OSError as e:
    pass


output_dir = './model/'
output_dir_expand = os.path.expanduser(output_dir)
print(output_dir_expand)


# <a name="ex-6"></a>
# ### Exercise 6 - train_model
# **Instructions:** Implement `train_model` to train the model (`classifier` that you wrote earlier) for the given number of training steps (`n_steps`) using `TrainTask`, `EvalTask` and `Loop`. For the `EvalTask`, take a look to the cell next to the function definition: the `eval_task` is passed as a list explicitly, so take that into account in the implementation of your `train_model` function.

# In[124]:


# UNQ_C6 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: train_model
def train_model(classifier, train_task, eval_task, n_steps, output_dir):
    '''
    Input: 
        classifier - the model you are building
        train_task - Training task
        eval_task - Evaluation task. Received as a list.
        n_steps - the evaluation steps
        output_dir - folder to save your files
    Output:
        trainer -  trax trainer
    '''
    rnd.seed(31) # Do NOT modify this random seed. This makes the notebook easier to replicate
    
    ### START CODE HERE (Replace instances of 'None' with your code) ###          
    training_loop = training.Loop( 
                                classifier, # The learning model
                                train_task, # The training task
                                eval_tasks=eval_task, # The evaluation task
                                output_dir=output_dir, # The output directory
                                random_seed=31 # Do not modify this random seed in order to ensure reproducibility and for grading purposes.
    )

    training_loop.run(n_steps = n_steps)
    ### END CODE HERE ###
    
    # Return the training_loop, since it has the model.
    return training_loop


# In[125]:


# Do not modify this cell.
# Take a look on how the eval_task is inside square brackets and 
# take that into account for you train_model implementation
training_loop = train_model(model, train_task, [eval_task], 100, output_dir_expand)


# ##### Expected output (Approximately)
# 
# ```python
# Step      1: Total number of trainable weights: 2327042
# Step      1: Ran 1 train steps in 1.79 secs
# Step      1: train WeightedCategoryCrossEntropy |  0.69664621
# Step      1: eval  WeightedCategoryCrossEntropy |  0.70276678
# Step      1: eval      WeightedCategoryAccuracy |  0.43750000
# 
# Step     10: Ran 9 train steps in 9.90 secs
# Step     10: train WeightedCategoryCrossEntropy |  0.65194851
# Step     10: eval  WeightedCategoryCrossEntropy |  0.55310017
# Step     10: eval      WeightedCategoryAccuracy |  0.87500000
# 
# Step     20: Ran 10 train steps in 3.03 secs
# Step     20: train WeightedCategoryCrossEntropy |  0.47625321
# Step     20: eval  WeightedCategoryCrossEntropy |  0.35441157
# Step     20: eval      WeightedCategoryAccuracy |  1.00000000
# 
# Step     30: Ran 10 train steps in 1.97 secs
# Step     30: train WeightedCategoryCrossEntropy |  0.26038250
# Step     30: eval  WeightedCategoryCrossEntropy |  0.17245120
# Step     30: eval      WeightedCategoryAccuracy |  1.00000000
# 
# Step     40: Ran 10 train steps in 0.92 secs
# Step     40: train WeightedCategoryCrossEntropy |  0.13840821
# Step     40: eval  WeightedCategoryCrossEntropy |  0.06517925
# Step     40: eval      WeightedCategoryAccuracy |  1.00000000
# 
# Step     50: Ran 10 train steps in 1.87 secs
# Step     50: train WeightedCategoryCrossEntropy |  0.08931129
# Step     50: eval  WeightedCategoryCrossEntropy |  0.05949062
# Step     50: eval      WeightedCategoryAccuracy |  1.00000000
# 
# Step     60: Ran 10 train steps in 0.95 secs
# Step     60: train WeightedCategoryCrossEntropy |  0.04529145
# Step     60: eval  WeightedCategoryCrossEntropy |  0.02183468
# Step     60: eval      WeightedCategoryAccuracy |  1.00000000
# 
# Step     70: Ran 10 train steps in 0.95 secs
# Step     70: train WeightedCategoryCrossEntropy |  0.04261621
# Step     70: eval  WeightedCategoryCrossEntropy |  0.00225742
# Step     70: eval      WeightedCategoryAccuracy |  1.00000000
# 
# Step     80: Ran 10 train steps in 0.97 secs
# Step     80: train WeightedCategoryCrossEntropy |  0.02085698
# Step     80: eval  WeightedCategoryCrossEntropy |  0.00488479
# Step     80: eval      WeightedCategoryAccuracy |  1.00000000
# 
# Step     90: Ran 10 train steps in 1.00 secs
# Step     90: train WeightedCategoryCrossEntropy |  0.04042089
# Step     90: eval  WeightedCategoryCrossEntropy |  0.00711416
# Step     90: eval      WeightedCategoryAccuracy |  1.00000000
# 
# Step    100: Ran 10 train steps in 1.79 secs
# Step    100: train WeightedCategoryCrossEntropy |  0.01717071
# Step    100: eval  WeightedCategoryCrossEntropy |  0.10006869
# Step    100: eval      WeightedCategoryAccuracy |  0.93750000
# ```

# In[126]:


# Test your function. Do not modify this cell.
# Take a look on how the eval_task is inside square brackets.
try:
    shutil.rmtree('./model_test/')
except OSError as e:
    pass

w1_unittest.test_train_model(train_model(classifier(), train_task, [eval_task], 10, './model_test/'))


# <a name="4-2"></a>
# ### 4.2 - Practice Making a Prediction
# 
# Now that you have trained a model, you can access it as `training_loop.model` object. We will actually use `training_loop.eval_model` and in the next weeks you will learn why we sometimes use a different model for evaluation, e.g., one without dropout. For now, make predictions with your model.
# 
# Use the training data just to see how the prediction process works.  
# - Later, you will use validation data to evaluate your model's performance.
# 

# In[127]:


# Create a generator object
tmp_train_generator = train_generator(16, train_pos
                    , train_neg, Vocab, loop=True
                    , shuffle = False)



# get one batch
tmp_batch = next(tmp_train_generator)

# Position 0 has the model inputs (tweets as tensors)
# position 1 has the targets (the actual labels)
tmp_inputs, tmp_targets, tmp_example_weights = tmp_batch

print(f"The batch is a tuple of length {len(tmp_batch)} because position 0 contains the tweets, and position 1 contains the targets.") 
print(f"The shape of the tweet tensors is {tmp_inputs.shape} (num of examples, length of tweet tensors)")
print(f"The shape of the labels is {tmp_targets.shape}, which is the batch size.")
print(f"The shape of the example_weights is {tmp_example_weights.shape}, which is the same as inputs/targets size.")


# In[128]:


# feed the tweet tensors into the model to get a prediction
tmp_pred = training_loop.eval_model(tmp_inputs)
print(f"The prediction shape is {tmp_pred.shape}, num of tensor_tweets as rows")
print("Column 0 is the probability of a negative sentiment (class 0)")
print("Column 1 is the probability of a positive sentiment (class 1)")
print()
print("View the prediction array")
tmp_pred


# To turn these probabilities into categories (negative or positive sentiment prediction), for each row:
# - Compare the probabilities in each column.
# - If column 1 has a value greater than column 0, classify that as a positive tweet.
# - Otherwise if column 1 is less than or equal to column 0, classify that example as a negative tweet.

# In[129]:


# turn probabilites into category predictions
tmp_is_positive = tmp_pred[:,1] > tmp_pred[:,0]
for i, p in enumerate(tmp_is_positive):
    print(f"Neg log prob {tmp_pred[i,0]:.4f}\tPos log prob {tmp_pred[i,1]:.4f}\t is positive? {p}\t actual {tmp_targets[i]}")


# Notice that since you are making a prediction using a training batch, it's more likely that the model's predictions match the actual targets (labels).  
# - Every prediction that the tweet is positive is also matching the actual target of 1 (positive sentiment).
# - Similarly, all predictions that the sentiment is not positive matches the actual target of 0 (negative sentiment)

# One more useful thing to know is how to compare if the prediction is matching the actual target (label).  
# - The result of calculation `is_positive` is a boolean.
# - The target is a type trax.fastmath.numpy.int32
# - If you expect to be doing division, you may prefer to work with decimal numbers with the data type type trax.fastmath.numpy.int32

# In[130]:


# View the array of booleans
print("Array of booleans")
display(tmp_is_positive)

# convert boolean to type int32
# True is converted to 1
# False is converted to 0
tmp_is_positive_int = tmp_is_positive.astype(np.int32)


# View the array of integers
print("Array of integers")
display(tmp_is_positive_int)

# convert boolean to type float32
tmp_is_positive_float = tmp_is_positive.astype(np.float32)

# View the array of floats
print("Array of floats")
display(tmp_is_positive_float)


# Note that Python usually does type conversion for you when you compare a boolean to an integer
# - True compared to 1 is True, otherwise any other integer is False.
# - False compared to 0 is True, otherwise any ohter integer is False.

# In[131]:


print(f"True == 1: {True == 1}")
print(f"True == 2: {True == 2}")
print(f"False == 0: {False == 0}")
print(f"False == 2: {False == 2}")


# However, we recommend that you keep track of the data type of your variables to avoid unexpected outcomes.  So it helps to convert the booleans into integers
# - Compare 1 to 1 rather than comparing True to 1.

# Hopefully you are now familiar with what kinds of inputs and outputs the model uses when making a prediction.
# - This will help you implement a function that estimates the accuracy of the model's predictions.

# <a name="5"></a>
# ## 5 - Evaluation  
# 
# <a name="5-1"></a>
# ### 5.1 - Computing the Accuracy on a Batch
# 
# You will now write a function that evaluates your model on the validation set and returns the accuracy. 
# - `preds` contains the predictions.
#     - Its dimensions are `(batch_size, output_dim)`.  `output_dim` is two in this case.  Column 0 contains the probability that the tweet belongs to class 0 (negative sentiment). Column 1 contains probability that it belongs to class 1 (positive sentiment).
#     - If the probability in column 1 is greater than the probability in column 0, then interpret this as the model's prediction that the example has label 1 (positive sentiment).  
#     - Otherwise, if the probabilities are equal or the probability in column 0 is higher, the model's prediction is 0 (negative sentiment).
# - `y` contains the actual labels.
# - `y_weights` contains the weights to give to predictions.

# <a name="ex-7"></a>
# ### Exercise 7 - compute_accuracy
# Implement `compute_accuracy`.

# In[132]:


# UNQ_C7 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: compute_accuracy
def compute_accuracy(preds, y, y_weights):
    """
    Input: 
        preds: a tensor of shape (dim_batch, output_dim) 
        y: a tensor of shape (dim_batch,) with the true labels
        y_weights: a n.ndarray with the a weight for each example
    Output: 
        accuracy: a float between 0-1 
        weighted_num_correct (np.float32): Sum of the weighted correct predictions
        sum_weights (np.float32): Sum of the weights
    """
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    # Create an array of booleans, 
    # True if the probability of positive sentiment is greater than
    # the probability of negative sentiment
    # else False
    is_pos = preds[:, 1]>preds[:, 0]

    # convert the array of booleans into an array of np.int32
    is_pos_int = is_pos.astype(np.int32)
    
    # compare the array of predictions (as int32) with the target (labels) of type int32
    correct = y==is_pos_int

    # Count the sum of the weights.
    sum_weights = np.sum(y_weights)
    
    # convert the array of correct predictions (boolean) into an arrayof np.float32
    correct_float = correct.astype(np.float32)
    
    # Multiply each prediction with its corresponding weight.
    weighted_correct_float = correct_float@y_weights


    # Sum up the weighted correct predictions (of type np.float32), to go in the
    # numerator.
    weighted_num_correct = np.sum(weighted_correct_float)

    # Divide the number of weighted correct predictions by the sum of the
    # weights.
    accuracy = weighted_num_correct/sum_weights

    ### END CODE HERE ###
    return accuracy, weighted_num_correct, sum_weights


# In[133]:


# test your function
tmp_val_generator = val_generator(64, val_pos
                    , val_neg, Vocab, loop=True
                    , shuffle = False)

# get one batch
tmp_batch = next(tmp_val_generator)

# Position 0 has the model inputs (tweets as tensors)
# position 1 has the targets (the actual labels)
tmp_inputs, tmp_targets, tmp_example_weights = tmp_batch

# feed the tweet tensors into the model to get a prediction
tmp_pred = training_loop.eval_model(tmp_inputs)
tmp_acc, tmp_num_correct, tmp_num_predictions = compute_accuracy(preds=tmp_pred, y=tmp_targets, y_weights=tmp_example_weights)

print(f"Model's prediction accuracy on a single training batch is: {100 * tmp_acc}%")
print(f"Weighted number of correct predictions {tmp_num_correct}; weighted number of total observations predicted {tmp_num_predictions}")


# ##### Expected output (Approximately)
# 
# ```
# Model's prediction accuracy on a single training batch is: 100.0%
# Weighted number of correct predictions 64.0; weighted number of total observations predicted 64
# ```

# In[134]:


# Test your function
w1_unittest.test_compute_accuracy(compute_accuracy)


# <a name="5-2"></a>
# ### 5.2 - Testing your Model on Validation Data
# 
# Now you will write a test function to check your model's prediction accuracy on validation data. 
# 
# This program will take in a data generator and your model. 
# - The generator allows you to get batches of data. You can use it with a `for` loop:
# 
# ```
# for batch in iterator: 
#    # do something with that batch
# ```
# 
# `batch` has `3` elements:
# - the first element contains the inputs
# - the second element contains the targets
# - the third element contains the weights

# <a name="ex-8"></a>
# ### Exercise 8 - test_model
# 
# **Instructions:** 
# - Compute the accuracy over all the batches in the validation iterator. 
# - Make use of `compute_accuracy`, which you recently implemented, and return the overall accuracy.

# In[135]:


# UNQ_C8 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: test_model
def test_model(generator, model, compute_accuracy=compute_accuracy):
    '''
    Input: 
        generator: an iterator instance that provides batches of inputs and targets
        model: a model instance 
    Output: 
        accuracy: float corresponding to the accuracy
    '''
    
    accuracy = 0.
    total_num_correct = 1
    total_num_pred = 1
        
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    for batch in generator: 
        
        # Retrieve the inputs from the batch
        inputs = batch[0]

        # Retrieve the targets (actual labels) from the batch
        targets = batch[1]
        
        # Retrieve the example weight.
        example_weight = batch[2]

        # Make predictions using the inputs
        pred = model(inputs)
        
        # Calculate accuracy for the batch by comparing its predictions and targets
        batch_accuracy, batch_num_correct, batch_num_pred = compute_accuracy(pred, targets, example_weight)
                
        # Update the total number of correct predictions
        # by adding the number of correct predictions from this batch
        total_num_correct += batch_num_correct
        
        # Update the total number of predictions 
        # by adding the number of predictions made for the batch
        total_num_pred += batch_num_pred

    # Calculate accuracy over all examples
    accuracy = total_num_correct/total_num_pred
    
    ### END CODE HERE ###
    return accuracy


# In[136]:


# DO NOT EDIT THIS CELL
# testing the accuracy of your model: this takes around 20 seconds
model = training_loop.eval_model
accuracy = test_model(test_generator(16, val_pos
                    , val_neg, Vocab, loop=False
                    , shuffle = False), model)

print(f'The accuracy of your model on the validation set is {accuracy:.4f}', )


# ##### Expected Output (Approximately)
# 
# ```CPP
# The accuracy of your model on the validation set is 0.9950
# ```

# In[137]:


w1_unittest.unittest_test_model(test_model, test_generator(16, val_pos , val_neg, Vocab, loop=False, shuffle = False), model)


# <a name="6"></a>
# ## 6 - Testing with your Own Input
# 
# Finally you will test with your own input. You will see that deepnets are more powerful than the older methods you have used before. Although you go close to 100% accuracy on the first two assignments, the task was way easier. 

# In[138]:


# this is used to predict on your own sentnece
def predict(sentence):
    inputs = np.array(tweet_to_tensor(sentence, vocab_dict=Vocab))
    
    # Batch size 1, add dimension for batch, to work with the model
    inputs = inputs[None, :]  
    
    # predict with the model
    preds_probs = model(inputs)
    
    # Turn probabilities into categories
    preds = int(preds_probs[0, 1] > preds_probs[0, 0])
    
    sentiment = "negative"
    if preds == 1:
        sentiment = 'positive'

    return preds, sentiment


# In[139]:


# try a positive sentence
sentence = "It's such a nice day, I think I'll be taking Sid to Ramsgate for lunch and then to the beach maybe."
tmp_pred, tmp_sentiment = predict(sentence)
print(f"The sentiment of the sentence \n***\n\"{sentence}\"\n***\nis {tmp_sentiment}.")

print()
# try a negative sentence
sentence = "I hated my day, it was the worst, I'm so sad."
tmp_pred, tmp_sentiment = predict(sentence)
print(f"The sentiment of the sentence \n***\n\"{sentence}\"\n***\nis {tmp_sentiment}.")


# Notice that the model works well even for complex sentences.

# <a name="7"></a>
# ## 7 - Word Embeddings

# In this section, you will visualize the word embeddings that were constructed for this sentiment analysis task. You can retrieve them by looking at the `model.weights` tuple (recall that the first layer of the model is the embedding layer).

# In[140]:


embeddings = model.weights[0]


# Let's take a look at the size of the embeddings. 

# In[141]:


embeddings.shape


# To visualize the word embeddings, it is necessary to choose 2 directions to use as axes for the plot. You could use random directions or the first two eigenvectors from PCA. Here, you'll use scikit-learn to perform dimensionality reduction of the word embeddings using PCA. 

# In[142]:


from sklearn.decomposition import PCA #Import PCA from scikit-learn
pca = PCA(n_components=2) #PCA with two dimensions

emb_2dim = pca.fit_transform(embeddings) #Dimensionality reduction of the word embeddings


# Now, everything is ready to plot a selection of words in 2d. 

# In[143]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#Selection of negative and positive words
neg_words = ['worst', 'bad', 'hurt', 'sad', 'hate']
pos_words = ['best', 'good', 'nice', 'better', 'love']

#Index of each selected word
neg_n = [Vocab[w] for w in neg_words]
pos_n = [Vocab[w] for w in pos_words]

plt.figure()

#Scatter plot for negative words
plt.scatter(emb_2dim[neg_n][:,0],emb_2dim[neg_n][:,1], color = 'r')
for i, txt in enumerate(neg_words): 
    plt.annotate(txt, (emb_2dim[neg_n][i,0],emb_2dim[neg_n][i,1]))

#Scatter plot for positive words
plt.scatter(emb_2dim[pos_n][:,0],emb_2dim[pos_n][:,1], color = 'g')
for i, txt in enumerate(pos_words): 
    plt.annotate(txt,(emb_2dim[pos_n][i,0],emb_2dim[pos_n][i,1]))

plt.title('Word embeddings in 2d')

plt.show()


# As you can see, the word embeddings for this task seem to distinguish negative and positive meanings very well. However, clusters don't necessarily have similar words since you only trained the model to analyze overall sentiment. 

# ### On Deep Nets
# 
# Deep nets allow you to understand and capture dependencies that you would have not been able to capture with a simple linear regression, or logistic regression. 
# - It also allows you to better use pre-trained embeddings for classification and tends to generalize better.
