#!/usr/bin/env python
# coding: utf-8

# # Working with JAX numpy and calculating perplexity: Ungraded Lecture Notebook

# Normally you would import `numpy` and rename it as `np`. 
# 
# However in this week's assignment you will notice that this convention has been changed. 
# 
# Now standard `numpy` is not renamed and `trax.fastmath.numpy` is renamed as `np`. 
# 
# The rationale behind this change is that you will be using Trax's numpy (which is compatible with JAX) far more often. Trax's numpy supports most of the same functions as the regular numpy so the change won't be noticeable in most cases.
# 

# In[1]:


import numpy
import trax
import trax.fastmath.numpy as np

# Setting random seeds
numpy.random.seed(32)


# One important change to take into consideration is that the types of the resulting objects will be different depending on the version of numpy. With regular numpy you get `numpy.ndarray` but with Trax's numpy you will get `jax.interpreters.xla.DeviceArray`. These two types map to each other. So if you find some error logs mentioning DeviceArray type, don't worry about it, treat it like you would treat an ndarray and march ahead.
# 
# You can get a randomized numpy array by using the `numpy.random.random()` function.
# 
# This is one of the functionalities that Trax's numpy does not currently support in the same way as the regular numpy. 

# In[2]:


numpy_array = numpy.random.random((5,10))
print(f"The regular numpy array looks like this:\n\n {numpy_array}\n")
print(f"It is of type: {type(numpy_array)}")


# You can easily cast regular numpy arrays or lists into trax numpy arrays using the `trax.fastmath.numpy.array()` function:

# In[3]:


trax_numpy_array = np.array(numpy_array)
print(f"The trax numpy array looks like this:\n\n {trax_numpy_array}\n")
print(f"It is of type: {type(trax_numpy_array)}")


# Hope you now understand the differences (and similarities) between these two versions and numpy. **Great!**
# 
# The previous section was a quick look at Trax's numpy. However this notebook also aims to teach you how you can calculate the perplexity of a trained model.
# 

# ## Calculating Perplexity

# The perplexity is a metric that measures how well a probability model predicts a sample and it is commonly used to evaluate language models. It is defined as: 
# 
# $$P(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i| w_1,...,w_{i-1})}}$$
# 
# As an implementation hack, you would usually take the log of that formula (so the computation is less prone to underflow problems). You would also need to take care of the padding, since you do not want to include the padding when calculating the perplexity (to avoid an artificially good metric).
# 
# After taking the logarithm of $P(W)$ you have:
# 
# $$log P(W) = {\log\left(\sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i| w_1,...,w_{i-1})}}\right)}$$
# 
# 
# $$ = \log\left(\left(\prod_{i=1}^{N} \frac{1}{P(w_i| w_1,...,w_{i-1})}\right)^{\frac{1}{N}}\right)$$
# 
# $$ = \log\left(\left({\prod_{i=1}^{N}{P(w_i| w_1,...,w_{i-1})}}\right)^{-\frac{1}{N}}\right)$$
# 
# $$ = -\frac{1}{N}{\log\left({\prod_{i=1}^{N}{P(w_i| w_1,...,w_{i-1})}}\right)} $$
# 
# $$ = -\frac{1}{N}{{\sum_{i=1}^{N}{\log P(w_i| w_1,...,w_{i-1})}}} $$
# 

# You will be working with a real example from this week's assignment. The example is made up of:
#    - `predictions` : log probabilities for each element in the vocabulary for 32 sequences with 64 elements (after padding).
#    - `targets` : 32 observed sequences of 64 elements (after padding).

# In[4]:


from trax import layers as tl

# Load from .npy files
predictions = numpy.load('predictions.npy')
targets = numpy.load('targets.npy')

# Cast to jax.interpreters.xla.DeviceArray
predictions = np.array(predictions)
targets = np.array(targets)

# Print shapes
print(f'predictions has shape: {predictions.shape}')
print(f'targets has shape: {targets.shape}')


# Notice that the predictions have an extra dimension with the same length as the size of the vocabulary used.
# 
# Because of this you will need a way of reshaping `targets` to match this shape. For this you can use `trax.layers.one_hot()`.
# 
# Notice that `predictions.shape[-1]` will return the size of the last dimension of `predictions`.

# In[5]:


reshaped_targets = tl.one_hot(targets, predictions.shape[-1]) #trax's one_hot function takes the input as one_hot(x, n_categories, dtype=optional)
print(f'reshaped_targets has shape: {reshaped_targets.shape}')


# By calculating the product of the predictions and the reshaped targets and summing across the last dimension, the total log propbability of each observed element within the sequences can be computed:

# In[6]:


log_p = np.sum(predictions * reshaped_targets, axis= -1)


# Now you will need to account for the padding so this metric is not artificially deflated (since a lower perplexity means a better model). For identifying which elements are padding and which are not, you can use `np.equal()` and get a tensor with `1s` in the positions of actual values and `0s` where there are paddings.

# In[7]:


non_pad = 1.0 - np.equal(targets, 0)
print(f'non_pad has shape: {non_pad.shape}\n')
print(f'non_pad looks like this: \n\n {non_pad}')


# By computing the product of the log probabilities and the non_pad tensor you remove the effect of padding on the metric:

# In[8]:


real_log_p = log_p * non_pad
print(f'real log probabilities still have shape: {real_log_p.shape}')


# You can check the effect of filtering out the padding by looking at the two log probabilities tensors:

# In[9]:


print(f'log probabilities before filtering padding: \n\n {log_p}\n')
print(f'log probabilities after filtering padding: \n\n {real_log_p}')


# Finally, to get the average log perplexity of the model across all sequences in the batch, you will sum the log probabilities in each sequence and divide by the number of non padding elements (which will give you the negative log perplexity per sequence). After that, you can get the mean of the log perplexity across all sequences in the batch.

# In[10]:


log_ppx = np.sum(real_log_p, axis=1) / np.sum(non_pad, axis=1)
log_ppx = np.mean(-log_ppx)
print(f'The log perplexity and perplexity of the model are respectively: {log_ppx} and {np.exp(log_ppx)}')


# **Congratulations on finishing this lecture notebook!** Now you should have a clear understanding of how to work with Trax's numpy and how to compute the perplexity to evaluate your language models. **Keep it up!**
