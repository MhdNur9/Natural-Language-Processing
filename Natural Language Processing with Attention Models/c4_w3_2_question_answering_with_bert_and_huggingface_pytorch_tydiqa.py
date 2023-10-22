# -*- coding: utf-8 -*-
"""C4_W3_2_Question_Answering_with_BERT_and_HuggingFace_Pytorch_tydiqa.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1P8COnbYLphJNaW3v8wS1AwpahnV-653A

# Question Answering with BERT and HuggingFace 🤗 (Fine-tuning)

In the previous Hugging Face ungraded lab, you saw how to use the pipeline objects to use transformer models for NLP tasks. I showed you that the model didn't output the desired answers to a series of precise questions for a context related to the history of comic books.

In this lab, you will fine-tune the model from that lab to give better answers for that type of context. To do that, you'll be using the [TyDi QA dataset](https://ai.google.com/research/tydiqa) but on a filtered version with only English examples. Additionally, you will use a lot of the tools that Hugging Face has to offer.

You have to note that, in general, you will fine-tune general-purpose transformer models to work for specific tasks. However, fine-tuning a general-purpose model can take a lot of time. That's why you will be using the model from the question answering pipeline in this lab.

First, let's install some packages that you will use during the lab.
"""

!pip install transformers datasets torch;
!pip install transformers[torch]
!pip install accelerate -U

"""## Fine-tuning a BERT model

As you saw in the previous lab, you can use these pipelines as they are. But sometimes, you'll need something more specific to your problem, or maybe you need it to perform better on your production data. In these cases, you'll need to fine-tune a model.

Here, you'll fine-tune a pre-trained DistilBERT model on the TyDi QA dataset.

To fine-tune your model, you will leverage three components provided by Hugging Face:

* Datasets: Library that contains some datasets and different metrics to evaluate the performance of your models.
* Tokenizer: Object in charge of preprocessing your text to be given as input for the transformer models.
* Transformers: Library with the pre-trained model checkpoints and the trainer object.

### Datasets

To get the dataset to fine-tune your model, you will use [🤗 Datasets](https://huggingface.co/docs/datasets/), a lightweight and extensible library to share and access datasets and evaluation metrics for NLP easily. You can download Hugging Face datasets directly using the `load_dataset` function from the `datasets` library. Although the most common approach is to use `load_dataset`, for this lab you will use a filtered version containing only the English examples. You can read them from a public GCP bucket and use the `load_from_disk` function.

Hugging Face `datasets` allows to load data in several formats, such as CSV, JSON, text files and even parquet. You can see more about the supported formats in the [documentation](https://huggingface.co/docs/datasets/loading.html)

We already prepared the dataset for you, so you don't need to uncomment the code from the cell below if you don't want to load all the data and then filter the English examples. If you want to download the dataset by yourself, you can uncomment the following cell and then jump to the [cell](#datasets_type) in which you can see the type of object you get after loading the dataset.
"""

# You can download the dataset and process it to obtain the same dataset we are loading from disk
# Uncomment the following lines to download the dataset directly
# from datasets import load_dataset
# train_data = load_dataset('tydiqa', 'primary_task')
# tydiqa_data =  train_data.filter(lambda example: example['language'] == 'english')

"""If you want to use the dataset provided by us, please run the following cells. First, we will download the dataset from the GCP bucket."""

# Download dataset from bucket.
!wget https://storage.googleapis.com/nlprefresh-public/tydiqa_data.zip

# Uncomment if you want to check the size of the file. It should be around 319M.
#!ls -alh tydiqa_data.zip

"""Now, let's unzip the dataset"""

# Unzip inside the dataset folder
!unzip tydiqa_data

"""Given that we used Apache Arrow format to save the dataset, you have to use the `load_from_disk` function from the `datasets` library to load it. To access the preprocessed dataset we created, you should execute the following commands."""

# Execute this cell if you will use the data we processed instead of downloading it.
from datasets import load_from_disk

#The path where the dataset is stored
path = '/content/tydiqa_data/'

#Load Dataset
tydiqa_data = load_from_disk(path)

tydiqa_data

"""<a id='datasets_type'></a>
You can check below that the type of the loaded dataset is a `datasets.arrow_dataset.Dataset`. This object type corresponds to an Apache Arrow Table that allows creating a hash table that contains the position in memory where data is stored instead of loading the complete dataset into memory. But you don't have to worry too much about that. It is just an efficient way to work with lots of data.
"""

# Checking the object type for one of the elements in the dataset
type(tydiqa_data['train'])

"""You can also check the structure of the dataset:"""

tydiqa_data['train']

"""You can see that each example is like a dictionary object. This dataset consists of questions, contexts, and indices that point to the start and end position of the answer inside the context. You can access the index using the `annotations` key, which is a kind of dictionary."""

idx = 600

# start index
start_index = tydiqa_data['train'][idx]['annotations']['minimal_answers_start_byte'][0]

# end index
end_index = tydiqa_data['train'][idx]['annotations']['minimal_answers_end_byte'][0]

print("Question: " + tydiqa_data['train'][idx]['question_text'])
print("\nContext (truncated): "+ tydiqa_data['train'][idx]['document_plaintext'][0:512] + '...')
print("\nAnswer: " + tydiqa_data['train'][idx]['document_plaintext'][start_index:end_index])

"""The question answering model predicts a start and endpoint in the context to extract as the answer. That's why this NLP task is known as extractive question answering.

To train your model, you need to pass start and endpoints as labels. So, you need to implement a function that extracts the start and end positions from the dataset.

The dataset contains unanswerable questions. For these, the start and end indices for the answer are equal to `-1`.
"""

tydiqa_data['train'][0]['annotations']

"""Now, you have to flatten the dataset to work with an object with a table structure instead of a dictionary structure. This step facilitates the pre-processing steps."""

# Flattening the datasets
flattened_train_data = tydiqa_data['train'].flatten()
flattened_test_data =  tydiqa_data['validation'].flatten()

"""Also, to make the training more straightforward and faster, we will extract a subset of the train and test datasets. For that purpose, we will use the Hugging Face Dataset object's method called `select()`. This method allows you to take some data points by their index. Here, you will select the first 3000 rows; you can play with the number of data points but consider that this will increase the training time."""

# Selecting a subset of the train dataset
flattened_train_data = flattened_train_data.select(range(3000))

# Selecting a subset of the test dataset
flattened_test_data = flattened_test_data.select(range(1000))

"""### Tokenizers

Now, you will use the [tokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html) object from Hugging Face. You can load a tokenizer using different methods. Here, you will retrieve it from the pipeline object you created in the previous Hugging Face lab. With this tokenizer, you can ensure that the tokens you get for the dataset will match the tokens used in the original DistilBERT implementation.

When loading a tokenizer with any method, you must pass the model checkpoint that you want to fine-tune. Here, you are using the`'distilbert-base-cased-distilled-squad'` checkpoint.

"""

# Import the AutoTokenizer from the transformers library
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")

"""Given the characteristics of the dataset and the question-answering task, you will need to add some steps to pre-process the data after the tokenization:

1. When there is no answer to a question given a context, you will use the `CLS` token, a unique token used to represent the start of the sequence.

2.  Tokenizers can split a given string into substrings, resulting in a subtoken for each substring, creating misalignment between the list of dataset tags and the labels generated by the tokenizer. Therefore, you will need to align the start and end indices with the tokens associated with the target answer word.

3. Finally, a tokenizer can truncate a very long sequence. So, if the start/end position of an answer is `None`, you will assume that it was truncated and assign the maximum length of the tokenizer to those positions.

Those three steps are done within the `process_samples` function defined below.
"""

# Processing samples using the 3 steps described.
def process_samples(sample):
    tokenized_data = tokenizer(sample['document_plaintext'], sample['question_text'], truncation="only_first", padding="max_length")

    input_ids = tokenized_data["input_ids"]

    # We will label impossible answers with the index of the CLS token.
    cls_index = input_ids.index(tokenizer.cls_token_id)

    # If no answers are given, set the cls_index as answer.
    if sample["annotations.minimal_answers_start_byte"][0] == -1:
        start_position = cls_index
        end_position = cls_index
    else:
        # Start/end character index of the answer in the text.
        gold_text = sample["document_plaintext"][sample['annotations.minimal_answers_start_byte'][0]:sample['annotations.minimal_answers_end_byte'][0]]
        start_char = sample["annotations.minimal_answers_start_byte"][0]
        end_char = sample['annotations.minimal_answers_end_byte'][0] #start_char + len(gold_text)

        # sometimes answers are off by a character or two – fix this
        if sample['document_plaintext'][start_char-1:end_char-1] == gold_text:
            start_char = start_char - 1
            end_char = end_char - 1     # When the gold label is off by one character
        elif sample['document_plaintext'][start_char-2:end_char-2] == gold_text:
            start_char = start_char - 2
            end_char = end_char - 2     # When the gold label is off by two characters

        start_token = tokenized_data.char_to_token(start_char)
        end_token = tokenized_data.char_to_token(end_char - 1)

        # if start position is None, the answer passage has been truncated
        if start_token is None:
            start_token = tokenizer.model_max_length
        if end_token is None:
            end_token = tokenizer.model_max_length

        start_position = start_token
        end_position = end_token

    return {'input_ids': tokenized_data['input_ids'],
          'attention_mask': tokenized_data['attention_mask'],
          'start_positions': start_position,
          'end_positions': end_position}

"""To apply the `process_samples` function defined above to the whole  dataset, you can use the `map` method as follows:"""

# Tokenizing and processing the flattened dataset
processed_train_data = flattened_train_data.map(process_samples)
processed_test_data = flattened_test_data.map(process_samples)

"""# Transformers

The last component of Hugging Face that is useful for fine-tuning a transformer corresponds to the pre-trained models you can access in multiple ways.

For this lab, you will use the same model from the question-answering pipeline that you loaded before.
"""

# Import the AutoModelForQuestionAnswering for the pre-trained model. We will only fine tune the head of the model
from transformers import AutoModelForQuestionAnswering
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

"""Now, you can take the necessary columns from the datasets to train/test and return them as Pytorch Tensors."""

columns_to_return = ['input_ids','attention_mask', 'start_positions', 'end_positions']
processed_train_data.set_format(type='pt', columns=columns_to_return)
processed_test_data.set_format(type='pt', columns=columns_to_return)

"""Here, we give you the F1 score as a metric to evaluate your model's performance. We will use this metric for simplicity, although it is based on the start and end values predicted by the model. If you want to dig deeper on other metrics that can be used for a question and answering task, you can also check [this colab notebook resource](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/question_answering.ipynb) from the Hugging Face team."""

from sklearn.metrics import f1_score

def compute_f1_metrics(pred):
    start_labels = pred.label_ids[0]
    start_preds = pred.predictions[0].argmax(-1)
    end_labels = pred.label_ids[1]
    end_preds = pred.predictions[1].argmax(-1)

    f1_start = f1_score(start_labels, start_preds, average='macro')
    f1_end = f1_score(end_labels, end_preds, average='macro')

    return {
        'f1_start': f1_start,
        'f1_end': f1_end,
    }

"""Now, you will use the Hugging Face [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) to fine-tune your model."""

# Training the model may take around 15 minutes.
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='model_results5',          # output directory
    overwrite_output_dir=True,
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=20,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir=None,            # directory for storing logs
    logging_steps=50
)

trainer = Trainer(
    model=model, # the instantiated 🤗 Transformers model to be trained
    args=training_args, # training arguments, defined above
    train_dataset=processed_train_data, # training dataset
    eval_dataset=processed_test_data, # evaluation dataset
    compute_metrics=compute_f1_metrics
)

trainer.train()

"""And, in the next cell, you can evaluate the fine-tuned model's performance on the test set."""

# The evaluation may take around 30 seconds
trainer.evaluate(processed_test_data)

"""### Using your Fine-Tuned Model

After training and evaluating your fine-tuned model, you can check its results for the same questions from the previous lab.

For that, you will tell Pytorch to use your GPU or your CPU to run the model. Additionally, you will need to tokenize your input context and questions. Finally, you need to post-process the output results to transform them from tokens to human-readable strings using the `tokenizer`.
"""

import torch

text = r"""
The Golden Age of Comic Books describes an era of American comic books from the
late 1930s to circa 1950. During this time, modern comic books were first published
and rapidly increased in popularity. The superhero archetype was created and many
well-known characters were introduced, including Superman, Batman, Captain Marvel
(later known as SHAZAM!), Captain America, and Wonder Woman.
Between 1939 and 1941 Detective Comics and its sister company, All-American Publications,
introduced popular superheroes such as Batman and Robin, Wonder Woman, the Flash,
Green Lantern, Doctor Fate, the Atom, Hawkman, Green Arrow and Aquaman.[7] Timely Comics,
the 1940s predecessor of Marvel Comics, had million-selling titles featuring the Human Torch,
the Sub-Mariner, and Captain America.[8]
As comic books grew in popularity, publishers began launching titles that expanded
into a variety of genres. Dell Comics' non-superhero characters (particularly the
licensed Walt Disney animated-character comics) outsold the superhero comics of the day.[12]
The publisher featured licensed movie and literary characters such as Mickey Mouse, Donald Duck,
Roy Rogers and Tarzan.[13] It was during this era that noted Donald Duck writer-artist
Carl Barks rose to prominence.[14] Additionally, MLJ's introduction of Archie Andrews
in Pep Comics #22 (December 1941) gave rise to teen humor comics,[15] with the Archie
Andrews character remaining in print well into the 21st century.[16]
At the same time in Canada, American comic books were prohibited importation under
the War Exchange Conservation Act[17] which restricted the importation of non-essential
goods. As a result, a domestic publishing industry flourished during the duration
of the war which were collectively informally called the Canadian Whites.
The educational comic book Dagwood Splits the Atom used characters from the comic
strip Blondie.[18] According to historian Michael A. Amundson, appealing comic-book
characters helped ease young readers' fear of nuclear war and neutralize anxiety
about the questions posed by atomic power.[19] It was during this period that long-running
humor comics debuted, including EC's Mad and Carl Barks' Uncle Scrooge in Dell's Four
Color Comics (both in 1952).[20][21]
"""

questions = ["What superheroes were introduced between 1939 and 1941 by Detective Comics and its sister company?",
             "What comic book characters were created between 1939 and 1941?",
             "What well-known characters were created between 1939 and 1941?",
             "What well-known superheroes were introduced between 1939 and 1941 by Detective Comics?"]

for question in questions:
    inputs = tokenizer.encode_plus(question, text, return_tensors="pt")
    #print("inputs", inputs)
    #print("inputs", type(inputs))
    input_ids = inputs["input_ids"].tolist()[0]
    inputs.to("cuda")

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_model = model(**inputs)

    answer_start = torch.argmax(
        answer_model['start_logits']
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_model['end_logits']) + 1  # Get the most likely end of answer with the argmax of the score

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    print(f"Question: {question}")
    print(f"Answer: {answer}\n")

"""You can compare those results with those obtained using the pipeline, as you did in the previous lab. As a reminder, here are those results:

```
What popular superheroes were introduced between 1939 and 1941?
>> teen humor comics
What superheroes were introduced between 1939 and 1941 by Detective Comics and its sister company?
>> Archie Andrews
What comic book characters were created between 1939 and 1941?
>> Archie
Andrews
What well-known characters were created between 1939 and 1941?
>> Archie
Andrews
What well-known superheroes were introduced between 1939 and 1941 by Detective Comics?
>> Archie Andrews
```

**Congratulations!**

You have finished this series of ungraded labs. You were able to:

* Explore the Hugging Face Pipelines, which can be used right out of the bat.

* Fine-tune a model for the Extractive Question & Answering task.

I recommend you go through the free [Hugging Face course](https://huggingface.co/course/chapter1) to explore their ecosystem in more detail and find different ways to use the `transformers` library.
"""