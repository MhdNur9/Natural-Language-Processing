# -*- coding: utf-8 -*-
"""C4_W3_1_Question_Answering_with_BERT_and_HuggingFace_Pytorch_tydiqa.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O4LvdhHw6Zx7Kd43HK-p5a1rtsHUEia5

# Question Answering with BERT and HuggingFace

You've seen how to use BERT, and other transformer models for a wide range of natural language tasks, including machine translation, summarization, and question answering. Transformers have become the standard model for NLP, similar to convolutional models in computer vision. And all started with Attention!

In practice, you'll rarely train a transformer model from scratch.  Transformers tend to be very large, so they take time, money, and lots of data to train fully. Instead, you'll want to start with a pre-trained model and fine-tune it with your dataset if you need to.

[Hugging Face](https://huggingface.co/) (🤗) is the best resource for pre-trained transformers. Their open-source libraries simplify downloading and using transformer models like BERT, T5, and GPT-2. And the best part, you can use them alongside either TensorFlow, PyTorch and Flax.

In this notebook, you'll use 🤗  transformers to download and use the DistilBERT model for question answering.

First, let's install some packages that we will use during the lab.
"""

!pip install transformers

"""## Pipelines

Before fine-tuning a model, you will look to the pipelines from Hugging Face
to use pre-trained transformer models for specific tasks. The `transformers` library provides pipelines for popular tasks like sentiment analysis, summarization, and text generation. A pipeline consists of a tokenizer, a model, and the model configuration. All these are packaged together into an easy-to-use object. Hugging Face makes life easier.

Pipelines are intended to be used without fine-tuning and will often be immediately helpful in your projects. For example, `transformers` provides a pipeline for [question answering](https://huggingface.co/transformers/main_classes/pipelines.html#the-pipeline-abstraction) that you can directly use to answer your questions if you give some context. Let's see how to do just that.

You will import `pipeline` from `transformers` for creating pipelines.
"""

from transformers import pipeline

"""Now, you will create the pipeline for question-answering, which uses the [DistilBert](https://hf.co/distilbert-base-cased-distilled-squad) model for extractive question answering (i.e., answering questions with the exact wording provided in the context)."""

# The task "question-answering" will return a QuestionAnsweringPipeline object
question_answerer = pipeline(task="question-answering", model="distilbert-base-cased-distilled-squad")

"""After running the last cell, you have a pipeline for performing question answering given a context string. The pipeline `question_answerer` you just created needs you to pass the question and context as strings. It returns an answer to the question from the context you provided. For example, here are the first few paragraphs from the [Wikipedia entry for tea](https://en.wikipedia.org/wiki/Tea) that you will use as the context.



"""

context = """
Tea is an aromatic beverage prepared by pouring hot or boiling water over cured or fresh leaves of Camellia sinensis,
an evergreen shrub native to China and East Asia. After water, it is the most widely consumed drink in the world.
There are many different types of tea; some, like Chinese greens and Darjeeling, have a cooling, slightly bitter,
and astringent flavour, while others have vastly different profiles that include sweet, nutty, floral, or grassy
notes. Tea has a stimulating effect in humans primarily due to its caffeine content.

The tea plant originated in the region encompassing today's Southwest China, Tibet, north Myanmar and Northeast India,
where it was used as a medicinal drink by various ethnic groups. An early credible record of tea drinking dates to
the 3rd century AD, in a medical text written by Hua Tuo. It was popularised as a recreational drink during the
Chinese Tang dynasty, and tea drinking spread to other East Asian countries. Portuguese priests and merchants
introduced it to Europe during the 16th century. During the 17th century, drinking tea became fashionable among the
English, who started to plant tea on a large scale in India.

The term herbal tea refers to drinks not made from Camellia sinensis: infusions of fruit, leaves, or other plant
parts, such as steeps of rosehip, chamomile, or rooibos. These may be called tisanes or herbal infusions to prevent
confusion with 'tea' made from the tea plant.
"""

"""Now, you can ask your model anything related to that passage. For instance, "Where is tea native to?"."""

result = question_answerer(question="Where is tea native to?", context=context)
print(result['answer'])

"""You can also pass multiple questions to your pipeline within a list so that you can ask:

*   "Where is tea native to?"
*   "When was tea discovered?"
*   "What is the species name for tea?"

at the same time, and your `question-answerer` will return all the answers.
"""

questions = ["Where is tea native to?",
             "When was tea discovered?",
             "What is the species name for tea?"]

results = question_answerer(question=questions, context=context)

for q, r in zip(questions, results):
    print(q, "\n>> " + r['answer'])

"""Although the models used in the Hugging Face pipelines generally give outstanding results, sometimes you will have particular examples where they don't perform so well. Let's use the following example with a context string about the Golden Age of Comic Books:"""

context = """
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

"""Let's ask the following question: "What popular superheroes were introduced between 1939 and 1941?" The answer is in the fourth paragraph of the context string."""

question = "What popular superheroes were introduced between 1939 and 1941?"

result = question_answerer(question=question, context=context)
print(result['answer'])

"""Here, the answer should be:
"Batman and Robin, Wonder Woman, the Flash,
Green Lantern, Doctor Fate, the Atom, Hawkman, Green Arrow, and Aquaman", instead, the pipeline returned a different answer.  You can even try different question wordings:

*   "What superheroes were introduced between 1939 and 1941?"
*   "What comic book characters were created between 1939 and 1941?"
*   "What well-known characters were created between 1939 and 1941?"
*   "What well-known superheroes were introduced between 1939 and 1941 by Detective Comics?"

and you will only get incorrect answers.
"""

questions = ["What popular superheroes were introduced between 1939 and 1941?",
             "What superheroes were introduced between 1939 and 1941 by Detective Comics and its sister company?",
             "What comic book characters were created between 1939 and 1941?",
             "What well-known characters were created between 1939 and 1941?",
             "What well-known superheroes were introduced between 1939 and 1941 by Detective Comics?"]

results = question_answerer(question=questions, context=context)

for q, r in zip(questions, results):
    print(q, "\n>> " + r['answer'])

"""It seems like this model is a **huge fan** of Archie Andrews. It even considers him a superhero!

The example that fooled your `question_answerer` belongs to the [TyDi QA dataset](https://ai.google.com/research/tydiqa), a dataset from Google for question/answering in diverse languages. To achieve better results when you know that the pipeline isn't working as it should, you need to consider fine-tuning your model.

In the next ungraded lab, you will get the chance to fine-tune the DistilBert model using the TyDi QA dataset.


"""