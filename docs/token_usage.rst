.. _token_usage:

Token usage
===========

EDSL comes with a variety of features for monitoring token usage.
These include:

* Options for setting requests per minute (RPM) and tokens per minute (TPM) for models at your account.
* Methods for setting RPM and TPM from your workspace.
* Methods for turning off default prompt features to reduce token usage. 
* Methods for calculating next token probabilities.


Token limits 
------------

Token limits refer to the maximum number of tokens that a language model can process in a single input prompt or output generation.
A token limit affects how much text you can send to a model in one go. 
A language model provider should provide information about the token limits for each model that is associated with your account and API key.
When running a big job in EDSL, you may encounter token limits, which can be managed by adjusting the token limits for a model.


RPM: Requests Per Minute
^^^^^^^^^^^^^^^^^^^^^^^^

RPM stands for Requests Per Minute, which measures the number of API requests that a user can make to a language model within a minute. 
This is a metric for managing the load and traffic that a model can handle.


TPM: Tokens Per Minute
^^^^^^^^^^^^^^^^^^^^^^

TPM stands for Tokens Per Minute, which is a metric for tracking the volume of tokens processed by a language model within a minute. 
This metric typically tracks usage for billing purposes. 


Default token limits
--------------------

You can inspect and modify token limits for a model at the `Keys <https://www.expectedparrot.com/home/keys>`_ page of your account:

.. image:: static/home-keys-token-rates.png
   :alt: Key page view for setting model token rates
   :align: center
   :width: 100%


.. raw:: html

  <br>


Here we use methods for inspecting the default language model and its parameters from your workspace, including the token limits:

.. code-block:: python

  from edsl import Model

  model = Model() 
  model


This will show the following information:

.. list-table::
   :header-rows: 1

   * - key
     - value
   * - model
     - gpt-4o
   * - parameters:temperature
     - 0.5
   * - parameters:max_tokens
     - 1000
   * - parameters:top_p
     - 1
   * - parameters:frequency_penalty
     - 0
   * - parameters:presence_penalty
     - 0
   * - parameters:logprobs
     - False
   * - parameters:top_logprobs
     - 3



We can also inspect the RPM and TPM for the model:

.. code-block:: python

  [model.rpm, model.tpm]


This will show the following information:

.. code-block:: python

  [100, 2000000]


Modifying token limits
----------------------

We can reset the default RPM and TPM and then check the new values:

.. code-block:: python

  model = Model()

  model.rpm=10
  model.tpm=10

  [model.rpm, model.tpm]


This will show the following information:

.. code-block:: python

  [10, 10]


Here we change it again:

.. code-block:: python

  model = Model()

  model.rpm=100
  model.tpm=1000

  [model.rpm, model.tpm]


Output:

.. code-block:: python

  [100, 1000]


Please note that the token limits are subject to the constraints of the model and the API key associated with the model.
Let us know if you have any questions or need further assistance with token limits. 


Methods for reducing token usage 
--------------------------------

There are several ways to reduce the tokens required to run a question or survey.


Turning off question commments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each question type (other than `free_text`) automatically includes a `comment` field that gives the answering model a place to put additional information about its response to a question.
This serves as an outlet for a chatty model to return context about an answer without violating formatting instructions (e.g., a model may want to provide an explanation for a mutiple choice response but the answer to the question must only be one of the answer options).
Question comments can also be useful when used with survey "memory" rules, giving a model an opportunity to simulate a "chain of thought" across multiple survey questions.
(By default, questions are administered asynchronously; a model does not have context of other questions and answers in a survey unless memory rules are applied.)
Comments can also provide insight into non-responsive (`None`) answers: a model may use the comments field to describe a point of confusion about a question.

Because the question `comment` field requires additional tokens, it can sometimes be cost-effective to exclude the field from question prompts, especially when the comment is unlikely to be useful.
This is done by passing a boolean parameter `include_comment = False` when constructing a question. 
For example, here we compare a question with comments left on and turned off:

.. code-block:: python

  from edsl import QuestionNumerical, Survey, ScenarioList

  q1 = QuestionNumerical(
    question_name = "sum",
    question_text = "What is the sum of {{ number_1 }} and {{ number_2 }}?"
  )

  q2 = QuestionNumerical(
    question_name = "sum_silent",
    question_text = "What is the sum of {{ number_1 }} and {{ number_2 }}?",
    include_comment = False
  )

  survey = Survey([q1, q2])

  some_numbers = {
    "number_1": [0,1,2,3,4],
    "number_2": [5,4,3,2,1]
  }

  s = ScenarioList.from_nested_dict(some_numbers)

  results = survey.by(s).run()


We can check the responses and confirm that the `comment` field for the `sum_silent` question is `None`:

.. code-block:: python 

  results.select("number_1", "number_2", "sum", "sum_comment", "sum_silent", "sum_silent_comment")


Output:

.. list-table::
   :header-rows: 1

   * - scenario.number_1
     - scenario.number_2
     - answer.sum
     - comment.sum_comment
     - answer.sum_silent
     - comment.sum_silent_comment
   * - 0
     - 5
     - 5
     - The sum of 0 and 5 is 5.
     - 5
     - None
   * - 1
     - 4
     - 5
     - The sum of 1 and 4 is 5.
     - 5
     - None
   * - 2
     - 3
     - 5
     - The sum of 2 and 3 is 5.
     - 5
     - None
   * - 3
     - 2
     - 5
     - The sum of 3 and 2 is 5.
     - 5
     - None
   * - 4
     - 1
     - 5
     - The sum of 4 and 1 is 5.
     - 5
     - None


Coding question options 
^^^^^^^^^^^^^^^^^^^^^^^

Question instructions can be modified to prompt a model to use codes (integers) in lieu of text responses for answer options, reducing generated tokens.

This is done by passing a boolean parameter `use_code = True` to a `Question` when it is constructed. For example:

.. code-block:: python 

  from edsl import QuestionMultipleChoice

  q = QuestionMultipleChoice(
    question_name = "income_pref_coded", 
    question_text = "Which of the following is more important to you: ", 
    question_options = ["Financial stability", "Moving up the income ladder"], 
    use_code = True
  )


We can inspect the difference in the question prompt that is created by creating an identical question without the parameter and comparing the job prompts.
Here we also pass the parameter `include_comment = False`:

.. code-block:: python 

  from edsl import QuestionMultipleChoice, Survey, Agent, Model

  q1 = QuestionMultipleChoice(
    question_name = "income_pref", 
    question_text = "Which of the following is more important to you: ", 
    question_options = ["Financial stability", "Moving up the income ladder"]
  )

  q2 = QuestionMultipleChoice(
    question_name = "income_pref_coded", 
    question_text = "Which of the following is more important to you: ", 
    question_options = ["Financial stability", "Moving up the income ladder"], 
    use_code = True,
    include_comment = False
  )

  survey = Survey([q1, q2])

  # Construct a job with the survey and the default model
  job = survey.by(Model())

  # Inspect the question prompts
  job.prompts().select("question_index", "user_prompt")


Output:

.. list-table::
   :header-rows: 1

   * - question_index
     - user_prompt
   * - income_pref
     - Which of the following is more important to you:

       Financial stability

       Moving up the income ladder

       Only 1 option may be selected.

       Respond only with a string corresponding to one of the options.

       After the answer, you can put a comment explaining why you chose that option on the next line.
   * - income_pref_coded
     - Which of the following is more important to you:

       0: Financial stability

       1: Moving up the income ladder

       Only 1 option may be selected.

       Respond only with the code corresponding to one of the options.



The prompts can also be inspected after the survey is run:

.. code-block:: python

  results = survey.by(Model()).run()

  (
    results
    .select(
      "income_pref_user_prompt", "income_pref_generated_tokens",
      "income_pref_coded_user_prompt", "income_pref_coded_generated_tokens"
    )
  )


Output:

.. list-table::
   :header-rows: 1
   :widths: 50 50 50 50

   * - prompt.income_pref_user_prompt
     - generated_tokens.income_pref_generated_tokens
     - prompt.income_pref_coded_user_prompt
     - generated_tokens.income_pref_coded_generated_tokens
   * - Which of the following is more important to you:

       Financial stability

       Moving up the income ladder

       Only 1 option may be selected.

       Respond only with a string corresponding to one of the options.

       After the answer, you can put a comment explaining why you chose that option on the next line.
     - Financial stability provides a secure foundation and peace of mind, allowing for better long-term planning and resilience against unexpected challenges.
     - Which of the following is more important to you:

       0: Financial stability

       1: Moving up the income ladder

       Only 1 option may be selected.

       Respond only with the code corresponding to one of the options.
     - 0


No agent instructions
^^^^^^^^^^^^^^^^^^^^^

If no agents are used with the survey, the base agent instructions are not sent to the model, reducing overall tokens.
(This is a change from prior versions of EDSL.)


Calculating next token probabilities
------------------------------------

We can monitor tokens by calculating next token probabilities. 
This is done by setting model `logprobs = True` and then accessing the `raw_model_response` information in the results that are generated.
For example:

.. code-block:: python 

  from edsl import QuestionMultipleChoice, Agent, Model

  m = Model("gpt-4o", temperature = 1, logprobs = True)

  a = Agent(traits = {"persona":"financial advisor"})

  q = QuestionMultipleChoice(
    question_name = "income_pref_coded", 
    question_text = "Which of the following is more important to you: ", 
    question_options = ["Financial stability", "Moving up the income ladder"], 
    use_code = True,
    include_comment = False
  )

  results = q.by(a).by(m).run()

  example = results.select("raw_model_response.income_pref_coded_raw_model_response").to_list()[0]  

  example


Output:

.. code-block:: text 

  {'id': 'chatcmpl-AcgR7Wy6MMDqoorytBlzS2lzpwQAA',
  'choices': [{'finish_reason': 'stop',
  'index': 0,
  'logprobs': {'content': [{'token': '0',
      'bytes': [48],
      'logprob': -0.00019960667,
      'top_logprobs': [{'token': '0',
          'bytes': [48],
          'logprob': -0.00019960667},
      {'token': '1', 'bytes': [49], 'logprob': -9.000199},
      {'token': 'As', 'bytes': [65, 115], 'logprob': -10.875199}]}],
      'refusal': None},
  'message': {'content': '0',
      'refusal': None,
      'role': 'assistant',
      'audio': None,
      'function_call': None,
      'tool_calls': None}}],
  'created': 1733782953,
  'model': 'gpt-4o-2024-08-06',
  'object': 'chat.completion',
  'service_tier': None,
  'system_fingerprint': 'fp_9d50cd990b',
  'usage': {'completion_tokens': 1,
  'prompt_tokens': 82,
  'total_tokens': 83,
  'completion_tokens_details': {'accepted_prediction_tokens': 0,
  'audio_tokens': 0,
  'reasoning_tokens': 0,
  'rejected_prediction_tokens': 0},
  'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}}


We can use the information to calculate next token probabilities:

.. code-block:: python 
        
  next_token_probs = example['choices'][0]['logprobs']['content'][0]['top_logprobs']
  next_token_probs


Output:

.. code-block:: text 

  [{'token': '0', 'bytes': [48], 'logprob': -0.00019960667},
  {'token': '1', 'bytes': [49], 'logprob': -9.000199},
  {'token': 'As', 'bytes': [65, 115], 'logprob': -10.875199}]


Translating the information:

.. code-block:: python 

  import math

  # Specifying the codes for the answer options and non-responses:
  options = {'0': "Financial stability", '1':"Moving up the income ladder"}

  for token_info in next_token_probs:
    option = options.get(token_info['token'], None)
    if option:
      p = math.exp(token_info['logprob'])
  
      print(f"Probability of selecting '{option}' was {p:.3f}")


Output:

.. code-block:: text 

  Probability of selecting 'Financial stability' was 0.992
  Probability of selecting 'Moving up the income ladder' was 0.008