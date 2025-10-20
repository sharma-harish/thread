# W&B Weave
Weights & Biases (W&B) Weave is a framework for tracking, experimenting with, evaluating, deploying, and improving LLM-based applications. Designed for flexibility and scalability, Weave supports every stage of your LLM application development workflow:

## Features

- **Tracing & Monitoring:** Track LLM calls and application logic to debug and analyze production systems.  
- **Systematic Iteration:** Refine and iterate on prompts, datasets, and models.  
- **Experimentation:** Experiment with different models and prompts in the LLM Playground.  
- **Evaluation:** Use custom or pre-built scorers alongside our comparison tools to systematically assess and enhance application performance.  
- **Guardrails:** Protect your application with pre- and post-safeguards for content moderation, prompt safety, and more.


## Learn Weave with W&B Inference

This guide shows you how to use W&B Weave with W&B Inference. Using W&B Inference, you can build and trace LLM applications using live open-source models without setting up your own infrastructure or managing API keys from multiple providers. Just obtain your W&B API key and use it to interact with all models hosted by W&B Inference.

## What you'll learn

In this guide, you'll:

- Set up Weave and W&B Inference
- Build a basic LLM application with automatic tracing
- Compare multiple models
- Evaluate model performance on a dataset
- View your results in the Weave UI

## Prerequisites

Before you begin, you need a W&B account and an API key from [https://wandb.ai/authorize](https://wandb.ai/authorize).

Then, in a Python environment running version 3.9 or later, install the required libraries:

```bash
pip install weave openai
```
The `openai` library is installed because you use the standard OpenAI client to interact with W&B Inference, regardless of which hosted model you're actually calling. This allows you to swap between supported models by only changing the slug, and make use of any existing code you have that was written to use the OpenAI API.

---

## Step 1: Trace your first LLM call

Start with a basic example that uses **Llama 3.1-8B** through W&B Inference.

When you run this code, Weave:

- Traces your LLM call automatically  
- Logs inputs, outputs, latency, and token usage  
- Provides a link to view your trace in the Weave UI  

```python
import weave
import openai

## Initialize Weave - replace with your-team/your-project
weave.init("<team-name>/my-first-weave-project")

## Create an OpenAI-compatible client pointing to W&B Inference
client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key="YOUR_WANDB_API_KEY",  # Replace with your actual API key
    project="<team-name>/my-first-weave-project",  # Required for usage tracking
)

## Decorate your function to enable tracing; use the standard OpenAI client
@weave.op()
def ask_llama(question: str) -> str:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
    )
    return response.choices[0].message.content

## Call your function - Weave automatically traces everything
result = ask_llama("What are the benefits of using W&B Weave for LLM development?")
print(result)
```

## Step 2: Build a text summarization application

Next, try running this code, which is a basic summarization app that shows how Weave traces nested operations:

```python
import weave
import openai

## Initialize Weave - replace with your-team/your-project
weave.init("<team-name>/my-first-weave-project")

client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key="YOUR_WANDB_API_KEY",  # Replace with your actual API key
    project="<team-name>/my-first-weave-project",  # Required for usage tracking
)

@weave.op()
def extract_key_points(text: str) -> list[str]:
    """Extract key points from a text."""
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "Extract 3-5 key points from the text. Return each point on a new line."},
            {"role": "user", "content": text}
        ],
    )
    ## Returns response without blank lines
    return [line for line in response.choices[0].message.content.strip().splitlines() if line.strip()]

@weave.op()
def create_summary(key_points: list[str]) -> str:
    """Create a concise summary based on key points."""
    points_text = "\n".join(f"- {point}" for point in key_points)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "Create a one-sentence summary based on these key points."},
            {"role": "user", "content": f"Key points:\n{points_text}"}
        ],
    )
    return response.choices[0].message.content

@weave.op()
def summarize_text(text: str) -> dict:
    """Main summarization pipeline."""
    key_points = extract_key_points(text)
    summary = create_summary(key_points)
    return {
        "key_points": key_points,
        "summary": summary
    }

## Try it with sample text
sample_text = """
The Apollo 11 mission was a historic spaceflight that landed the first humans on the Moon 
on July 20, 1969. Commander Neil Armstrong and lunar module pilot Buzz Aldrin descended 
to the lunar surface while Michael Collins remained in orbit. Armstrong became the first 
person to step onto the Moon, followed by Aldrin 19 minutes later. They spent about 
two and a quarter hours together outside the spacecraft, collecting samples and taking photographs.
"""

result = summarize_text(sample_text)
print("Key Points:", result["key_points"])
print("\nSummary:", result["summary"])
```

## Step 3: Compare multiple models

W&B Inference provides access to multiple models. Use the following code to compare the performance of Llama and DeepSeek's respective responses:
```python
import weave
import openai

## Initialize Weave - replace with your-team/your-project
weave.init("<team-name>/my-first-weave-project")

client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key="YOUR_WANDB_API_KEY",  # Replace with your actual API key
    project="<team-name>/my-first-weave-project",  # Required for usage tracking
)

## Define a Model class to compare different LLMs
class InferenceModel(weave.Model):
    model_name: str
    
    @weave.op()
    def predict(self, question: str) -> str:
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": question}
            ],
        )
        return response.choices[0].message.content

## Create instances for different models
llama_model = InferenceModel(model_name="meta-llama/Llama-3.1-8B-Instruct")
deepseek_model = InferenceModel(model_name="deepseek-ai/DeepSeek-V3-0324")

## Compare their responses
test_question = "Explain quantum computing in one paragraph for a high school student."

print("Llama 3.1 8B response:")
print(llama_model.predict(test_question))
print("\n" + "="*50 + "\n")
print("DeepSeek V3 response:")
print(deepseek_model.predict(test_question))
```

## Step 4: Evaluate model performance

Evaluate how well a model performs on a Q&A task using Weave's built-in EvaluationLogger. This provides structured evaluation tracking with automatic aggregation, token usage capture, and rich comparison features in the UI.

Append the following code to the script you used in step 3:
```python
from typing import Optional
from weave import EvaluationLogger

## Create a simple dataset
dataset = [
    {"question": "What is 2 + 2?", "expected": "4"},
    {"question": "What is the capital of France?", "expected": "Paris"},
    {"question": "Name a primary color", "expected_one_of": ["red", "blue", "yellow"]},
]

## Define a scorer
@weave.op()
def accuracy_scorer(expected: str, output: str, expected_one_of: Optional[list[str]] = None) -> dict:
    """Score the accuracy of the model output."""
    output_clean = output.strip().lower()
    
    if expected_one_of:
        is_correct = any(option.lower() in output_clean for option in expected_one_of)
    else:
        is_correct = expected.lower() in output_clean
    
    return {"correct": is_correct, "score": 1.0 if is_correct else 0.0}

## Evaluate a model using Weave's EvaluationLogger
def evaluate_model(model: InferenceModel, dataset: list[dict]):
    """Run evaluation on a dataset using Weave's built-in evaluation framework."""
    ## Initialize EvaluationLogger BEFORE calling the model to capture token usage
    ## This is especially important for W&B Inference to track costs
    ## Convert model name to a valid format (replace non-alphanumeric chars with underscores)
    safe_model_name = model.model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    eval_logger = EvaluationLogger(
        model=safe_model_name,
        dataset="qa_dataset"
    )
    
    for example in dataset:
        ## Get model prediction
        output = model.predict(example["question"])
        
        ## Log the prediction
        pred_logger = eval_logger.log_prediction(
            inputs={"question": example["question"]},
            output=output
        )
        
        ## Score the output
        score = accuracy_scorer(
            expected=example.get("expected", ""),
            output=output,
            expected_one_of=example.get("expected_one_of")
        )
        
        ## Log the score
        pred_logger.log_score(
            scorer="accuracy",
            score=score["score"]
        )
        
        ## Finish logging for this prediction
        pred_logger.finish()
    
    ## Log summary - Weave automatically aggregates the accuracy scores
    eval_logger.log_summary()
    print(f"Evaluation complete for {model.model_name} (logged as: {safe_model_name}). View results in the Weave UI.")

## Compare multiple models - a key feature of Weave's evaluation framework
models_to_compare = [
    llama_model,
    deepseek_model,
]

for model in models_to_compare:
    evaluate_model(model, dataset)

## In the Weave UI, navigate to the Evals tab to compare results across models
```

Running these examples returns links to the traces in the terminal. Click any link to view traces in the Weave UI.

In the Weave UI, you can:

Review a timeline of all your LLM calls
Inspect inputs and outputs for each operation
View token usage and estimated costs (automatically captured by EvaluationLogger)
Analyze latency and performance metrics
Navigate to the Evals tab to see aggregated evaluation results
Use the Compare feature to analyze performance across different models
Page through specific examples to see how different models performed on the same inputs

# Tutorial: Build an Evaluation pipeline

To iterate on an application, we need a way to evaluate if it's improving. To do so, a common practice is to test it against the same set of examples when there is a change. W&B Weave has a first-class way to track evaluations with Model & Evaluation classes. We have built the APIs to make minimal assumptions to allow for the flexibility to support a wide array of use-cases.

## 1. Build a Model

Models store and version information about your system, such as prompts, temperatures, and more. Weave automatically captures when they are used and updates the version when there are changes.

Models are declared by subclassing Model and implementing a predict function definition, which takes one example and returns the response.

```python
import json
import openai
import weave

class ExtractFruitsModel(weave.Model):
    model_name: str
    prompt_template: str

    @weave.op()
    async def predict(self, sentence: str) -> dict:
        client = openai.AsyncClient()

        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": self.prompt_template.format(sentence=sentence)}
            ],
        )
        result = response.choices[0].message.content
        if result is None:
            raise ValueError("No response from model")
        parsed = json.loads(result)
        return parsed
```

You can instantiate Model objects as normal like this:
```python
import asyncio
import weave

weave.init('intro-example')

model = ExtractFruitsModel(model_name='gpt-3.5-turbo-1106',
                        prompt_template='Extract fields ("fruit": <str>, "color": <str>, "flavor": <str>) from the following text, as json: {sentence}')
sentence = "There are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles that grow there, which are purple and taste like candy."
print(asyncio.run(model.predict(sentence)))
## if you're in a Jupyter Notebook, run:
## await model.predict(sentence)
```

## 2. Collect some examples

```python
sentences = [
    "There are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles that grow there, which are purple and taste like candy.",
    "Pounits are a bright green color and are more savory than sweet.",
    "Finally, there are fruits called glowls, which have a very sour and bitter taste which is acidic and caustic, and a pale orange tinge to them."
]
labels = [
    {'fruit': 'neoskizzles', 'color': 'purple', 'flavor': 'candy'},
    {'fruit': 'pounits', 'color': 'bright green', 'flavor': 'savory'},
    {'fruit': 'glowls', 'color': 'pale orange', 'flavor': 'sour and bitter'}
]
examples = [
    {'id': '0', 'sentence': sentences[0], 'target': labels[0]},
    {'id': '1', 'sentence': sentences[1], 'target': labels[1]},
    {'id': '2', 'sentence': sentences[2], 'target': labels[2]}
]
```

## 3. Evaluate a Model
Evaluations assess a Models performance on a set of examples using a list of specified scoring functions or weave.scorer.Scorer classes.

Here, we'll use a default scoring class MultiTaskBinaryClassificationF1 and we'll also define our own fruit_name_score scoring function.

Here sentence is passed to the model's predict function, and target is used in the scoring function, these are inferred based on the argument names of the predict and scoring functions. The fruit key needs to be outputted by the model's predict function and must also be existing as a column in the dataset (or outputted by the preprocess_model_input function if defined).

```python
import weave
from weave.scorers import MultiTaskBinaryClassificationF1

weave.init('intro-example')

@weave.op()
def fruit_name_score(target: dict, output: dict) -> dict:
    return {'correct': target['fruit'] == output['fruit']}

evaluation = weave.Evaluation(
    dataset=examples,
    scorers=[
        MultiTaskBinaryClassificationF1(class_names=["fruit", "color", "flavor"]),
        fruit_name_score
    ],
)
print(asyncio.run(evaluation.evaluate(model)))
## if you're in a Jupyter Notebook, run:
## await evaluation.evaluate(model)
```

In some applications we want to create custom Scorer classes - where for example a standardized LLMJudge class should be created with specific parameters (e.g. chat model, prompt), specific scoring of each row, and specific calculation of an aggregate score. See the tutorial on defining a Scorer class in the next chapter on Model-Based Evaluation of RAG applications for more information.

## 4. Pulling it all together
```python
import json
import asyncio
import weave
from weave.scorers import MultiTaskBinaryClassificationF1
import openai

## We create a model class with one predict function.
## All inputs, predictions and parameters are automatically captured for easy inspection.

class ExtractFruitsModel(weave.Model):
    model_name: str
    prompt_template: str

    @weave.op()
    async def predict(self, sentence: str) -> dict:
        client = openai.AsyncClient()

        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": self.prompt_template.format(sentence=sentence)}
            ],
            response_format={ "type": "json_object" }
        )
        result = response.choices[0].message.content
        if result is None:
            raise ValueError("No response from model")
        parsed = json.loads(result)
        return parsed

## We call init to begin capturing data in the project, intro-example.
weave.init('intro-example')

## We create our model with our system prompt.
model = ExtractFruitsModel(name='gpt4',
                        model_name='gpt-4-0125-preview',
                        prompt_template='Extract fields ("fruit": <str>, "color": <str>, "flavor") from the following text, as json: {sentence}')
sentences = ["There are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles that grow there, which are purple and taste like candy.",
"Pounits are a bright green color and are more savory than sweet.",
"Finally, there are fruits called glowls, which have a very sour and bitter taste which is acidic and caustic, and a pale orange tinge to them."]
labels = [
    {'fruit': 'neoskizzles', 'color': 'purple', 'flavor': 'candy'},
    {'fruit': 'pounits', 'color': 'bright green', 'flavor': 'savory'},
    {'fruit': 'glowls', 'color': 'pale orange', 'flavor': 'sour and bitter'}
]
examples = [
    {'id': '0', 'sentence': sentences[0], 'target': labels[0]},
    {'id': '1', 'sentence': sentences[1], 'target': labels[1]},
    {'id': '2', 'sentence': sentences[2], 'target': labels[2]}
]
## If you have already published the Dataset, you can run:
## dataset = weave.ref('example_labels').get()

## We define a scoring function to compare our model predictions with a ground truth label.
@weave.op()
def fruit_name_score(target: dict, output: dict) -> dict:
    return {'correct': target['fruit'] == output['fruit']}

## Finally, we run an evaluation of this model.
## This will generate a prediction for each input example, and then score it with each scoring function.
evaluation = weave.Evaluation(
    name='fruit_eval',
    dataset=examples, scorers=[MultiTaskBinaryClassificationF1(class_names=["fruit", "color", "flavor"]), fruit_name_score],
)
print(asyncio.run(evaluation.evaluate(model)))
## if you're in a Jupyter Notebook, run:
## await evaluation.evaluate(model)
```

# Ops
A W&B Weave op is a versioned function that automatically logs all calls.

Python
To create an op, decorate a python function with weave.op()

```python
import weave

@weave.op()
def track_me(v):
    return v + 5

weave.init('intro-example')
track_me(15)
```

Calling an op will create a new op version if the code has changed from the last call, and log the inputs and outputs of the function.

NOTE
Functions decorated with @weave.op() will behave normally (without code versioning and tracking), if you don't call weave.init('your-project-name') before calling them.

## Customize display names
You can customize the op's display name by setting the name parameter in the @weave.op decorator:

```python
@weave.op(name="custom_name")
def func():
    ...
```

## Customize logged inputs and outputs
If you want to change the data that is logged to weave without modifying the original function (e.g. to hide sensitive data), you can pass postprocess_inputs and postprocess_output to the op decorator.

```python
postprocess_inputs takes in a dict where the keys are the argument names and the values are the argument values, and returns a dict with the transformed inputs.

postprocess_output takes in any value which would normally be returned by the function and returns the transformed output.

from dataclasses import dataclass
from typing import Any
import weave

@dataclass
class CustomObject:
    x: int
    secret_password: str

def postprocess_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {k:v for k,v in inputs.items() if k != "hide_me"}

def postprocess_output(output: CustomObject) -> CustomObject:
    return CustomObject(x=output.x, secret_password="REDACTED")

@weave.op(
    postprocess_inputs=postprocess_inputs,
    postprocess_output=postprocess_output,
)
def func(a: int, hide_me: str) -> CustomObject:
    return CustomObject(x=a, secret_password=hide_me)

weave.init('hide-data-example') # 🐝
func(a=1, hide_me="password123")
```

## Control sampling rate

ou can control how frequently an op's calls are traced by setting the tracing_sample_rate parameter in the @weave.op decorator. This is useful for high-frequency ops where you only need to trace a subset of calls.

Note that sampling rates are only applied to root calls. If an op has a sample rate, but is called by another op first, then that sampling rate will be ignored.

```python
@weave.op(tracing_sample_rate=0.1)  # Only trace ~10% of calls
def high_frequency_op(x: int) -> int:
    return x + 1

@weave.op(tracing_sample_rate=1.0)  # Always trace (default)
def always_traced_op(x: int) -> int:
    return x + 1
```

When an op's call is not sampled:

The function executes normally
No trace data is sent to Weave
Child ops are also not traced for that call
The sampling rate must be between 0.0 and 1.0 inclusive.

## Control call link output

If you want to suppress the printing of call links during logging, you can set the WEAVE_PRINT_CALL_LINK environment variable to false. This can be useful if you want to reduce output verbosity and reduce clutter in your logs.
```bash
export WEAVE_PRINT_CALL_LINK=false
```

## Deleting an op
To delete a version of an op, call .delete() on the op ref.
```python
weave.init('intro-example')
my_op_ref = weave.ref('track_me:v1')
my_op_ref.delete()
```
Trying to access a deleted op will result in an error.

# Datasets
W&B Weave Datasets help you to organize, collect, track, and version examples for LLM application evaluation for easy comparison. You can create and interact with Datasets programmatically and via the UI.

This page describes:

Basic Dataset operations in Python and TypeScript and how to get started
How to create a Dataset in Python and TypeScript from objects such as Weave calls
Available operations on a Dataset in the UI

## Dataset quickstart
The following code samples demonstrate how to perform fundamental Dataset operations using Python and TypeScript. Using the SDKs, you can:

Create a Dataset
Publish the Dataset
Retrieve the Dataset
Access a specific example in the Dataset
Select a tab to see Python and TypeScript-specific code.

```python
import weave
from weave import Dataset
# Initialize Weave
weave.init('intro-example')

# Create a dataset
dataset = Dataset(
    name='grammar',
    rows=[
        {'id': '0', 'sentence': "He no likes ice cream.", 'correction': "He doesn't like ice cream."},
        {'id': '1', 'sentence': "She goed to the store.", 'correction': "She went to the store."},
        {'id': '2', 'sentence': "They plays video games all day.", 'correction': "They play video games all day."}
    ]
)

# Publish the dataset
weave.publish(dataset)

# Retrieve the dataset
dataset_ref = weave.ref('grammar').get()

# Access a specific example
example_label = dataset_ref.rows[2]['sentence']
```

## Create a Dataset from other objects
n Python, Datasets can also be constructed from common Weave objects like calls, and Python objects like pandas.DataFrames. This feature is useful if you want to create an example Dataset from specific examples.

Weave call

To create a Dataset from one or more Weave calls, retrieve the call object(s), and add them to a list in the from_calls method.
```python
@weave.op
def model(task: str) -> str:
    return f"Now working on {task}"

res1, call1 = model.call(task="fetch")
res2, call2 = model.call(task="parse")

dataset = Dataset.from_calls([call1, call2])
# Now you can use the dataset to evaluate the model, etc.
```

## Pandas DataFrame
To create a Dataset from a Pandas DataFrame object, use the from_pandas method.

To convert the Dataset back, use to_pandas.

```python
import pandas as pd

df = pd.DataFrame([
    {'id': '0', 'sentence': "He no likes ice cream.", 'correction': "He doesn't like ice cream."},
    {'id': '1', 'sentence': "She goed to the store.", 'correction': "She went to the store."},
    {'id': '2', 'sentence': "They plays video games all day.", 'correction': "They play video games all day."}
])
dataset = Dataset.from_pandas(df)
df2 = dataset.to_pandas()

assert df.equals(df2)
```

## Hugging Face Datasets
To create a Dataset from a Hugging Face datasets.Dataset or datasets.DatasetDict object, first ensure you have the necessary dependencies installed:

```bash
pip install weave[huggingface]
```

Then, use the from_hf method. If you provide a DatasetDict with multiple splits (like 'train', 'test', 'validation'), Weave will automatically use the 'train' split and issue a warning. If the 'train' split is not present, it will raise an error. You can provide a specific split directly (e.g., hf_dataset_dict['test']).

To convert a weave.Dataset back to a Hugging Face Dataset, use the to_hf method.

```python
# Ensure datasets is installed: pip install datasets
from datasets import Dataset as HFDataset, DatasetDict

# Example with HF Dataset
hf_rows = [
    {'id': '0', 'sentence': "He no likes ice cream.", 'correction': "He doesn't like ice cream."},
    {'id': '1', 'sentence': "She goed to the store.", 'correction': "She went to the store."},
]
hf_ds = HFDataset.from_list(hf_rows)
weave_ds_from_hf = Dataset.from_hf(hf_ds)

# Convert back to HF Dataset
converted_hf_ds = weave_ds_from_hf.to_hf()

# Example with HF DatasetDict (uses 'train' split by default)
hf_dict = DatasetDict({
    'train': HFDataset.from_list(hf_rows),
    'test': HFDataset.from_list([{'id': '2', 'sentence': "Test sentence", 'correction': "Test correction"}])
})
# This will issue a warning and use the 'train' split
weave_ds_from_dict = Dataset.from_hf(hf_dict)

# Providing a specific split
weave_ds_from_test_split = Dataset.from_hf(hf_dict['test'])
```

## Create, edit, and delete a Dataset in the UI

You can create, edit, and delete Datasets in the UI.

### Create a new Dataset

Navigate to the Weave project you want to edit.

1. In the sidebar, select Traces.

2. Select one or more calls that you want to create a new Dataset for.

3. In the upper right-hand menu, click the Add selected rows to a dataset icon (located next to the trashcan icon).

4. From the Choose a dataset dropdown, select Create new. The Dataset name field appears.

5. In the Dataset name field, enter a name for your dataset. Options to Configure dataset fields appear.
6.(Optional) In Configure dataset fields, select the fields from your calls to include in the dataset.

7. You can customize the column names for each selected field.
You can select a subset of fields to include in the new Dataset, or deselect all fields.
Once you've configured the dataset fields, click Next. A preview of your new Dataset appears.

8. (Optional) Click any of the editable fields in your Dataset to edit the entry.

9. Click Create dataset. Your new dataset is created.

10. In the confirmation popup, click View the dataset to view the new Dataset. Alternatively, go to the Datasets tab.

## Delete a Dataset

1. Navigate to the Weave project containing the Dataset you want to edit.

2. From the sidebar, select Datasets. Your available Datasets display.

3. In the Object column, click the name and version of the Dataset you want to delete. A pop-out modal showing Dataset information like name, version, author, and Dataset rows displays.

4. In the upper right-hand corner of the modal, click the trashcan icon.

5. A pop-up modal prompting you to confirm Dataset deletion displays.
6. In the pop-up modal, click the red Delete button to delete the Dataset. Alternatively, click Cancel if you don't want to delete the Dataset.

Now, the Dataset is deleted, and no longer visible in the Datasets tab in your Weave dashboard.

## Add a new example to a Dataset

1. Navigate to the Weave project you want to edit.

2. In the sidebar, select Traces.

3. Select one or more calls with Datasets for which you want to create new examples.

4. In the upper right-hand menu, click the Add selected rows to a dataset icon (located next to the trashcan icon). Optionally, toggle Show latest versions to off to display all versions of all available datasets.

5. From the Choose a dataset dropdown, select the Dataset you want to add examples to. Options to Configure field mapping will display.
6. (Optional) In Configure field mapping, you can adjust the mapping of fields from your calls to the corresponding dataset columns.

7. Once you've configured field mappings, click Next. A preview of your new Dataset appears.

8. In the empty row (green), add your new example value(s). Note that the id field is not editable and is created automatically by Weave.

9. Click Add to dataset. Alternatively, to return to the Configure field mapping screen, click Back.

10. In the confirmation popup, click View the dataset to see the changes. Alternatively, navigate to the Datasets tab to view the updates to your Dataset.

