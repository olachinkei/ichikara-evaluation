import json
from typing import Optional, Any
import wandb
import weave
from weave import Dataset

# Initialize Weave
weave.init('wandb-japan/ichikara-test')

def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    with open(question_file, "r") as ques_file:
        try:
            data = json.load(ques_file)
            if isinstance(data, list):
                questions = data[begin:end] if begin is not None and end is not None else data
            else:
                questions = [data]
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            raise

    return questions

# Load questions
questions = load_questions("/workspace/filtered_ichikara_instruction_eval.json", None, None)

# Create a dataset
dataset = Dataset(name='ichikara_100', rows=questions)

# Publish the dataset
weave.publish(dataset)