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
with wandb.init(entity="wandb-japan", project="ichikara-test") as run:
    artifact = run.use_artifact('wandb-japan/ichikara-eval/ichikara-instruction-eval-001-001:v0', type='dataset')
    artifact_dir = artifact.download()
    questions = load_questions(artifact_dir + "/ichikara-instruction-eval-001-001.json", None, None)

# Create a dataset
dataset = Dataset(name='All_20240829', rows=questions)

# Publish the dataset
weave.publish(dataset)