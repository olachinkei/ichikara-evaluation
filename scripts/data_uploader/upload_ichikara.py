import json
import argparse
from typing import Optional
import wandb
import weave
from weave import Dataset

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Weave dataset uploader with wandb integration.")
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--question_file", type=str, required=True, help="Path to the JSON file containing the questions")
    
    return parser.parse_args()

def load_questions(question_file: str):
    """Load questions from a file."""
    with open(question_file, "r") as ques_file:
        try:
            data = json.load(ques_file)
            if isinstance(data, list):
                questions = data
            else:
                questions = [data]
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            raise

    return questions

def main():
    args = parse_arguments()
    weave.init(f"{args.entity}/{args.project}")
    questions = load_questions(args.question_file)
    dataset = Dataset(name='ichikara', rows=questions)
    weave.publish(dataset)
    
if __name__ == "__main__":
    main()