# Nejumi Leaderboard 3 for ichikara human evaluation
## Overview

This repository provides an evaluation framework for large language models (LLMs) using the Ichikara dataset. It allows for assessment of LLM outputs using both automated scoring by another LLM ("LLM as a Judge") and human evaluation. This repository serves as a comprehensive tool for comparing the quality of LLM responses with a hybrid evaluation approach.

For more detailed analysis and findings based on this code, please refer to the technical report available [here](https://wandb.ai/wandb-japan/ichikara-test/reports/Technical-Report-LLM-vs---Vmlldzo5NTQ2NjM1).

## Usage

1. Run `run_eval.py` with the specified configuration for the LLM you want to evaluate. This will generate outputs for each Ichikara problem and evaluate them using "LLM as a Judge." The results are logged to W&B's Weave, allowing you to review inputs, outputs, and prompt information for each problem.

    ```bash
    python scripts/run_eval.py -c /path/to/LLM/config/file
    ```

2. Download the evaluation results as a CSV from the Weave's Traces. Then, preprocess the CSV for human evaluation using `ichikara_preprocessed_for_human_eval.py`. The resulting CSV will include empty columns for 'Relevance', 'Accuracy', 'Fluency', 'Information Coverage', 'Overall Rating', and 'Reason'.

    ```bash
    python scripts/evaluator/ichikara_preprocessed_for_human_eval.py --auto_evaled_csv /path/to/auto/evaled/file
    ```

3. Fill in the human evaluation scores in the CSV file created in step 2. Complete the columns for 'Relevance', 'Accuracy', 'Fluency', 'Information Coverage', 'Overall Rating', and 'Reason'.

4. Upload the human evaluation results to Weave. Specify the path to the human-evaluated CSV file in `ichikara.upload_human_evaled_file` in `configs/base_config.yaml`. Then, run `run_eval.py` again.

    ```bash
    python scripts/run_eval.py -c /path/to/LLM/config/file
    ```

5. The results of both automated and human evaluations are logged to Weave. You can further analyze the results using the Compare feature in the Evaluations section.


### Environment Setup
1. Set up environment variables
```bash
export WANDB_API_KEY=<your WANDB_API_KEY>
export OPENAI_API_KEY=<your OPENAI_API_KEY>
# if needed, set the following API KEY too
export ANTHROPIC_API_KEY=<your ANTHROPIC_API_KEY>
export GOOGLE_API_KEY=<your GOOGLE_API_KEY>
export COHERE_API_KEY=<your COHERE_API_KEY>
export MISTRAL_API_KEY=<your MISTRAL_API_KEY>
export AWS_ACCESS_KEY_ID=<your AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<your AWS_SECRET_ACCESS_KEY>
export AWS_DEFAULT_REGION=<your AWS_DEFAULT_REGION>
export UPSTAGE_API_KEY=<your UPSTAGE_API_KEY>
# if needed, please login in huggingface
huggingface-cli login
```

2. Clone the repository

3. Set up a Python environment with `requirements.txt`

### Dataset Preparation

To upload the ichikara dataset to Weave, use the provided script. Run the script below to upload the dataset to W&B.

#### Command Line Argument Descriptions

- `--entity`: Specify the WandB entity name. This corresponds to the project owner in W&B.
- `--project`: Specify the W&B project name where the dataset will be saved.
- `--question_file`: Path to the ichikara dataset as JSON format.

#### Example

```bash
python scripts/data_uploader/upload_ichikara.py --entity <W&B Entity> --project <W&B Project> --question_file /path/to/ichikara/dataset
```

Running the above command will upload the specified question file to the designated W&B project. You can review the uploaded data via W&B’s Weave.

### Configuration

#### Base configuration

The `base_config.yaml` file contains basic settings, and you can create a separate YAML file for model-specific settings. This allows for easy customization of settings for each model while maintaining a consistent base configuration.

Below, you will find a detailed description of the variables utilized in the `base_config.yaml` file.

- **wandb:** Information used for Weights & Biases (W&B) support.
    - `entity`: Name of the W&B Entity.
    - `project`: Name of the W&B Project.
- **inference_interval:** Set inference interval in seconds. This is particularly effective when there are rate limits, such as with APIs.
- **model:** Information about the model.
    - `artifacts_path`: Path of the wandb artifacts where the model is located.
    - `max_model_len`: Maximum token length of the input.
    - `chat_template`: Path to the chat template file. This is required for open-weights models.
    - `dtype`: Data type. Choose from float32, float16, bfloat16.
    - `trust_remote_code`:  Default is true.
    - `device_map`: Device map. Default is "auto".
    - `load_in_8bit`: 8-bit quantization. Default is false.
    - `load_in_4bit`: 4-bit quantization. Default is false.

- **generator:** Settings for generation. For more details, refer to the [generation_utils](https://huggingface.co/docs/transformers/internal/generation_utils) in Hugging Face Transformers.
    - `top_p`: top-p sampling. Default is 1.0.
    - `temperature`: The temperature for sampling. Default is 0.1.
    - `max_tokens`: Maximum number of tokens to generate. This value will be overwritten in the script.

- **ichikara:**  Settings for the ichikara dataset.
    - `data_path`: URL of the Weave Dataset for the ichikara dataset.
    - `auto_eval_model`: Model used for LLM as a judge.
    - `upload_human_evaled_file`: Path to the csv file that has undergone human evaluation.

### Model configuration
After setting up the base-configuration file, the next step is to set up a configuration file for model under `configs/`.
#### API Model Configurations
This framework supports evaluating models using APIs such as OpenAI, Anthropic, Google, and Cohere. You need to create a separate config file for each API model. For example, the config file for OpenAI's gpt-4o-2024-05-13 would be named `configs/config-gpt-4o-2024-05-13.yaml`.

- **wandb:** Information used for Weights & Biases (W&B) support.
    - `run_name`: Name of the W&B run.
- **model:** Information about the model. 
    - `pretrained_model_name_or_path`: Name of the API model.
    - `size_category`: Specify "api" to indicate using an API model.
    - `size`: Model size (leave as null for API models).
    - `release_date`: Model release date. (MM/DD/YYYY)

#### Other Model Configurations

This framework also supports evaluating models using VLLM.  You need to create a separate config file for each VLLM model. For example, the config file for Microsoft's Phi-3-medium-128k-instruct would be named `configs/config-Phi-3-medium-128k-instruct.yaml`.

- **wandb:** Information used for Weights & Biases (W&B) support.
    - `run_name`: Name of the W&B run.
- **num_gpus:** Number of GPUs to use.
- **batch_size:** Batch size for VLLM (recommended: 256).
- **model:** Information about the model.
    - `artifacts_path`: When loading a model from wandb artifacts, it is necessary to include a description. If not, there is no need to write it. Example notation: wandb-japan/llm-leaderboard/llm-jp-13b-instruct-lora-jaster-v1.0:v0   
    - `pretrained_model_name_or_path`: Name of the VLLM model.
    - `chat_template`: Path to the chat template file (if needed).
    - `size_category`: Specify model size category. In Nejumi Leaderboard, the category is defined as "10B<", "10B<= <30B", "<=30B" and "api".
    - `size`: Model size (parameter).
    - `release_date`: Model release date (MM/DD/YYYY).
    - `max_model_len`: Maximum token length of the input (if needed).


#### Create Chat template (needed for models except for API)
1. create chat_templates/model_id.jinja
If the chat_template is specified in the tokenizer_config.json of the evaluation model, create a .jinja file with that configuration.
If chat_template is not specified in tokenizer_config.json, refer to the model card or other relevant documentation to create a chat_template and document it in a .jinja file.

2. test chat_templates
If you want to check the output of the chat_templates, you can use the following script:
```bash
python3 scripts/test_chat_template.py -m <model_id> -c <chat_template>
```
If the model ID and chat_template are the same, you can omit -c <chat_template>.

## Contributing
Contributions to this repository is welcom. Please submit your suggestions via pull requests. Please note that we may not accept all pull requests.

## License
This repository is available for commercial use. However, please adhere to the respective rights and licenses of each evaluation dataset used.

## Contact
For questions or support, please concatct to contact-jp@wandb.com.
