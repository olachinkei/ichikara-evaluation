import pandas as pd

# 
def adjust_csv_columns(file_path, reference_columns, output_path):
    df = pd.read_csv(file_path)
    df_columns = df.columns.tolist()
    columns_to_keep = [col for col in reference_columns if col in df_columns]
    df_filtered = df[columns_to_keep]
    empty_columns = ['関連性', '正確性', '流暢性', '情報量', '総合評価', '理由']

    for col in empty_columns:
        if col not in df_filtered.columns:
            df_filtered[col] = ''

    for col in reference_columns:
        if col not in df_filtered.columns:
            df_filtered[col] = ''


    df_filtered = df_filtered[reference_columns]
    df_filtered.to_csv(output_path, index=False)
    print(f"File saved successfully to {output_path}")

# Reference columns (使用したいカラム)
reference_columns = [
    'inputs.example.text', 
    'output.model_output.generated_text.content',
    '関連性', '正確性', '流暢性', '情報量', '総合評価', '理由',
    'output.model_output.generated_text.response_metadata.model_name',
    'output.scores.scores.individual_score.関連性',
    'output.scores.scores.individual_score.正確性',
    'output.scores.scores.individual_score.流暢性',
    'output.scores.scores.individual_score.情報量',
    'output.scores.scores.domain_score.ビジネス',
    'output.scores.scores.domain_score.経済',
    'output.scores.scores.domain_score.教育',
    'output.scores.scores.domain_score.医療',
    'output.scores.scores.domain_score.法律',
    'output.scores.scores.総合評価',
    'inputs.example.meta.source-to-answer',
    'inputs.example.meta.output-reference',
    'inputs.example.meta.time-dependency',
    'inputs.example.meta.output-producer',
    'inputs.example.meta.text-producer',
    'inputs.example.meta.perspective',
    'inputs.example.meta.output-type',
    'inputs.example.meta.alert-type',
    'inputs.example.meta.domain',
    'inputs.example.meta.task',
    'inputs.example.ID'
]

# Weaveで自動評価後のデータをDLし、inputとする
input_file_path = '/workspace/tokyotech-llmLlama-3-Swallow-70B-Instruct-v01.csv'
output_file_path = '/workspace/tokyotech-llmLlama-3-Swallow-70B-Instruct-v01_for_humaneval.csv'

# Adjust the CSV file columns to match the reference structure
adjust_csv_columns(input_file_path, reference_columns, output_file_path)