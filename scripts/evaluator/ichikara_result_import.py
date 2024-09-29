import weave
from weave import Evaluation
import pandas as pd
import asyncio
from config_singleton import WandbConfigSingleton
from typing import Any, Dict, List
from tenacity import retry, stop_after_attempt, wait_exponential

VALID_DOMAINS = ["法律", "教育", "医療", "経済", "ビジネス"]

def normalize_text(text):
    return ' '.join(text.replace('\\n', ' ').replace('\n', ' ').split()).strip()

def preprocess_data(csv_path: str) -> Dict[str, Dict]:
    df = pd.read_csv(csv_path)
    df['inputs.example.text'] = df['inputs.example.text'].apply(normalize_text)
    return {normalize_text(row['inputs.example.text']): row.to_dict() for _, row in df.iterrows()}

def filter_domains(domains: List[str]) -> List[str]:
    return [domain for domain in domains if domain in VALID_DOMAINS]

def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config

    weave.init(cfg.wandb.entity+"/"+cfg.wandb.project)

    weave_data = weave.ref(cfg.data_path).get()
    weave_data_list = [dict(row) for row in weave_data.rows]

    for item in weave_data_list:
        item['text'] = normalize_text(item['text'])
        if 'meta' in item and 'domain' in item['meta']:
            item['meta']['domain'] = filter_domains(item['meta']['domain'])

    preprocessed_data = preprocess_data("/workspace/test_human_auto_eval.csv")

    def get_matching_row(text: str) -> Dict:
        if text not in preprocessed_data:
            raise ValueError(f"No matching row found for: {text[:100]}...")
        return preprocessed_data[text]

    class LLMinvoke(weave.Model):
        model_name: str
        api_type: str

        @weave.op()
        def predict(self, text: str):
            response = get_matching_row(text)
            return {'generated_text': response["output.model_output.generated_text.content"]}

    model = LLMinvoke(api_type=cfg.api, model_name=cfg.model.pretrained_model_name_or_path)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    @weave.op()
    def judge_with_human(text: str, model_output: str):
        return get_matching_row(text)

    @weave.op()
    def scores(text: str, meta: Any, model_output: dict) -> dict:
        result = judge_with_human(text, model_output["generated_text"])
        values = [
            float(result["関連性"]), float(result["正確性"]),
            float(result["流暢性"]), float(result["情報量"])
        ]
        total_score = sum(values) / len(values)
        domains = meta["domain"]
        domain_score = {}
        for d in domains:
            if d in ["法律", "ビジネス", "経済", "教育", "医療"]:
                domain_score[d] = total_score
        return {
            'individual_score': {
                '関連性': values[0], '正確性': values[1],
                '流暢性': values[2], '情報量': values[3]
            },
            'domain_score': domain_score,
            '総合評価': total_score,
        }
    
    evaluation = Evaluation(dataset=weave_data_list,
                            scorers=[scores],
                            name="test_20240905")
    
    with weave.attributes({'eval_method': 'llm', 'model_name':cfg.model.pretrained_model_name_or_path}):
        asyncio.run(evaluation.evaluate(model))