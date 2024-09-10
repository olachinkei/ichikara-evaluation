import weave
from weave import Evaluation
import json
from typing import Optional, Any
from pydantic import BaseModel
import asyncio
from config_singleton import WandbConfigSingleton
from .evaluate_utils import LLMAsyncProcessor, apply_chat_template
from tenacity import retry, stop_after_attempt, wait_exponential

import openai
from openai import OpenAI


def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm

    # Weave setup
    weave.init(cfg.wandb.entity+"/"+cfg.wandb.project)

    # Data download
    weave_data = weave.ref(cfg.data_path).get()
    weave_data_list = [dict(row) for row in weave_data.rows]
    weave_data_list = weave_data_list
   

    # Model definition
   
    class LLMinvoke(weave.Model):
        model_name: str
        llm: any
        api_type: str

        @weave.op()
        def predict(self, text: dict):
            # here's where you would add your LLM call and return the output
            if self.api_type == "google":
                llm.max_output_tokens = kwargs["max_tokens"]
            message = [{"role": "user", "content": text}]
            response = llm.invoke(message, max_tokens=1024)
            prompt = apply_chat_template(messages=message) # for logging
            return {'generated_text': response}

    model = LLMinvoke(llm=llm, api_type=cfg.api, model_name = cfg.model.pretrained_model_name_or_path)

    #Evaluation definition

    def generate_evalution_prompt(text:str, answer:str):
        evaluation_prompt= f"""あなたは、AIアシスタントの回答を評価する評価者です。
        以下の、ユーザーの質問に対する回答を評価してください。
        
        ###ユーザーの質問:
        {text}

        ###AIアシスタントの回答:
        {answer}
        
        ###評価基準:
        なお、評価は以下の基準に従って、1から5の整数で評価してください（5が最高評価）。
        1. 関連性: 質問に適切に答えているか
        2. 正確性: 提供された情報が正確か
        3. 流暢性: 日本語として自然で読みやすいか
        4. 情報量: 回答が十分に詳細で有用か
        5. 総合評価: 上記をもとにした総合得点
        また、評価を下した理由についてついても述べてください。

        ###注意事項:
        - 日本語以外の言語で出力された回答には低評価をつけてください。
        - 偏見を避け、客観的に評価してください。
        - 評価は必ず下記のJSON形式で提供してください。
        - 短い回答で十分な場合に、不当に"情報量"の点数を低くしないでください。

        ###評価の形式
        以下の形式で評価してください。
        {{"総合評価": "0","関連性": 0, "正確性": 0, "流暢性": 0, "情報量": 0, "理由":<text>}} 
        """
        return evaluation_prompt



    def is_valid_result(result):
        required_keys = ["総合評価", "関連性", "正確性", "流暢性", "情報量", "理由"]
        return all(key in result for key in required_keys) and \
            all(isinstance(result[key], (int, str)) for key in required_keys if key != "理由")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    @weave.op()
    def judge_with_llm(text: str, model_output: str):
        client = OpenAI()
        evaluation_prompt = generate_evalution_prompt(text=text, answer=model_output)
        completion = client.chat.completions.create(
            model="gpt-4-0613",
            messages=[
                {
                    "role": "system",
                    "content": "あなたは、回答の質をチェックするための審判員です。ユーザーの指示に従って下さい。"
                },
                {
                    "role": "user",
                    "content": evaluation_prompt
                }
            ]
        )
        result = completion.choices[0].message.content
        try:
            parsed_result = json.loads(result)
            if not is_valid_result(parsed_result):
                raise ValueError("Invalid result format")
            return parsed_result
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing result: {e}")
            print(f"Raw result: {result}")
            raise  # This will trigger a retry

    @weave.op()
    def scores(text: str, meta: Any, model_output: dict) -> dict:
        result = judge_with_llm(text,model_output["generated_text"])
        values = [result["関連性"], result["正確性"], result["流暢性"], result["情報量"]]
        total_score = result["総合評価"]
        domains = meta["domain"]
        domain_score = {}
        for d in domains:
            domain_score[d] = total_score
        return {
                'individual_score':{
                    '関連性': result["関連性"],'正確性': result["正確性"],'流暢性': result["流暢性"],'情報量': result["情報量"]
                    },
                'domain_score':domain_score,
                '総合評価':result["総合評価"],
                }
    
    evaluation = Evaluation(dataset=weave_data_list,
                            scorers=[scores],
                            name="test_20240905")
    
    # Inference and Evaluation
    with weave.attributes({'eval_method': 'llm', 'model_name':cfg.model.pretrained_model_name_or_path}):
        asyncio.run(evaluation.evaluate(model))





















