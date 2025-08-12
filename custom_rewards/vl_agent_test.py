# Copyright 2025 Individual Contributor: Sudong Wang, Zuhao Yang, Kaichen Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import os
import random
import re

import requests
from math_verify import parse, verify
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base_list = [
    os.environ.get("LLM_AS_A_JUDGE_BASE", "https://sd285v869b9467c7sab70.apigateway-cn-shanghai.volceapi.com/v1"),
]

client_list = []
for api_base in openai_api_base_list:
    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
    )
    client_list.append(client)
model_name_list = []
for client in client_list:
    response = requests.get(f"{api_base}/models")
    models = response.json()
    model_name_list.append(models["data"][0]["id"])


def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, 
and [Model_answer] is the answer extracted from a model's output to this question.  
Determine whether these two answers are consistent. 
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. 
If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. 
Just output Judement and don't output anything else.\n\n
"""
    return chat_template


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
"""  # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
"""  # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
"""  # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
"""  # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
"""  # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
"""  # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
"""  # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


COMMON_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level reasoning problems. 
I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. 
My job is to assess whether the student's answer captures the same meaning as the reference answer, 
even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Semantic Equivalence: Carefully examine the expression in both answers. 
Confirm whether the semantic meaning of student's final answer is equivalent to the reference answer, 
even when expressed with different wording or format.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. 
 You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. 
 The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. 
I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. 
My job is to assess whether the student's answer captures the same meaning as the reference answer, 
even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. 
Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether 
 the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is 
 FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + "\n\n"
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f"{demo_prompt}{test_prompt}"

    return full_prompt


def extract_answer(text):
    """
    从给定的文本中提取<answer></answer>标签内部的内容。

    参数:
        text (str): 包含<answer>标签的文本

    返回:
        str or None: 标签内部的内容，如果未找到则返回None。
    """
    # 使用非贪婪模式匹配<answer>和</answer>之间的内容
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def compute_score_think(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    # there is a <think> in chat_template
    if count_think_1 + 1 != count_think_2 or count_think_2 == 0:  # exclude the situation that <think>==</think>==0
        is_format_error = True

    count_vision_1 = predict_str.count("<tool_call>")
    count_vision_2 = predict_str.count("</tool_call>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    # count_vision_response_1 = predict_str.count("<tool_response>")
    # count_vision_response_2 = predict_str.count("</tool_response>")
    # if count_vision_response_1 != count_vision_response_2:
    #     is_format_error = True

    predict_no_think = predict_str.split("</think>")[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2 or count_answer_1 == 0:  ##
        is_format_error = True

    if count_answer_1 == 0 or count_answer_2 == 0:
        answer_text = ""
    else:
        answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    # skip the case that the answer is empty
    if answer_text == "":
        acc_reward = 0.0
    else:
        question_text = extra_info["question"]
        full_prompt = get_prompt(answer_text, ground_truth, question_text)

        client_idx = random.randint(0, len(client_list) - 1)
        client = client_list[client_idx]
        model_name = model_name_list[client_idx]

        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt},
            ],
            seed=random.randint(0, 1000000),
            temperature=0.3,
        )
        response = chat_response.choices[0].message.content.strip()
        # print(response)
        if "Judgement:" in response:
            response = response.split("Judgement:")[-1].strip()
            if "1" in response:
                acc_reward = 1.0
            elif "0" in response:
                acc_reward = 0.0
            else:
                print(f" [WARNING] resp format error {response=}")
                acc_reward = 0.0
        else:
            if response == "1":
                acc_reward = 1.0
            elif response == "0":
                acc_reward = 0.0
            else:
                print(f" [WARNING] resp format error {response=}")
                acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    # tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    # tool_reward = 1.0 if count_vision_response_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = 0.0 if is_format_error else 1.0

    return 0.8 * acc_reward + 0.2 * format_reward, acc_reward, format_reward


def compute_score(predict_str: str, ground_truth: str, extra_info=None, **kwargs) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2 or count_think_1 == 0:  # exclude the situation that <think>==</think>==0
        is_format_error = True

    count_vision_1 = predict_str.count("<tool_call>")
    count_vision_2 = predict_str.count("</tool_call>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    # count_vision_response_1 = predict_str.count("<tool_response>")
    # count_vision_response_2 = predict_str.count("</tool_response>")
    # if count_vision_response_1 != count_vision_response_2:
    #     is_format_error = True

    predict_no_think = predict_str.split("</think>")[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2 or count_answer_1 == 0:  ##
        is_format_error = True

    if count_answer_1 == 0 or count_answer_2 == 0:
        answer_text = ""
    else:
        answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    # skip the case that the answer is empty
    if answer_text == "":
        acc_reward = 0.0
    else:
        question_text = extra_info["question"]
        full_prompt = get_prompt(answer_text, ground_truth, question_text)

        client_idx = random.randint(0, len(client_list) - 1)
        client = client_list[client_idx]
        model_name = model_name_list[client_idx]

        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt},
            ],
            seed=random.randint(0, 1000000),
            temperature=0.3,
        )
        response = chat_response.choices[0].message.content.strip()
        # print(response)
        if "Judgement:" in response:
            response = response.split("Judgement:")[-1].strip()
            if "1" in response:
                acc_reward = 1.0
            elif "0" in response:
                acc_reward = 0.0
            else:
                print(f" [WARNING] resp format error {response=}")
                acc_reward = 0.0
        else:
            if response == "1":
                acc_reward = 1.0
            elif response == "0":
                acc_reward = 0.0
            else:
                print(f" [WARNING] resp format error {response=}")
                acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    # tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    # tool_reward = 1.0 if count_vision_response_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = 0.0 if is_format_error else 1.0

    tool_use_reward = kwargs.get("tool_use_reward", False)
    use_new_reward = kwargs.get("use_new_reward", False)

    tool_reward = 0.0
    if tool_use_reward:
        count_vision_response_1 = predict_str.count("<tool_response>")
        tool_reward = 1.0 if count_vision_response_1 > 0 and acc_reward > 0.5 else 0.0
        return (1.0 * acc_reward + 1.0 * format_reward + 1.0 * tool_reward, acc_reward, format_reward, tool_reward)

    if use_new_reward:
        return (1.0 * acc_reward + 1.0 * format_reward, acc_reward, format_reward)
    else:
        return (0.8 * acc_reward + 0.2 * format_reward, acc_reward, format_reward)


def compute_common_reasoning(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split("</think>")[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = extract_answer(
        predict_no_think
    )  # predict_no_think.split("<answer>")[-1].split("</answer>")[0].strip()
    if not answer_text:
        acc_reward = 0.0
        is_format_error = True
    elif len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True
    else:
        question_text = extra_info["question"]
        client_idx = random.randint(0, len(client_list) - 1)
        client = client_list[client_idx]
        model_name = model_name_list[client_idx]
        full_prompt = COMMON_VERIFY_PROMPT.format(
            query=question_text,
            gold_ans=ground_truth,
            pred_ans=answer_text,
        )

        acc_reward = 0.0
        for ix in range(8):
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed=random.randint(0, 1000000),
                temperature=0.5,
            )
            response = chat_response.choices[0].message.content.strip()
            judgement = response.split("## Equivalence Judgement")[-1].lower()
            if "true" in judgement and "false" not in judgement:
                acc_reward = 1.0
                break
            elif "false" in judgement and "true" not in judgement:
                acc_reward = 0.0
                break
            else:
                print(f" [ERROR] judgement format invalid: {judgement}")
                continue

    # tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    print(
        f"[DEBUG] tool_query={extra_info['question']}, {ground_truth=}, {answer_text=}, {acc_reward=}, {format_reward=}"
    )
    return 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward


def rule_math_verify(ground_truth, model_answer):
    gold = parse(ground_truth)
    answer = parse(model_answer)
    return verify(gold, answer)


def generative_verify(query, ground_truth, model_answer):
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    full_prompt = MATH_VERIFY_PROMPT.format(
        query=query,
        gold_ans=ground_truth,
        pred_ans=model_answer,
    )

    response = ""
    for it in range(8):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed=random.randint(0, 1000000),
                temperature=0.0,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f" [ERROR math] generative_verify error: {e}")
            continue

    judgement = response.split("## Equivalence Judgement")[-1].lower()
    if "true" in judgement and "false" not in judgement:
        return True
    elif "false" in judgement and "true" not in judgement:
        return False
    else:
        print(" [ERROR math] verify bug output: ")


def compute_score_math(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2 or count_think_1 == 0:  # reward hacking
        is_format_error = True

    predict_no_think = predict_str.split("</think>")[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2 or count_answer_1 == 0:
        is_format_error = True

    # extract answer content from answer tag
    if count_answer_1 == 0 or count_answer_2 == 0:
        answer_content = ""
    else:
        answer_content = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    model_answer = ""
    if answer_content == "":
        acc_reward = 0.0
    else:
        answer_pattern = r"\\boxed{([^}]+)}"
        answer_list = re.findall(answer_pattern, answer_content, flags=re.DOTALL)
        if len(answer_list) == 0:
            acc_reward = 0.0
            is_format_error = True
        else:
            if len(answer_list) > 1:
                is_format_error = True

            model_answer = answer_list[-1]
            if rule_math_verify(ground_truth, model_answer):
                acc_reward = 1.0
            else:
                acc_reward = 1.0 if generative_verify(extra_info["question"], ground_truth, model_answer) else 0.0

    format_reward = 0.0 if is_format_error else 1.0
    # print(
    #     f"[DEBUG] math_query={extra_info['question']}, "
    #     f"{ground_truth=}, "
    #     f"{model_answer=}, "
    #     f"{acc_reward=}, "
    #     f"{format_reward=}"
    # )

    return 0.8 * acc_reward + 0.2 * format_reward, acc_reward, format_reward


def compute_score_time_r1(predict_str: str, ground_truth: str, extra_info=None, use_recall=False) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2 or count_think_1 == 0:  # reward hacking
        is_format_error = True

    count_vision_1 = predict_str.count("<tool_call>")
    count_vision_2 = predict_str.count("</tool_call>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split("</think>")[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2 or count_answer_1 == 0:
        is_format_error = True

    # extract answer content from answer tag
    if count_answer_1 == 0 or count_answer_2 == 0:
        answer_content = ""
    else:
        answer_content = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    if answer_content == "":
        acc_reward = 0.0
    else:
        try:
            predicted_interval = ast.literal_eval(answer_content)
            if not isinstance(predicted_interval, list) or len(predicted_interval) != 2:
                acc_reward = 0.0
            else:
                pred_start, pred_end = float(predicted_interval[0]), float(predicted_interval[1])

                ground_truth_interval = ast.literal_eval(ground_truth)
                if not isinstance(ground_truth_interval, list) or len(ground_truth_interval) != 2:
                    acc_reward = 0.0
                else:
                    gt_start, gt_end = float(ground_truth_interval[0]), float(ground_truth_interval[1])

                    if use_recall:
                        # 使用 Recall 计算: 真实区间中被预测区间覆盖的比例
                        # Recall = intersection / ground_truth_length
                        intersection_start = max(pred_start, gt_start)
                        intersection_end = min(pred_end, gt_end)
                        intersection = max(0, intersection_end - intersection_start)

                        gt_length = gt_end - gt_start
                        if gt_length > 0:
                            acc_reward = intersection / gt_length
                        else:
                            acc_reward = 1.0 if intersection > 0 else 0.0
                    else:
                        # 使用 IoU 计算 (原有逻辑)
                        # compute IoU (Intersection over Union)
                        # compute intersection
                        intersection_start = max(pred_start, gt_start)
                        intersection_end = min(pred_end, gt_end)
                        intersection = max(0, intersection_end - intersection_start)

                        # compute union
                        union_start = min(pred_start, gt_start)
                        union_end = max(pred_end, gt_end)
                        union = union_end - union_start

                        # compute IoU
                        if union > 0:
                            acc_reward = intersection / union
                        else:
                            acc_reward = 1.0 if intersection > 0 else 0.0

        except (SyntaxError, ValueError, TypeError):
            acc_reward = 0.0

    format_reward = 0.0 if is_format_error else 1.0

    return 0.8 * acc_reward + 0.2 * format_reward, acc_reward, format_reward


if __name__ == "__main__":
    # predict_str = "The answer is <think> 2 + 2 = 4 </think> <answer> right </answer> <answer> left </answer>"
    # ground_truth = "left"
    # extra_info = {
    #     "answer": "The woman is to the left of the man who is holding the camera.",
    #     "id": 0,
    #     "image": "/cpfs/user/honglingyi/DATA/LLM/Vstar/gqa/images/713270.jpg",
    #     "pred_ans": "The woman is to the right of the man who is holding the camera.",
    #     "question": "Is the woman to the left or to the right of the man who is holding the camera?",
    # }

    # score = compute_score(predict_str, ground_truth, extra_info)
    # print(f"Score: {score}")

    print("\n=== 测试 IoU 时间区间计算 ===")

    # 测试1: 完全重叠 - IoU应该为1.0
    predict_str1 = "<think>分析视频内容</think><answer>[10.5, 20.3]</answer>"
    ground_truth1 = "[10.5, 20.3]"
    score1 = compute_score_time_r1(predict_str1, ground_truth1)
    print(f"测试1 - 完全重叠: 预测={ground_truth1}, 真实={ground_truth1}")
    print(f"Score: {score1:.3f} (期望: 1.0)")

    # 测试2: 部分重叠 - 手动计算IoU
    predict_str2 = "<think>分析视频内容</think><answer>[15.0, 25.0]</answer>"
    ground_truth2 = "[10.0, 20.0]"
    # 交集: [15.0, 20.0] = 5.0
    # 并集: [10.0, 25.0] = 15.0
    # IoU = 5.0/15.0 = 0.333
    score2 = compute_score_time_r1(predict_str2, ground_truth2)
    print("\n测试2 - 部分重叠: 预测=[15.0, 25.0], 真实=[10.0, 20.0]")
    print(f"Score: {score2:.3f} (期望: 0.8*0.333+0.2*1.0 = 0.467)")

    # 测试3: 完全不重叠 - IoU应该为0
    predict_str3 = "<think>分析视频内容</think><answer>[30.0, 40.0]</answer>"
    ground_truth3 = "[10.0, 20.0]"
    score3 = compute_score_time_r1(predict_str3, ground_truth3)
    print("\n测试3 - 完全不重叠: 预测=[30.0, 40.0], 真实=[10.0, 20.0]")
    print(f"Score: {score3:.3f} (期望: 0.8*0.0+0.2*1.0 = 0.2)")

    # 测试4: 格式错误
    predict_str4 = "<think>分析视频内容<answer>[15.0, 25.0]</answer>"  # 缺少</think>
    ground_truth4 = "[10.0, 20.0]"
    score4 = compute_score_time_r1(predict_str4, ground_truth4)
    print("\n测试4 - 格式错误: 缺少</think>标签")
    print(f"Score: {score4:.3f} (期望: 0.267)")

    # 测试5: 解析错误 - 应该返回0.2（只有格式分）
    predict_str5 = "<think>分析视频内容</think><answer>这不是一个列表</answer>"
    ground_truth5 = "[10.0, 20.0]"
    score5 = compute_score_time_r1(predict_str5, ground_truth5)
    print("\n测试5 - 解析错误: answer内容不是列表格式")
    print(f"Score: {score5:.3f} (期望: 0.8*0.0+0.2*1.0 = 0.2)")

    print("\n=== 测试 Recall 时间区间计算 ===")

    # 测试1: 完全重叠 - Recall应该为1.0
    predict_str1 = "<think>分析视频内容</think><answer>[10.5, 20.3]</answer>"
    ground_truth1 = "[10.5, 20.3]"
    score1_recall = compute_score_time_r1(predict_str1, ground_truth1, use_recall=True)
    print(f"测试1 - 完全重叠 (Recall): 预测={ground_truth1}, 真实={ground_truth1}")
    print(f"Score: {score1_recall:.3f} (期望: 1.0)")

    # 测试2: 部分重叠 - 计算Recall
    predict_str2 = "<think>分析视频内容</think><answer>[15.0, 25.0]</answer>"
    ground_truth2 = "[10.0, 20.0]"
    # 交集: [15.0, 20.0] = 5.0
    # 真实区间长度: 20.0-10.0 = 10.0
    # Recall = 5.0/10.0 = 0.5
    score2_recall = compute_score_time_r1(predict_str2, ground_truth2, use_recall=True)
    print("\n测试2 - 部分重叠 (Recall): 预测=[15.0, 25.0], 真实=[10.0, 20.0]")
    print(f"Score: {score2_recall:.3f} (期望: 0.8*0.5+0.2*1.0 = 0.6)")

    # 测试3: 预测区间包含真实区间 - Recall应该为1.0
    predict_str3 = "<think>分析视频内容</think><answer>[5.0, 25.0]</answer>"
    ground_truth3 = "[10.0, 20.0]"
    # 交集: [10.0, 20.0] = 10.0
    # 真实区间长度: 20.0-10.0 = 10.0
    # Recall = 10.0/10.0 = 1.0
    score3_recall = compute_score_time_r1(predict_str3, ground_truth3, use_recall=True)
    print("\n测试3 - 预测包含真实 (Recall): 预测=[5.0, 25.0], 真实=[10.0, 20.0]")
    print(f"Score: {score3_recall:.3f} (期望: 0.8*1.0+0.2*1.0 = 1.0)")

    # 测试4: 完全不重叠 - Recall应该为0
    predict_str4 = "<think>分析视频内容</think><answer>[30.0, 40.0]</answer>"
    ground_truth4 = "[10.0, 20.0]"
    score4_recall = compute_score_time_r1(predict_str4, ground_truth4, use_recall=True)
    print("\n测试4 - 完全不重叠 (Recall): 预测=[30.0, 40.0], 真实=[10.0, 20.0]")
    print(f"Score: {score4_recall:.3f} (期望: 0.8*0.0+0.2*1.0 = 0.2)")

    print("\n=== 对比 IoU vs Recall ===")
    # 对比同一个例子的IoU和Recall结果
    predict_str_cmp = "<think>分析视频内容</think><answer>[15.0, 25.0]</answer>"
    ground_truth_cmp = "[10.0, 20.0]"

    score_iou = compute_score_time_r1(predict_str_cmp, ground_truth_cmp, use_recall=False)
    score_recall = compute_score_time_r1(predict_str_cmp, ground_truth_cmp, use_recall=True)

    print("预测=[15.0, 25.0], 真实=[10.0, 20.0]")
    print(f"IoU Score: {score_iou:.3f}")
    print(f"Recall Score: {score_recall:.3f}")
