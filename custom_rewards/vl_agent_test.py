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
from datetime import datetime

import requests
from math_verify import parse, verify
from openai import OpenAI
from rouge_score import rouge_scorer

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
    #     chat_template = """
    # Below are two answers to a question. Question is [Question], [Standard Answer] is the
    # standard answer to the question,
    # and [Model_answer] is the answer extracted from a model's output to this question.
    # Determine whether these two answers are consistent.
    # Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same.
    # If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
    # If they are consistent, Judement is 1; if they are different, Judement is 0.
    # Just output Judement and don't output anything else.\n\n
    # """
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, 
and [Model_answer] is the answer extracted from a model's output to this question. 

Judge how consistent the two answers are.

Scoring rules  
• 1    — Fully consistent: they convey the same meaning (e.g., “pink” vs. “it is pink”).  
• 0.5 — Partially consistent: they overlap on some key points but not all.  
• 0    — Inconsistent: they conflict or share no essential overlap.

Output **only** one of the following numbers: 1, 0.5, or 0.
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
[Question]: What happens immediately after the fireworks illuminate the sky?
[Standard Answer]: The crowd cheers loudly and waves flags.
[Model_answer] : The crowd cheers.
Judgement: 0.5
"""  # noqa

    example_4 = """
[Question]: What items does the waitress hand to the customer?
[Standard Answer]: She hands over a sandwich and a cup of coffee.
[Model_answer] : She hands over a sandwich and a cup of tea.
Judgement: 0.5
"""  # noqa

    example_5 = """
[Question]: Where is the cat sitting when the dog first walks into the kitchen?
[Standard Answer]: On top of the kitchen counter.
[Model_answer] : In the kitchen, sitting on the floor near the counter.
Judgement: 0.5
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
    # 使用非贪婪模式匹配<answer>和</answer>之间的内容
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def compute_score(predict_str: str, ground_truth: str, extra_info=None, **kwargs) -> float:
    is_format_error = False

    # 基本标签匹配检查
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    count_tool_call_1 = predict_str.count("<tool_call>")
    count_tool_call_2 = predict_str.count("</tool_call>")
    count_answer_1 = predict_str.count("<answer>")
    count_answer_2 = predict_str.count("</answer>")

    # 检查标签是否匹配
    if count_think_1 != count_think_2 or count_think_1 == 0:
        is_format_error = True
    if count_tool_call_1 != count_tool_call_2:
        is_format_error = True
    if count_answer_1 != count_answer_2 or count_answer_1 != 1:  # 必须有且仅有一个answer
        is_format_error = True

    # 严格格式检查
    if not is_format_error:
        if count_tool_call_1 == 0:
            # 不使用tool的情况：<think>...</think><answer>...</answer>
            # 允许前后有空白符
            pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
            if not re.match(pattern, predict_str, re.DOTALL):
                is_format_error = True
        else:
            # 使用tool的情况：必须严格按照 <think></think><tool_call></tool_call><think></think>... 的交替模式
            # 使用更精确的方法检查结构
            stripped_str = predict_str.strip()

            # 检查是否以<think>开头，以</answer>结尾
            if not (stripped_str.startswith("<think>") and stripped_str.endswith("</answer>")):
                is_format_error = True
            else:
                # 分析标签序列，确保tool_call和think正确交替
                # 找到所有开始标签的位置和类型
                tag_pattern = r"<(think|tool_call|answer)>"
                tags = re.findall(tag_pattern, stripped_str)

                # 期望的模式：think, (tool_call, think)*, answer
                expected_pattern = ["think"]
                for _ in range(count_tool_call_1):
                    expected_pattern.extend(["tool_call", "think"])
                expected_pattern.append("answer")

                if tags != expected_pattern:
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
        if "Judgement:" in response:
            response = response.split("Judgement:")[-1].strip()
            if "1" in response:
                acc_reward = 1.0
            elif "0.5" in response:
                acc_reward = 0.5
            elif "0" in response:
                acc_reward = 0.0
            else:
                print(f" [WARNING] resp format error {response=}")
                acc_reward = 0.0
        else:
            if response == "1":
                acc_reward = 1.0
            elif response == "0.5":
                acc_reward = 0.5
            elif response == "0":
                acc_reward = 0.0
            else:
                print(f" [WARNING] resp format error {response=}")
                acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    format_reward = 0.0 if is_format_error else 1.0

    # args definition
    tool_use_reward = kwargs.get("tool_use_reward", False)  # tool reawrd
    use_time_reward = kwargs.get("use_time_reward", False)  # recall reward
    use_iou_reward = kwargs.get("use_iou_reward", False)  # iou reward
    use_tiou_reward = kwargs.get("use_tiou_reward", False)  # tiou reward (IoU v2 with normalized temporal distance)
    use_new_reward = kwargs.get("use_new_reward", False)  # acc:format = 1:1

    tool_reward = 0.0
    if tool_use_reward:
        count_vision_response_1 = predict_str.count("<tool_response>")
        tool_reward = 1.0 if count_vision_response_1 > 0 and acc_reward >= 0.5 else 0.0
        return (1.0 * acc_reward + 1.0 * format_reward + 1.0 * tool_reward, acc_reward, format_reward, tool_reward)
    if use_time_reward:
        count_vision_response_1 = predict_str.count("<tool_response>")
        if count_vision_response_1 > 0:
            ground_truth_time = extra_info["video_segment"]

            # extract the time from the tool_response
            time_reward = 0.0
            try:
                # find all tool_calls and get the last crop_video
                tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
                tool_calls = re.findall(tool_call_pattern, predict_str, re.DOTALL)

                # find the last crop_video tool call
                last_crop_video = None
                for tool_call in reversed(tool_calls):  # iterate from the end
                    try:
                        tool_data = ast.literal_eval(tool_call.strip())
                        if isinstance(tool_data, dict) and tool_data.get("name") == "crop_video":
                            arguments = tool_data.get("arguments", {})
                            if "start_time" in arguments and "end_time" in arguments:
                                last_crop_video = (float(arguments["start_time"]), float(arguments["end_time"]))
                                break  # found the last crop_video, exit immediately
                    except (ValueError, SyntaxError, KeyError):
                        continue

                # calculate time reward using the last crop_video
                if last_crop_video:
                    pred_start, pred_end = last_crop_video

                    # get the ground truth time interval
                    # Support both list and numpy array
                    if (isinstance(ground_truth_time, list | tuple) or hasattr(ground_truth_time, "__len__")) and len(
                        ground_truth_time
                    ) == 2:
                        gt_start, gt_end = float(ground_truth_time[0]), float(ground_truth_time[1])

                        # use recall to calculate the time reward (the same logic as compute_score_time_r1)
                        intersection_start = max(pred_start, gt_start)
                        intersection_end = min(pred_end, gt_end)
                        intersection = max(0, intersection_end - intersection_start)

                        gt_length = gt_end - gt_start
                        if gt_length > 0:
                            time_reward = intersection / gt_length
                        else:
                            time_reward = 1.0 if intersection > 0 else 0.0
            except Exception:
                time_reward = 0.0
        else:
            time_reward = 0.0

        return (1.0 * acc_reward + 1.0 * format_reward + 1.0 * time_reward, acc_reward, format_reward, time_reward)

    if use_iou_reward:
        count_vision_response_1 = predict_str.count("<tool_response>")
        if count_vision_response_1 > 0:
            ground_truth_time = extra_info["video_segment"]

            # extract the time from the tool_response
            time_reward = 0.0
            try:
                # find all tool_calls and get the last crop_video
                tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
                tool_calls = re.findall(tool_call_pattern, predict_str, re.DOTALL)

                # find the last crop_video tool call
                last_crop_video = None
                for tool_call in reversed(tool_calls):  # iterate from the end
                    try:
                        tool_data = ast.literal_eval(tool_call.strip())
                        if isinstance(tool_data, dict) and tool_data.get("name") == "crop_video":
                            arguments = tool_data.get("arguments", {})
                            if "start_time" in arguments and "end_time" in arguments:
                                last_crop_video = (float(arguments["start_time"]), float(arguments["end_time"]))
                                break  # found the last crop_video, exit immediately
                    except (ValueError, SyntaxError, KeyError):
                        continue

                # calculate time reward using the last crop_video with IoU
                if last_crop_video:
                    pred_start, pred_end = last_crop_video

                    # get the ground truth time interval
                    # Support both list and numpy array
                    if (isinstance(ground_truth_time, list | tuple) or hasattr(ground_truth_time, "__len__")) and len(
                        ground_truth_time
                    ) == 2:
                        gt_start, gt_end = float(ground_truth_time[0]), float(ground_truth_time[1])

                        # use IoU to calculate the time reward (the same logic as
                        # compute_score_time_r1 with use_recall=False)
                        intersection_start = max(pred_start, gt_start)
                        intersection_end = min(pred_end, gt_end)
                        intersection = max(0, intersection_end - intersection_start)

                        # compute union
                        union_start = min(pred_start, gt_start)
                        union_end = max(pred_end, gt_end)
                        union = union_end - union_start

                        # compute IoU
                        if union > 0:
                            time_reward = intersection / union
                        else:
                            time_reward = 1.0 if intersection > 0 else 0.0
            except Exception:
                time_reward = 0.0
        else:
            time_reward = 0.0

        return (1.0 * acc_reward + 1.0 * format_reward + 1.0 * time_reward, acc_reward, format_reward, time_reward)

    if use_tiou_reward:
        count_vision_response_1 = predict_str.count("<tool_response>")
        if count_vision_response_1 > 0:
            ground_truth_time = extra_info["video_segment"]
            video_duration = extra_info.get("duration", None)

            # extract the time from the tool_response
            tiou_reward = 0.0
            try:
                # find all tool_calls and get the last crop_video
                tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
                tool_calls = re.findall(tool_call_pattern, predict_str, re.DOTALL)

                # find the last crop_video tool call
                last_crop_video = None
                for tool_call in reversed(tool_calls):  # iterate from the end
                    try:
                        tool_data = ast.literal_eval(tool_call.strip())
                        if isinstance(tool_data, dict) and tool_data.get("name") == "crop_video":
                            arguments = tool_data.get("arguments", {})
                            if "start_time" in arguments and "end_time" in arguments:
                                last_crop_video = (float(arguments["start_time"]), float(arguments["end_time"]))
                                break  # found the last crop_video, exit immediately
                    except (ValueError, SyntaxError, KeyError):
                        continue

                # calculate tiou reward (IoU v2 with normalized temporal distance)
                if last_crop_video and video_duration is not None:
                    pred_start, pred_end = last_crop_video

                    # get the ground truth time interval
                    # Support both list and numpy array
                    if (isinstance(ground_truth_time, list | tuple) or hasattr(ground_truth_time, "__len__")) and len(
                        ground_truth_time
                    ) == 2:
                        gt_start, gt_end = float(ground_truth_time[0]), float(ground_truth_time[1])

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
                            iou = intersection / union
                        else:
                            iou = 1.0 if intersection > 0 else 0.0

                        # compute normalized temporal distances (same as time-r1's iou_timestamp_reward_v2)
                        gt_start_norm = gt_start / video_duration
                        gt_end_norm = gt_end / video_duration
                        pred_start_norm = pred_start / video_duration
                        pred_end_norm = pred_end / video_duration

                        # tiou = IoU * (1 - |gt_start_norm - pred_start_norm|) * (1 - |gt_end_norm - pred_end_norm|)
                        tiou_reward = (
                            iou * (1 - abs(gt_start_norm - pred_start_norm)) * (1 - abs(gt_end_norm - pred_end_norm))
                        )
            except Exception as e:
                print(f" [WARNING] tiou reward computation error: {e}")
                tiou_reward = 0.0
        else:
            tiou_reward = 0.0

        return (1.0 * acc_reward + 1.0 * format_reward + 1.0 * tiou_reward, acc_reward, format_reward, tiou_reward)

    if use_new_reward:
        return (1.0 * acc_reward + 1.0 * format_reward, acc_reward, format_reward)
    else:
        return (0.8 * acc_reward + 0.2 * format_reward, acc_reward, format_reward)


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

    return 0.8 * acc_reward + 0.2 * format_reward, acc_reward, format_reward


def compute_score_time_r1(predict_str: str, ground_truth: str, extra_info=None, use_recall=False) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2 or count_think_1 == 0:  # reward hacking
        is_format_error = True

    count_tool_call_1 = predict_str.count("<tool_call>")
    count_tool_call_2 = predict_str.count("</tool_call>")
    if count_tool_call_1 != count_tool_call_2:
        is_format_error = True

    predict_no_think = predict_str.split("</think>")[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2 or count_answer_1 == 0:
        is_format_error = True

    # more strict format check
    if not is_format_error:
        if count_tool_call_1 == 0:
            # <think>...</think><answer>...</answer>
            pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
            if not re.match(pattern, predict_str, re.DOTALL):
                is_format_error = True
        else:
            # use tool case: must strictly follow the
            # alternating pattern of <think></think><tool_call></tool_call><think></think>...

            stripped_str = predict_str.strip()

            if not (stripped_str.startswith("<think>") and stripped_str.endswith("</answer>")):
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

    return 1.0 * acc_reward + 1.0 * format_reward, acc_reward, format_reward


def compute_score_videor1(predict_str: str, ground_truth: str, extra_info=None, **kwargs) -> float:
    """
    Video-R1 style reward computation with accuracy and format rewards.

    Args:
        predict_str: Model prediction string
        ground_truth: Ground truth answer
        extra_info: Dictionary containing additional info like 'question' and 'problem_type'
        **kwargs: Additional arguments like temporal, len_control settings

    Returns:
        Tuple of (total_reward, accuracy_reward, format_reward) or just total_reward
    """

    def extract_answer_videor1(text):
        """Extract answer from <answer></answer> tags"""
        pattern = r"<answer>\s*(.*?)\s*</answer>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        """Normalize number string to float"""
        try:
            num_str = num_str.replace(",", "")
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        """Word Error Rate calculation"""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            d[i][0] = i
        for j in range(n + 1):
            d[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
        return d[m][n] / max(1, m)

    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        """Compute ROUGE score"""
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores["rouge1"].fmeasure + scores["rouge2"].fmeasure + scores["rougeL"].fmeasure) / 3
        return average_fmeasure

    # Initialize variables
    is_format_error = False

    # Format checking - Video-R1 style
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    count_answer_1 = predict_str.count("<answer>")
    count_answer_2 = predict_str.count("</answer>")

    # Check format requirements
    if count_think_1 != count_think_2 or count_think_1 == 0:
        is_format_error = True
    if count_answer_1 != count_answer_2 or count_answer_1 == 0:
        is_format_error = True

    # Check basic format pattern
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    if not re.search(pattern, predict_str, re.DOTALL):
        is_format_error = True

    # Extract answer
    if count_answer_1 == 0 or count_answer_2 == 0:
        answer_text = ""
    else:
        answer_text = extract_answer_videor1(predict_str)

    # Compute accuracy reward
    # If there's a format error, set accuracy reward to 0 regardless of answer content
    if answer_text == "" or is_format_error:
        acc_reward = 0.0
    else:
        try:
            # Get problem type from extra_info
            problem_type = extra_info.get("problem_type", "free-form") if extra_info else "free-form"

            output_ans = answer_text
            gt_ans = extract_answer_videor1(ground_truth) if "<answer>" in ground_truth else ground_truth

            if problem_type == "multiple choice":
                acc_reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif problem_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    acc_reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        acc_reward = 0.0
                    else:
                        acc_reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif problem_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                acc_reward = 1 - error_rate
                acc_reward = max(0.0, min(1.0, acc_reward))
            elif problem_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                acc_reward = max(0.0, min(1.0, score))
            elif problem_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    acc_reward = 0.0
                else:
                    rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                    rel_diff = min(1.0, max(0.0, rel_diff))
                    acc_reward = 1 - rel_diff
            else:
                # Unknown problem type - return 0 (same as original Video-R1)
                acc_reward = 0.0

        except Exception as e:
            print(f"Error in Video-R1 accuracy reward computation: {e}")
            acc_reward = 0.0

    # Penalize for overly long answers
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    # Format reward
    format_reward = 0.0 if is_format_error else 1.0

    # Debug logging
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH", "./videor1_debug.log")
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"------------- {current_time} Video-R1 Reward -------------\n")
            f.write(f"Prediction: {predict_str}\n")
            f.write(f"Ground Truth: {ground_truth}\n")
            f.write(f"Extracted Answer: {answer_text}\n")
            f.write(f"Accuracy Reward: {acc_reward}\n")
            f.write(f"Format Reward: {format_reward}\n")
            f.write(f"Format Error: {is_format_error}\n")
            f.write("=" * 50 + "\n")

    return (acc_reward + format_reward, acc_reward, format_reward)


def test_compute_score_videor1():
    """Test function for compute_score_videor1"""
    print("\n=== 测试 Video-R1 Reward 计算 ===")

    # 测试1: 完美格式和正确答案 - multiple choice
    predict_str1 = "<think>这是一个选择题，我需要分析选项</think><answer>A</answer>"
    ground_truth1 = "A"
    extra_info1 = {"question": "What is the correct answer?", "problem_type": "multiple choice"}
    score1 = compute_score_videor1(predict_str1, ground_truth1, extra_info1, return_detailed=True)
    print(f"测试1 - 选择题正确: {score1} (期望: (2.0, 1.0, 1.0))")

    # 测试2: 格式错误 - 缺少think标签
    predict_str2 = "<answer>A</answer>"
    ground_truth2 = "A"
    extra_info2 = {"question": "What is the correct answer?", "problem_type": "multiple choice"}
    score2 = compute_score_videor1(predict_str2, ground_truth2, extra_info2, return_detailed=True)
    print(f"测试2 - 格式错误: {score2} (期望: (0.0, 0.0, 0.0))")

    # 测试3: 数值类型 - 正确答案
    predict_str3 = "<think>计算结果是42</think><answer>42</answer>"
    ground_truth3 = "42"
    extra_info3 = {"question": "What is 6*7?", "problem_type": "numerical"}
    score3 = compute_score_videor1(predict_str3, ground_truth3, extra_info3, return_detailed=True)
    print(f"测试3 - 数值正确: {score3} (期望: (2.0, 1.0, 1.0))")

    # 测试4: 数值类型 - 错误答案
    predict_str4 = "<think>计算结果是43</think><answer>43</answer>"
    ground_truth4 = "42"
    extra_info4 = {"question": "What is 6*7?", "problem_type": "numerical"}
    score4 = compute_score_videor1(predict_str4, ground_truth4, extra_info4, return_detailed=True)
    print(f"测试4 - 数值错误: {score4} (期望: (1.0, 0.0, 1.0))")

    # 测试5: 自由形式 - 使用ROUGE评分
    predict_str5 = "<think>这是一个描述性问题</think><answer>The cat is sitting on the mat</answer>"
    ground_truth5 = "The cat sits on the mat"
    extra_info5 = {"question": "Describe what you see", "problem_type": "free-form"}
    score5 = compute_score_videor1(predict_str5, ground_truth5, extra_info5, return_detailed=True)
    print(f"测试5 - 自由形式: {score5[0]:.3f} (ROUGE评分 + 格式分)")

    # 测试6: 答案过长 - 应该被惩罚
    long_answer = "A" * 1001  # 超过1000字符
    predict_str6 = f"<think>思考过程</think><answer>{long_answer}</answer>"
    ground_truth6 = "A"
    extra_info6 = {"question": "What is the answer?", "problem_type": "multiple choice"}
    score6 = compute_score_videor1(predict_str6, ground_truth6, extra_info6, return_detailed=True)
    print(f"测试6 - 答案过长: {score6} (期望: (0.0, 0.0, 0.0))")

    print("=== Video-R1 Reward 测试完成 ===\n")


if __name__ == "__main__":
    # 测试新的Video-R1 reward函数
    test_compute_score_videor1()
