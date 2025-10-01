# Copyright 2025 Individual Contributor: Kaichen Zhang
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

import re


def extract_boxed_answer(predict_str: str) -> str:
    """Extract the answer from \boxed{} format.

    Args:
        predict_str (str): The prediction string containing the boxed answer.

    Returns:
        str: The extracted answer from \boxed{}, or an empty string if not found.
    """
    # Look for \boxed{content} pattern
    pattern = re.compile(r"\\boxed\{([^}]*)\}", re.DOTALL)
    matches = re.findall(pattern, predict_str)
    if matches:
        return matches[-1]  # Return the last match if multiple found
    return ""


def extract_anwser_tag(predict_str: str) -> str:
    """Extract the answer tag from the prediction string.

    This function now handles both <answer> tags and \boxed{} format.

    Args:
        predict_str (str): The prediction string containing the answer tag.

    Returns:
        str: The extracted answer tag, or an empty string if not found.
    """
    # First try to extract from <answer> tags
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    match_result = re.search(pattern, predict_str)
    if match_result:
        return match_result.group(1)

    # If no <answer> tag found, try to extract from \boxed{} format
    boxed_answer = extract_boxed_answer(predict_str)
    if boxed_answer:
        return boxed_answer

    # If neither format found, try to extract the last number or expression
    # This is a fallback for cases where the answer is just stated without formatting
    lines = predict_str.strip().split("\n")
    for line in reversed(lines):
        # Look for patterns like "The answer is 204" or just "204"
        if line.strip():
            # Try to find numbers at the end of the line
            number_match = re.search(r"\b(\d+(?:\.\d+)?)\b(?:\s*\.?\s*$)", line)
            if number_match:
                return number_match.group(1)

    return ""


def format_reward(predict_str: str) -> float:
    """Check if the prediction string follows the expected format.

    Now handles both <think><answer> format and \boxed{} format.
    """
    # Check for <think>.*</think>.*<answer>.*</answer> pattern
    think_answer_pattern = re.compile(r"<think>.*</think>.*<answer>.*</answer>", re.DOTALL)
    if re.fullmatch(think_answer_pattern, predict_str):
        return 1.0

    # # Check for \boxed{} format (common in mathematical solutions)
    # if extract_boxed_answer(predict_str):
    #     return 1.0

    # # Check for basic answer format (contains some mathematical content and ends with a number)
    # if len(predict_str.strip()) > 50:  # Reasonable solution length
    #     # Look for mathematical expressions or reasoning
    #     has_math = bool(re.search(r"[=\+\-\*/\(\)\[\]\\]", predict_str))
    #     # Look for final answer
    #     has_answer = bool(extract_anwser_tag(predict_str))

    #     if has_math and has_answer:
    #         return 0.8  # Partial credit for reasonable format

    return 0.0


def simple_parse(predict_str: str) -> str:
    """Parse the prediction string to extract the answer.

    Args:
        predict_str (str): The prediction string to be parsed.

    Returns:
        str: The parsed answer from the prediction string.
    """
    if predict_str.endswith("."):
        predict_str = predict_str[:-1]

    return predict_str.strip()


def relax_exact_match(predict_str: str, ground_truth: str, relax_portion: float = 0.9) -> float:
    """Check if the prediction string matches the ground truth exactly.

    Args:
        predict_str (str): The prediction string to be checked.
        ground_truth (str): The ground truth string for comparison.
        relax_portion (float): The minimum portion of length required for partial matches.

    Returns:
        float: 1.0 if the prediction matches the ground truth, otherwise 0.0.
    """
    if predict_str in ground_truth and len(predict_str) >= relax_portion * len(ground_truth):
        return 1.0
    if ground_truth in predict_str and len(ground_truth) >= relax_portion * len(predict_str):
        return 1.0
    return 1.0 if predict_str.strip() == ground_truth.strip() else 0.0


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.
        sandbox_fusion_url: Not used in this implementation.
        concurrent_semaphore: Not used in this implementation.
        **kwargs: Additional keyword arguments, including use_recall for time-r1 scoring.

    Returns:
        dict: A dictionary containing the computed score and other metrics.
    """
    # breakpoint()
    score_dict = {}
    if data_source in ["vstar", "vl_agent", "chart", "longvideo-reason", "hacs", "ego4d_naq", "longvt"]:
        from custom_rewards import vl_agent_test

        score, acc_score, format_reward_score, *extra = vl_agent_test.compute_score(
            solution_str, ground_truth, extra_info, **kwargs
        )

        score_dict["score"] = score
        score_dict["acc_score"] = acc_score
        score_dict["format_reward_score"] = format_reward_score

        if kwargs.get("tool_use_reward", False) and extra:
            score_dict["tool_reward_score"] = extra[0]
        elif (kwargs.get("use_time_reward", False) or kwargs.get("use_iou_reward", False)) and extra:
            score_dict["time_reward_score"] = extra[0]
    elif data_source in ["thinklite_eureka", "xince"]:
        from custom_rewards import vl_agent_test

        score, acc_score, format_reward_score = vl_agent_test.compute_score_math(solution_str, ground_truth, extra_info)
        score_dict["score"] = score
        score_dict["acc_score"] = acc_score
        score_dict["format_reward_score"] = format_reward_score
    elif data_source in ["longvt-val"]:
        from custom_rewards import vl_agent_test

        # val score
        score, acc_score, format_reward_score, *extra = vl_agent_test.compute_score(
            solution_str, ground_truth, extra_info, **kwargs
        )
        score_dict["score"] = acc_score
    elif data_source in ["videor1"]:
        from custom_rewards import vl_agent_test

        score, acc_score, format_reward_score = vl_agent_test.compute_score_videor1(
            solution_str, ground_truth, extra_info, **kwargs
        )
        score_dict["score"] = score
        score_dict["acc_score"] = acc_score
        score_dict["format_reward_score"] = format_reward_score

        # Add dummy values for optional reward fields to maintain batch consistency
        # Use -1.0 as a special dummy value (videor1 doesn't support these rewards)
        # This maintains tensor compatibility while clearly marking unsupported features
        if kwargs.get("tool_use_reward", False):
            score_dict["tool_reward_score"] = -1.0  # videor1 doesn't have tool rewards
        elif kwargs.get("use_time_reward", False) or kwargs.get("use_iou_reward", False):
            score_dict["time_reward_score"] = -1.0  # videor1 doesn't have time/iou rewards
    elif data_source in ["time-r1"]:
        from custom_rewards import vl_agent_test

        # read use_recall from kwargs, default to False
        use_recall = kwargs.get("use_recall", False)
        score, acc_score, format_reward_score = vl_agent_test.compute_score_time_r1(
            solution_str, ground_truth, extra_info, use_recall=use_recall
        )
        score_dict["score"] = score
        score_dict["acc_score"] = acc_score
        score_dict["format_reward_score"] = format_reward_score
    elif data_source in ["longvideo-reason-think"]:
        from custom_rewards import vl_agent_test

        score, acc_score, format_reward_score = vl_agent_test.compute_score_think(
            solution_str, ground_truth, extra_info
        )
        score_dict["score"] = score
        score_dict["acc_score"] = acc_score
        score_dict["format_reward_score"] = format_reward_score
    else:
        format_score = 0.1
        format_reward_score = format_reward(solution_str)
        if not isinstance(ground_truth, str):
            ground_truth = str(ground_truth)

        extracted_answer = extract_anwser_tag(solution_str).strip()
        predict_str = simple_parse(extracted_answer)
        gt = simple_parse(ground_truth)

        acc_score = relax_exact_match(predict_str, gt)
        score = (1.0 - format_score) * acc_score + format_score * format_reward_score
        score_dict["score"] = score
        score_dict["acc_score"] = acc_score
        score_dict["format_reward_score"] = format_reward_score

    score_dict["predict_str"] = solution_str
    score_dict["ground_truth"] = ground_truth

    return score_dict
