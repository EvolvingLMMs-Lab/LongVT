# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
import difflib
import logging
import os
from enum import Enum
from typing import Any, Optional

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema, ToolResponse
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

BASE_CHAT_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


class FinishReasonTypeEnum(str, Enum):
    """The enum for finish reason type."""

    LENGTH = "length"
    STOP = "stop"
    TOOL_CALL = "tool_calls"

    @classmethod
    def from_str(cls, value: str) -> "FinishReasonTypeEnum":
        if value == "stop":
            return cls.STOP
        elif value == "length":
            return cls.LENGTH
        elif value == "tool_calls":
            return cls.TOOL_CALL
        else:
            raise ValueError(f"Unsupported finish reason type: {value}")


class Message(BaseModel):
    role: str
    content: str | dict[str, Any] | list[dict[str, Any]] | ToolResponse
    tool_calls: Optional[list[OpenAIFunctionToolCall]] = None


class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"
    INTERACTING = "interacting"


class TokenizationSanityCheckModeEnum(str, Enum):
    """The enum for tokenization sanity check mode."""

    DISABLE = "disable"
    STRICT = "strict"
    IGNORE_STRIPPABLE = "ignore_strippable"


class AsyncRolloutRequest(BaseModel):
    """The data model for async rollout."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_data_id: int = 0
    rollout_offset: int = 0
    request_id: str
    state: AsyncRolloutRequestStateEnum
    messages: list[Message]
    multi_modal_keys: Optional[list[str]] = None
    multi_modal_data: Optional[dict[str, Any]] = None
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    tool_schemas: Optional[list[OpenAIFunctionToolSchema]] = None
    tools_kwargs: dict[str, Any] = {}
    interaction_kwargs: dict[str, Any] = {}
    input_ids: Optional[torch.Tensor] = None
    prompt_ids: Optional[torch.Tensor] = None
    response_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    prompt_attention_mask: Optional[torch.Tensor] = None
    response_attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    prompt_position_ids: Optional[torch.Tensor] = None
    response_position_ids: Optional[torch.Tensor] = None
    loss_mask: Optional[torch.Tensor] = None
    prompt_loss_mask: Optional[torch.Tensor] = None
    response_loss_mask: Optional[torch.Tensor] = None
    reward_scores: dict[str, float]
    max_prompt_len: int
    max_response_len: int = 8192
    max_model_len: int = 32768
    metrics: dict[str, list[Any]] = {}
    output_token_ids: torch.Tensor | None = None
    rollout_log_probs: torch.Tensor | None = None

    use_inference_chat_template: bool
    tokenization_sanity_check_mode: TokenizationSanityCheckModeEnum
    generation_prompt_ids: Optional[torch.Tensor] = None
    base_conv_wo_gen_prompt_end_pos: int
    base_conv_with_gen_prompt_end_pos: int

    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        # breakpoint()
        if not (messages := values.get("messages")):
            raise ValueError("messages is required for AsyncRolloutRequest initialization")
        if not (max_prompt_len := values.get("max_prompt_len")):
            raise ValueError("max_prompt_len is required for AsyncRolloutRequest initialization")
        if not (processing_class := values.pop("processing_class", None)):
            raise ValueError("processing_class is required for AsyncRolloutRequest initialization")

        values["messages"] = [Message.model_validate(msg) for msg in messages]

        # If there is no multi_modal_keys, we assume the multi-modal data is image and video.
        if not values.get("multi_modal_keys"):
            values["multi_modal_keys"] = ["image", "video"]
        if not values.get("multi_modal_data"):
            values["multi_modal_data"] = {key: [] for key in values["multi_modal_keys"]}
        else:
            # check if all multi_modal_keys are in multi_modal_data
            for key in values["multi_modal_keys"]:
                if key not in values["multi_modal_data"]:
                    values["multi_modal_data"][key] = []
        if not values.get("multi_modal_inputs"):
            values["multi_modal_inputs"] = {}

        tools = (
            [tool.model_dump() for tool in tool_schemas] if (tool_schemas := values.get("tool_schemas", [])) else None
        )

        multi_modal_data = values["multi_modal_data"]
        tokens_without_prompt = cls._handle_apply_chat_template(
            processing_class,
            messages,
            multi_modal_data=multi_modal_data,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
        )
        if (
            values.get("input_ids") is None
            or values.get("attention_mask") is None
            or values.get("position_ids") is None
        ):
            tokenization_dict_with_prompt = cls._handle_apply_chat_template(
                processing_class,
                messages,
                multi_modal_data=multi_modal_data,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
            )

            values["input_ids"], values["attention_mask"] = (
                tokenization_dict_with_prompt["input_ids"],
                tokenization_dict_with_prompt["attention_mask"],
            )
            if values["input_ids"].shape[-1] > max_prompt_len:
                # Only log the warning to avoid truncating in the middle of generation prompt. Consider raising an
                # error for this case in the future.
                # Ensure batch_data_id exists with default value if not provided
                if "batch_data_id" not in values:
                    values["batch_data_id"] = cls.model_fields["batch_data_id"].default
                logger.warning(
                    f"Prompt {values['batch_data_id']} has length {values['input_ids'].shape[-1]} "
                    f"which is greater than max_prompt_len {max_prompt_len} after applied chat template with tools."
                )

            # Process multi_modal_inputs
            multi_modal_inputs = tokenization_dict_with_prompt.copy()
            multi_modal_inputs.pop("input_ids", None)
            multi_modal_inputs.pop("attention_mask", None)
            values["multi_modal_inputs"] = multi_modal_inputs

            values["position_ids"] = values["prompt_position_ids"] = cls._get_position_ids(
                processing_class, values["input_ids"], values["attention_mask"], multi_modal_inputs
            )

        values["prompt_ids"], values["prompt_attention_mask"] = values["input_ids"], values["attention_mask"]
        values["loss_mask"] = values["prompt_loss_mask"] = torch.zeros_like(values["input_ids"], dtype=torch.bool)
        values["generation_prompt_ids"] = values["input_ids"][..., tokens_without_prompt.shape[-1] :]
        values["base_conv_wo_gen_prompt_end_pos"] = cls._handle_apply_chat_template(
            processing_class,
            BASE_CHAT_HISTORY,
            multi_modal_data=multi_modal_data,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
        ).shape[-1]

        values["base_conv_with_gen_prompt_end_pos"] = cls._handle_apply_chat_template(
            processing_class,
            BASE_CHAT_HISTORY,
            multi_modal_data=multi_modal_data,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        ).shape[-1]

        return values

    @staticmethod
    def _handle_apply_chat_template(
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        messages: list[Message],
        multi_modal_data: dict[str, Any],
        tools: Optional[list[OpenAIFunctionToolSchema]] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        return_dict: bool = False,
    ):
        messages_dicts = [msg.model_dump() if hasattr(msg, "model_dump") else dict(msg) for msg in messages]

        messages_for_processing = messages_dicts
        supports_video_tokens = False
        has_video_offsets = bool(multi_modal_data.get("video_image_offsets")) if multi_modal_data else False
        if isinstance(processing_class, ProcessorMixin):
            supports_video_tokens = AsyncRolloutRequest._processor_supports_video(processing_class)
            video_offsets = multi_modal_data.get("video_image_offsets") if multi_modal_data else None
            if (
                supports_video_tokens
                and video_offsets
                and multi_modal_data
                and len(multi_modal_data.get("video", [])) > 0
            ):
                messages_for_processing = AsyncRolloutRequest._collapse_video_placeholders(
                    messages_dicts, video_offsets
                )

        raw_prompt = processing_class.apply_chat_template(
            messages_for_processing, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False
        )
        if not tokenize:
            return raw_prompt
        # breakpoint()
        video_token_id = getattr(processing_class, "video_token_id", None)
        image_token_id = getattr(processing_class, "image_token_id", None)

        if isinstance(processing_class, PreTrainedTokenizer) or isinstance(processing_class, PreTrainedTokenizerFast):
            if any(len(values) > 0 for values in multi_modal_data.values()):
                logger.warning(
                    "There is multi_modal_data but you are not using a processor. Multi-modal data will be ignored."
                )
            model_inputs = processing_class(text=[raw_prompt], return_tensors="pt")
        elif isinstance(processing_class, ProcessorMixin):
            # When we update multi_model_keys, we also need to update this logic
            images_list = multi_modal_data.get("image", [])
            videos_list = multi_modal_data.get("video", [])

            videos = None
            if supports_video_tokens and len(videos_list) > 0 and not has_video_offsets:
                videos = videos_list

            images = images_list if len(images_list) > 0 else None

            processor_kwargs = {"text": [raw_prompt], "return_tensors": "pt"}
            if images is not None:
                processor_kwargs["images"] = images
            if videos is not None:
                processor_kwargs["videos"] = videos

            model_inputs = processing_class(**processor_kwargs)

            if videos is not None:
                pixel_values_videos = model_inputs.pop("pixel_values_videos", model_inputs.pop("pixel_values", None))
                video_grid_thw = model_inputs.pop("video_grid_thw", model_inputs.get("image_grid_thw"))

                # 清理掉会触发图像分支的不匹配键
                model_inputs.pop("pixel_values", None)
                if pixel_values_videos is not None:
                    model_inputs["pixel_values_videos"] = pixel_values_videos
                if video_grid_thw is not None:
                    model_inputs["video_grid_thw"] = video_grid_thw
            else:
                # 纯图像场景保留原逻辑
                model_inputs.pop("video_grid_thw", None)
                model_inputs.pop("pixel_values_videos", None)

            if (
                supports_video_tokens
                and video_token_id is not None
                and image_token_id is not None
                and "input_ids" in model_inputs
            ):
                # rollout 阶段需要回退到真实视频 token，避免 features mismatch
                input_ids_tensor = model_inputs["input_ids"].clone()
                # 仅当当前 input_ids 中存在 video token 时才替换，保持图像样本不受影响
                if (input_ids_tensor == image_token_id).any() and videos is not None:
                    input_ids_tensor[input_ids_tensor == image_token_id] = video_token_id
                model_inputs["input_ids"] = input_ids_tensor
        else:
            raise ValueError(f"Unsupported processing class type: {type(processing_class)}")

        model_inputs = dict(model_inputs)
        if return_dict:
            return model_inputs
        else:
            input_ids_tensor = model_inputs["input_ids"]
            if (
                (not supports_video_tokens or has_video_offsets)
                and videos_list
                and video_token_id is not None
                and image_token_id is not None
            ):
                input_ids_tensor = input_ids_tensor.clone()
                input_ids_tensor[input_ids_tensor == video_token_id] = image_token_id
            return input_ids_tensor

    @staticmethod
    def _get_position_ids(
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        processor_cls_name = processing_class.__class__.__name__ if hasattr(processing_class, "__class__") else ""
        is_qwen3vl = "Qwen3VLProcessor" in processor_cls_name
        is_qwen2vl = (
            hasattr(processing_class, "image_processor")
            and "Qwen2VLImageProcessor" in processing_class.image_processor.__class__.__name__
        )
        if is_qwen3vl or is_qwen2vl:
            if is_qwen3vl:
                from verl.models.transformers.qwen3_vl import get_rope_index as get_rope_index_fn
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index as get_rope_index_fn

            image_grid_thw = video_grid_thw = second_per_grid_ts = None
            if multi_modal_inputs:
                image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

            assert input_ids.dim() == 2 and input_ids.shape[0] == 1, (
                f"input_ids should be 2D with batch size 1, but got shape {input_ids.shape}"
            )
            assert attention_mask.dim() == 2 and attention_mask.shape[0] == 1, (
                f"attention_mask should be 2D with batch size 1, but got shape {attention_mask.shape}"
            )
            new_position_ids = get_rope_index_fn(
                processing_class,
                input_ids=input_ids.squeeze(0),
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask.squeeze(0),
            )
            return new_position_ids  # (3, seq_len)
        else:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

    @staticmethod
    def _collapse_video_placeholders(
        messages: list[dict[str, Any]],
        video_offsets: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        """Collapse consecutive image placeholders back into video placeholders based on offsets."""

        if not video_offsets:
            return messages

        collapsed_messages: list[dict[str, Any]] = []
        image_cursor = 0
        video_idx = 0

        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                collapsed_messages.append(message)
                continue

            new_content = []
            idx = 0
            while idx < len(content):
                entry = content[idx]
                if isinstance(entry, dict) and entry.get("type") == "image" and video_idx < len(video_offsets):
                    start, length = video_offsets[video_idx]
                    if image_cursor == start:
                        new_content.append({"type": "video"})
                        image_cursor += length
                        idx += length
                        video_idx += 1
                        continue
                if isinstance(entry, dict) and entry.get("type") == "image":
                    image_cursor += 1
                new_content.append(entry)
                idx += 1

            collapsed_messages.append({**message, "content": new_content})

        return collapsed_messages

    @staticmethod
    def _processor_supports_video(processing_class: ProcessorMixin) -> bool:
        if not isinstance(processing_class, ProcessorMixin):
            return False

        return "Qwen3VLProcessor" in processing_class.__class__.__name__

    def _update_input_ids(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        new_input_ids: torch.Tensor,
        attention_mask: bool,
        loss_mask: bool,
        new_multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Update the input_ids, attention_mask, position_ids, and loss_mask of the request in additive manner.
        """
        self.input_ids = torch.cat([self.input_ids, new_input_ids], dim=-1)
        attention_mask = torch.ones_like(new_input_ids) * int(attention_mask)
        self.attention_mask = torch.cat([self.attention_mask, attention_mask], dim=-1)
        loss_mask = torch.ones_like(new_input_ids) * int(loss_mask)
        self.loss_mask = torch.cat([self.loss_mask, loss_mask], dim=-1)

        if new_multi_modal_inputs:
            self._update_multi_modal_inputs(new_multi_modal_inputs)

        new_position_ids = self._get_position_ids(
            processing_class, new_input_ids, attention_mask, new_multi_modal_inputs
        )

        last_pos = self.position_ids[..., -1:]
        new_position_ids = new_position_ids + (last_pos + 1)

        self.position_ids = torch.cat([self.position_ids, new_position_ids], dim=-1)

        assert (
            self.input_ids.shape[-1]
            == self.attention_mask.shape[-1]
            == self.position_ids.shape[-1]
            == self.loss_mask.shape[-1]
        ), f"""Request {self.request_id} has different length of {self.input_ids.shape[-1]=}, 
            {self.attention_mask.shape[-1]=}, {self.position_ids.shape[-1]=}, {self.loss_mask.shape[-1]=}"""

    def _update_multi_modal_inputs(self, new_multi_modal_inputs: dict[str, torch.Tensor]) -> None:
        """
        Update the multi_modal_inputs of the request in additive manner.
        """
        for key in new_multi_modal_inputs:
            input_tensor = new_multi_modal_inputs[key]
            self.multi_modal_inputs[key] = (
                torch.cat([self.multi_modal_inputs[key], input_tensor], dim=0)
                if key in self.multi_modal_inputs
                else input_tensor
            )

    def get_generation_prompt_ids(
        self, processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin
    ) -> list[int]:
        """
        Get the generation prompt ids for rollout engine.

        Because rollout engine(SGLang) requires the ids to be a list, we need to convert the tensor to a list.
        """
        generation_prompt_ids = (
            None
            if self.input_ids[..., -self.generation_prompt_ids.shape[-1] :].eq(self.generation_prompt_ids).all()
            else self.generation_prompt_ids
        )
        if generation_prompt_ids is not None:
            self._update_input_ids(processing_class, generation_prompt_ids, attention_mask=True, loss_mask=False)

        if self.use_inference_chat_template:
            messages = [msg.model_dump() for msg in self.messages]
            tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
            generation_prompt_ids = self._handle_apply_chat_template(
                processing_class,
                messages,
                multi_modal_data=self.multi_modal_data,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
            )
            return generation_prompt_ids.squeeze(0).tolist()
        else:
            return self.input_ids.squeeze(0).tolist()

    def add_user_message(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        content: str,
    ) -> None:
        self.messages.append(Message(role="user", content=content))
        messages = [*BASE_CHAT_HISTORY, self.messages[-1]]
        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None

        # We don't need to pass multi_modal_data here because we don't have any multi-modal data from Engine
        # Inference, it is pure text.
        content_ids = self._handle_apply_chat_template(
            processing_class, messages, multi_modal_data={}, tools=tools, add_generation_prompt=False, tokenize=True
        )[..., self.base_conv_wo_gen_prompt_end_pos :]
        self._update_input_ids(processing_class, content_ids, attention_mask=True, loss_mask=False)

    def add_assistant_message(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        content: str,
        content_ids: Optional[torch.Tensor] = None,
        tool_calls: Optional[list[OpenAIFunctionToolCall]] = None,
    ) -> None:
        self.messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))
        if content_ids is None:
            messages = [*BASE_CHAT_HISTORY, self.messages[-1]]
            tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None

            # We don't need to pass multi_modal_data here because we don't have any multi-modal data from Engine
            # Inference, it is pure text.
            content_ids = self._handle_apply_chat_template(
                processing_class, messages, multi_modal_data={}, tools=tools, add_generation_prompt=False, tokenize=True
            )[..., self.base_conv_with_gen_prompt_end_pos :]
        self._update_input_ids(processing_class, content_ids, attention_mask=True, loss_mask=True)

    def add_tool_response_messages(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        contents: list[str | dict[str, Any]],
        max_model_len: Optional[int] = None,  # 新增参数
    ) -> bool:  # 返回是否成功添加
        """
        Returns:
            bool: True if messages were added successfully, False if truncated due to length limit
        """

        if not contents:
            return True

        # 记录添加前的状态，用于可能的回滚
        original_input_ids = self.input_ids.clone() if self.input_ids is not None else None
        original_attention_mask = self.attention_mask.clone() if self.attention_mask is not None else None
        original_position_ids = self.position_ids.clone() if self.position_ids is not None else None
        original_loss_mask = self.loss_mask.clone() if self.loss_mask is not None else None
        original_multi_modal_data = (
            {key: val.copy() for key, val in self.multi_modal_data.items()} if self.multi_modal_data else {}
        )
        original_multi_modal_inputs = (
            {key: val.clone() if isinstance(val, torch.Tensor) else val for key, val in self.multi_modal_inputs.items()}
            if self.multi_modal_inputs
            else {}
        )
        original_messages_count = len(self.messages)

        # We also handle the case when tool returns image
        # We require the processing of the image and video to be done at tool.execute() level
        delta_multi_modal_data = {key: [] for key in self.multi_modal_keys}
        added_images_count = 0  # 记录新增的图像数量

        for content in contents:
            if content.is_text_only():
                self.messages.append(Message(role="tool", content=content.text))
            else:
                content_list = []
                # When we update multi_model_keys, we also need to update this logic
                if content.image:
                    content_list.extend([{"type": "image"} for _ in content.image])
                    delta_multi_modal_data["image"].extend(content.image)
                    added_images_count += len(content.image)
                if content.video:
                    content_list.extend([{"type": "video"} for _ in content.video])
                    delta_multi_modal_data["video"].extend(content.video)
                if content.text:
                    content_list.append({"type": "text", "text": content.text})
                self.messages.append(Message(role="tool", content=content_list))

        messages = [*BASE_CHAT_HISTORY, *self.messages[-len(contents) :]]
        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None

        multi_modal_updated = False
        for key in self.multi_modal_keys:
            if len(delta_multi_modal_data[key]) > 0:
                if key not in self.multi_modal_data:
                    self.multi_modal_data[key] = []
                self.multi_modal_data[key].extend(delta_multi_modal_data[key])
                multi_modal_updated = True

        # We just passed the new multi-modal data to the chat template to update the input_ids.
        content_info = self._handle_apply_chat_template(
            processing_class,
            messages,
            multi_modal_data=delta_multi_modal_data,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
        )
        content_ids = content_info["input_ids"][..., self.base_conv_wo_gen_prompt_end_pos :]

        # chat templates include generation prompt tokens (e.g., "<im_start>assistant\n")
        # So when tool response is added, we need to explicitly remove these tokens.
        self._remove_generation_prompt_ids_if_present()

        self._update_input_ids(
            processing_class,
            content_ids,
            attention_mask=True,
            loss_mask=False,
            new_multi_modal_inputs=None,
        )

        if multi_modal_updated:
            self._rebuild_multi_modal_inputs(processing_class)

        should_rollback = False

        # check if the response will be truncated
        if max_model_len and self.input_ids is not None and self.input_ids.shape[-1] > max_model_len:
            logger.warning(
                f"Adding tool response would exceed max_model_len ({self.input_ids.shape[-1]} > {max_model_len}). "
                f"Added {added_images_count} images. Attempting to truncate..."
            )
            should_rollback = True

        # check if there are images and the response will be truncated
        if (
            added_images_count > 0
            and hasattr(self, "max_response_len")
            and self.max_response_len
            and self.prompt_ids is not None
        ):
            current_response_len = self.input_ids.shape[-1] - self.prompt_ids.shape[-1]

            if current_response_len > self.max_response_len:
                logger.warning(
                    f"Added {added_images_count} images but response would be truncated "
                    f"(current: {current_response_len}, max: {self.max_response_len}). "
                    f"Rolling back to prevent image token corruption."
                )
                should_rollback = True

            # rollback
        if should_rollback:
            logger.warning("Rolling back all changes")
            self._rollback_to_state(
                original_input_ids,
                original_attention_mask,
                original_position_ids,
                original_loss_mask,
                original_multi_modal_data,
                original_multi_modal_inputs,
                original_messages_count,
            )
            return False

        return True

    def _rebuild_multi_modal_inputs(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
    ) -> None:
        if not self.multi_modal_data:
            return

        messages = [*BASE_CHAT_HISTORY, *self.messages]
        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None

        model_inputs = self._handle_apply_chat_template(
            processing_class,
            messages,
            multi_modal_data=self.multi_modal_data,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
        )

        model_inputs = dict(model_inputs)
        model_inputs.pop("input_ids", None)
        model_inputs.pop("attention_mask", None)

        processed_inputs: dict[str, torch.Tensor | Any] = {}
        for key, value in model_inputs.items():
            if isinstance(value, torch.Tensor):
                processed_inputs[key] = value.clone()
            else:
                processed_inputs[key] = value

        self.multi_modal_inputs = processed_inputs

    def _rollback_to_state(
        self,
        original_input_ids,
        original_attention_mask,
        original_position_ids,
        original_loss_mask,
        original_multi_modal_data,
        original_multi_modal_inputs,
        original_messages_count,
    ):
        """回滚到添加tool response之前的状态"""
        self.input_ids = original_input_ids
        self.attention_mask = original_attention_mask
        self.position_ids = original_position_ids
        self.loss_mask = original_loss_mask
        self.multi_modal_data = original_multi_modal_data
        self.multi_modal_inputs = original_multi_modal_inputs

        # 回滚messages
        while len(self.messages) > original_messages_count:
            self.messages.pop()

    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        """
        metrics: should be a dict of tools_name -> Any
        """
        if self.metrics.get(tool_id) is None:
            self.metrics[tool_id] = []
        self.metrics[tool_id].append(metrics)

    def _get_prompt_diffs(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        full_prompt_ids: torch.Tensor,
        current_prompt_ids: torch.Tensor,
        diff_surrounding_chars: int = 10,
    ) -> list[dict[str, Any]]:
        """Get differences between full prompt and current prompt with surrounding context.

        This function helps debug tokenization mismatches by showing the differences between
        full prompt and current prompt with surrounding context. Instead of just showing
        the exact diff, it includes additional tokens before and after to help locate
        the issue in the chat template.

        For example, if the actual diff is a newline change from "\n\n" to "\n", with
        diff_surrounding_chars the output might look like:

        full_prompt_chunk:    "<|im_start|>assistant\n\nI think..."
        current_prompt_chunk: "<|im_start|>assistant\nI think..."

        This context makes it much easier to identify where in the chat template the
        mismatch occurs.

        Args:
            processing_class: The processing class to use for decoding the token IDs
            full_prompt_ids: Token IDs from applying chat template to all messages at once
            current_prompt_ids: Token IDs from incremental chat template application
            diff_surrounding_chars: Number of surrounding characters to include for context (default: 10)

        Returns:
            List of dicts containing the differing chunks with context and their indices
        """
        full_prompt_ids = full_prompt_ids.squeeze(0)
        current_prompt_ids = current_prompt_ids.squeeze(0)
        full_prompt = processing_class.decode(full_prompt_ids, skip_special_tokens=False)
        current_prompt = processing_class.decode(current_prompt_ids, skip_special_tokens=False)
        s = difflib.SequenceMatcher(None, full_prompt, current_prompt, autojunk=False)
        diffs = []
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag == "equal":
                continue

            # Get the surrounding context for better readability
            start_i = max(0, i1 - diff_surrounding_chars)
            end_i = min(len(full_prompt), i2 + diff_surrounding_chars)
            start_j = max(0, j1 - diff_surrounding_chars)
            end_j = min(len(current_prompt), j2 + diff_surrounding_chars)

            diffs.append(
                {
                    "full_prompt_chunk": full_prompt[start_i:end_i],
                    "current_prompt_chunk": current_prompt[start_j:end_j],
                    "indices": (start_i, end_i, start_j, end_j),
                }
            )
        return diffs

    def _remove_generation_prompt_ids_if_present(self) -> None:
        """
        Remove generation prompt IDs from input tensors if they are present at the end.
        """
        if self.input_ids[..., -self.generation_prompt_ids.shape[-1] :].eq(self.generation_prompt_ids).all():
            self.input_ids = self.input_ids[..., : -self.generation_prompt_ids.shape[-1]]
            self.attention_mask = self.attention_mask[..., : -self.generation_prompt_ids.shape[-1]]
            self.position_ids = self.position_ids[..., : -self.generation_prompt_ids.shape[-1]]
            self.loss_mask = self.loss_mask[..., : -self.generation_prompt_ids.shape[-1]]

    def finalize(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        reward_scores: dict[str, list[float]],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores

        # In case we failed to generate the assistant message and the generation prompt ids were already added to
        # input_ids, remove them from the end of input_ids
        self._remove_generation_prompt_ids_if_present()

        self.response_ids = self.input_ids[..., self.prompt_ids.shape[-1] :]

        # self.tokenization_sanity_check_mode = TokenizationSanityCheckModeEnum.DISABLE # bug
        # breakpoint()
        if self.tokenization_sanity_check_mode != TokenizationSanityCheckModeEnum.DISABLE:
            # When there is a diff, we log the diffs with diff_surrounding_chars context
            diff_surrounding_chars = 10

            messages = [msg.model_dump() for msg in self.messages]
            tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
            full_prompt_info = self._handle_apply_chat_template(
                processing_class,
                messages,
                multi_modal_data=self.multi_modal_data,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
            )
            full_prompt_ids = full_prompt_info["input_ids"]

            # We must use dict(full_prompt_info) to convert BatchFeature values to a new dict
            # because np.array() only keeps the keys for BatchFeature.
            full_prompt_multi_modal_inputs = full_prompt_info.copy()
            full_prompt_multi_modal_inputs.pop("input_ids", None)
            full_prompt_multi_modal_inputs.pop("attention_mask", None)

            for multi_modal_inputs_key in self.multi_modal_inputs:
                if multi_modal_inputs_key in full_prompt_multi_modal_inputs:
                    if (
                        not self.multi_modal_inputs[multi_modal_inputs_key]
                        .eq(full_prompt_multi_modal_inputs[multi_modal_inputs_key])
                        .all()
                    ):
                        logger.warning(
                            f"Multi-modal data {multi_modal_inputs_key} is not consistent. "
                            f"This may lead to unexpected behavior during training. "
                            f"Please review your multi_modal_inputs logic."
                        )
                else:
                    logger.warning(
                        f"Multi-modal inputs key {multi_modal_inputs_key} is not found in the multi_modal_inputs. "
                        f"This may lead to unexpected behavior during training."
                        f"Please review your multi_modal_inputs logic."
                    )

            if diffs := self._get_prompt_diffs(
                processing_class, full_prompt_ids, self.input_ids, diff_surrounding_chars=diff_surrounding_chars
            ):
                log_warning = False
                if self.tokenization_sanity_check_mode == TokenizationSanityCheckModeEnum.STRICT:
                    log_warning = True
                elif self.tokenization_sanity_check_mode == TokenizationSanityCheckModeEnum.IGNORE_STRIPPABLE:
                    non_strippable_diffs_exist = any(
                        d["full_prompt_chunk"].strip() or d["current_prompt_chunk"].strip() for d in diffs
                    )
                    if non_strippable_diffs_exist:
                        log_warning = True

                if log_warning:
                    mode_str = f" ({self.tokenization_sanity_check_mode.value})"
                    logger.warning(
                        f"Inconsistent training and inference tokenization detected{mode_str}. This may lead to "
                        f"unexpected behavior during training. Please review your chat template to determine if this "
                        f"is intentional. For more information, refer to the multiturn README.md."
                    )
                    logger.warning(
                        f"Showing {diff_surrounding_chars} characters before and after the diffs for context and "
                        f"better readability."
                    )
                    diff_details_list = []
                    for d in diffs:
                        i1, i2, j1, j2 = d["indices"]
                        diff_details_list.append(
                            f"idx {i1}:{i2} -> {j1}:{j2} | full_prompt_chunk: {repr(d['full_prompt_chunk'])} | "
                            f"current_prompt_chunk: {repr(d['current_prompt_chunk'])}"
                        )
                    diff_details = "\n".join(diff_details_list)
                    logger.warning(f"Found differences:\n{diff_details}")

        if finish_reason_type == FinishReasonTypeEnum.STOP:
            pass
        elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
            pass
        else:
            raise ValueError(f"Unsupported finalize finish reason type: {finish_reason_type}")
        self.truncate_output_ids(processing_class)

        assert (
            self.input_ids.shape[-1]
            == self.attention_mask.shape[-1]
            == self.position_ids.shape[-1]
            == self.loss_mask.shape[-1]
        ), f"""Request {self.request_id} has different length of {self.input_ids.shape[-1]=}, 
            {self.attention_mask.shape[-1]=}, {self.position_ids.shape[-1]=}, {self.loss_mask.shape[-1]=}"""

    def truncate_output_ids(
        self, processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin
    ) -> None:
        self.input_ids = self.input_ids[..., : self.max_model_len]
        self.attention_mask = self.attention_mask[..., : self.max_model_len]
        self.position_ids = self.position_ids[..., : self.max_model_len]
        self.loss_mask = self.loss_mask[..., : self.max_model_len]
        self.response_ids = self.input_ids[..., self.prompt_ids.shape[-1] :][..., : self.max_response_len]
        self.response_attention_mask = self.attention_mask[..., self.prompt_attention_mask.shape[-1] :][
            ..., : self.max_response_len
        ]
        self.response_position_ids = self.position_ids[..., self.prompt_position_ids.shape[-1] :][
            ..., : self.max_response_len
        ]
        self.response_loss_mask = self.loss_mask[..., self.prompt_loss_mask.shape[-1] :][..., : self.max_response_len]
