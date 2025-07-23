# Copyright 2025 Individual Contributor: Sudong Wang
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

from datasets import Dataset, Features, Image, Sequence


def make_map_fn():
    def fn(example):
        # 1) 过滤掉 system 消息
        example["prompt"] = [m for m in example["prompt"] if m.get("role") != "system"]
        # 2) 删除 env_name（如果存在）
        example.pop("env_name", None)

        # 3) 先弹出原 images（彻底删除包含 path 的那一整列）
        # original_images = example.pop("images", [])

        # 4) 只保留 bytes，构造一个新的列表
        # images_bytes = [img_dict["bytes"] for img_dict in original_images]
        # example["images"] = [{"bytes": img["bytes"]} for img in original_images]

        # 5) 更新 extra_info 中的工具调用参数
        extra = example["extra_info"]
        extra["need_tools_kwargs"] = True
        extra["tools_kwargs"] = {"image_zoom_in_tool": {"execute_kwargs": {"dummy": ""}}}

        # extra["tools_kwargs"] = {"_dummy": ""}   ##math

        example["extra_info"] = extra

        return example

    return fn


if __name__ == "__main__":
    src_parquet = "/pfs/training-data/zuhaoyang/data/train/DeepEyes-47k/data_v0.8_visual_toolbox_v2.parquet"
    out_dir = "/pfs/training-data/sudongwang/dataset/deepeyes_process/data_v0.8_visual_toolbox_v2_processed.parquet"

    # 加载
    # ds = load_dataset("parquet", data_files=src_parquet, split="train")
    ds = Dataset.from_parquet(src_parquet)
    features = ds.features
    features.pop("images")
    new_features = Features({"images": Sequence(Image()), **features})
    ds = ds.cast(new_features, num_proc=16)
    # 转换
    ds = ds.map(make_map_fn(), num_proc=16)
    # features = ds.features
    # extra_info_feature = features.pop("extra_info")
    # extra_info_feature.pop("tools_kwargs", None)
    # extra_info_feature = {
    #     "tools_kwargs": {
    #         "image_zoom_in_tool": {}
    #     },
    #     **extra_info_feature
    # }
    # new_features = Features(
    #     {
    #         "extra_info": extra_info_feature,
    #         **features
    #     }
    # )
    # ds = ds.cast(new_features, num_proc=16)

    # import pdb; pdb.set_trace()
    # 保存

    ds.to_parquet(out_dir)
