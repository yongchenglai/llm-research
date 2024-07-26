import argparse
import json
from typing import Dict
import logging

import torch
import transformers
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
IGNORE_TOKEN_ID = LabelSmoother.ignore_index



""" 
preprocess函数接收一个包含对话数据的json列表作为输入,\n
通过调用transformers库中的tokenizer对数据进行编码,\n
并按照特定格式构建输入ID序列和目标ID序列.\n
返回一个包含预处理数据的列表,这些数据已转换为PyTorch张量,适合于后续模型训练或推断
"""
def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    # roles字典：为对话中的角色("user"和"assistant")分配特殊的前缀标签，用于区分对话双方
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    # im_start和im_end：指定tokenizer中im_start_id和im_end_id对应的整数ID。
    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    # nl_tokens: 存储tokenizer处理换行符\n得到的输入ID序列。
    nl_tokens = tokenizer('\n').input_ids
    # _system、_user和_assistant: 分别存储经过tokenizer处理后的"system"、
    # "user"和"assistant"标签及其后的换行符对应的输入ID序列。
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    # 定义空列表data, 用于存放预处理后的数据样本
    data = []
    # input_ids, targets = [], []
    # 遍历输入数据sources中的每个样本(source)
    for i, source in enumerate(sources):
        source = source["conversations"]
        # 检查首个对话是否由用户发起(即source[0]["from"]是否为"user")，
        # 如果不是，则从源数据中移除首个对话。过滤无效的identity
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        # 初始化空列表input_id和target,分别用于存储当前样本的输入ID序列和目标ID序列
        input_id, target = [], []
        # #添加系统消息: 将系统消息(包含system_message内容)转换为ID序列，添加到input_id和target中。
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system

        # target中的非关键部分(如系统标签和消息内容)用IGNORE_TOKEN_ID填充。
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)

        # 遍历源数据中的每个对话(sentence)
        for j, sentence in enumerate(source):
            # 提取角色和消息内容，并转换为ID序列
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            # 添加到input_id中
            input_id += _input_id

            # 根据角色类型, 生成对应_target的目标ID序列,
            # _target只提取assistant的对话内容, 忽略user的对话内容。
            if role == '<|im_start|>user':
                # 若角色为"user", 则目标ID序列仅包含开始标签和结束标签, 用忽略ID填充对话内容。
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                # 若角色为"assistant"，则目标ID序列包含开始标签、
                # 忽略ID填充(仅对角色标签)、对话内容（不包括角色标签和结束标签）、
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target

        assert len(input_id) == len(target)
        # 截取并转换为张量：
        # 截取input_id和target至最大长度max_len
        input_id = torch.tensor(input_id[:max_len], dtype=torch.int)
        target = torch.tensor(target[:max_len], dtype=torch.int)
        # 创建一个字典，包含键input_ids(存储输入张量)和attention_mask
        # (等于输入张量,用于指示非填充位置)。将该字典添加到data列表中
        data.append(dict(input_ids=input_id,
                         attention_mask=input_id.ne(tokenizer.pad_token_id)))

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model Quantization using AutoGPTQ")
    parser.add_argument("--model_name_or_path", type=str, help="model path")
    parser.add_argument("--data_path", type=str, help="calibration data path")
    parser.add_argument("--out_path", type=str,
                        help="output path of the quantized model")
    parser.add_argument("--max_len", type=int, default=8192,
                        help="max length of calibration data")
    parser.add_argument("--bits", type=int, default=4,
                        help="the bits of quantized model. 4 indicates int4 models.")
    parser.add_argument("--group-size", type=int, default=128,
                        help="the group size of quantized model")
    args = parser.parse_args()
    
    quantize_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        damp_percent=0.01,
        # set to False can significantly speed up inference
        # but the perplexity may slightly bad
        desc_act=False,
        static_groups=False,
        sym=True,
        true_sequential=True,
        model_name_or_path=None,
        model_file_base_name="model"
    )

    # 使用AutoTokenizer类从给定路径args.model_name_or_path加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eod_id

    # 加载json数据文件, 调用process函数预处理数据, 返回处理后的数据
    data = preprocess(json.load(open(args.data_path)), tokenizer, args.max_len)

    model = AutoGPTQForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantize_config,
        device_map="auto",
        trust_remote_code=True)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 对模型进行量化,不在GPU上缓存示例数据
    model.quantize(data, cache_examples_on_gpu=False)

    # 保存量化后的模型
    model.save_quantized(args.out_path, use_safetensors=True)

    # 将tokenizer保存到输出路径
    tokenizer.save_pretrained(args.out_path)


