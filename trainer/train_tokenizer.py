# 注：不建议再重复训练tokenizer（“词典”），MiniMind已自带，此脚本仅供学习和参考。基于不同词典训练的模型将导致输出完全不统一，降低社区的模型复用性
import os
import json
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

# 训练数据来自预训练语料，每行是一个 JSON 对象，脚本默认读取其中的 text 字段。
DATA_PATH = '../dataset/pretrain_hq.jsonl'
# 为了避免覆盖项目内置 tokenizer，这里把学习版 tokenizer 输出到单独目录。
TOKENIZER_DIR = '../model_learn_tokenizer/'
# 词表大小需要和模型配置匹配；MiniMind 默认 vocab_size 也是 6400。
VOCAB_SIZE = 6400


def get_texts(data_path):
    """逐行读取 jsonl 语料，并用 yield 流式返回文本，避免一次性把数据全部读进内存。"""
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # 实验性限制：只用前 10000 行能快速验证流程；正式训练可删除或调大这个限制。
            if i >= 10000: break
            data = json.loads(line)
            yield data['text']


def train_tokenizer(data_path, tokenizer_dir, vocab_size):
    # BPE（Byte Pair Encoding）会从基础符号开始，反复合并语料中最常见的相邻符号对，
    # 最终得到一个固定大小的词表。词表越大，文本越容易被压缩成更少 token，但模型 embedding 参数也越多。
    tokenizer = Tokenizer(models.BPE())

    # ByteLevel 预分词先把文本映射到字节级表示。
    # 好处是几乎不会遇到真正无法表示的字符；缺点是小词表下可能把中文或生僻词切得比较碎。
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # BpeTrainer 负责统计语料中的 token 频率和合并规则。
    # special_tokens 会被优先放入词表，且顺序决定它们的 token id。
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        show_progress=True,
        # ByteLevel 的初始 alphabet 覆盖所有字节字符，保证任意 UTF-8 文本都能编码。
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    texts = get_texts(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 编码时使用 ByteLevel，解码也要使用对应的 ByteLevel decoder，才能还原空格和多字节字符。
    tokenizer.decoder = decoders.ByteLevel()

    # 固定特殊 token id，后续模型训练、推理和 chat_template 都会依赖这些 id 的含义。
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    os.makedirs(tokenizer_dir, exist_ok=True)
    # tokenizer.json 是 HuggingFace fast tokenizer 的核心文件，包含 vocab、merges、预分词器和解码器配置。
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    # 额外导出 vocab.json / merges.txt，便于单独查看 BPE 词表和合并规则。
    tokenizer.model.save(tokenizer_dir)

    # tokenizer_config.json 告诉 transformers 如何加载这个 tokenizer，以及聊天消息如何拼成模型输入。
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        # chat_template 使用 Jinja 模板，把多轮 messages 格式化为：
        # <|im_start|>role\ncontent<|im_end|>\n 这样的训练/推理文本。
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0]['role'] == 'system' -%}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else -%}\n        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n   {{- '<|im_start|>' + message.role + '\\n' + content }}\n  {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print("Tokenizer training completed.")


def eval_tokenizer(tokenizer_dir):
    # 用 transformers 加载刚训练出的 tokenizer，验证保存格式是否兼容 AutoTokenizer。
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print('-'*100)
    print(new_prompt)


    print('-'*100)
    print('tokenizer词表长度：', len(tokenizer))
    # 编码：自然语言字符串 -> input_ids，即模型真正看到的整数序列。
    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))
    # 解码：input_ids -> 字符串。这里检查编码再解码后是否能完整还原 prompt。
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
    print('decoder一致性：', response == new_prompt, "\n")


    print('-'*100)
    print('流式解码（字节缓冲）测试：')
    input_ids = model_inputs['input_ids']
    token_cache = []
    for tid in input_ids:
        token_cache.append(tid)
        # ByteLevel tokenizer 可能把一个 UTF-8 字符拆到多个 token。
        # 如果当前缓存还不能解出合法字符，decode 会出现 �，此时继续攒下一个 token。
        current_decode = tokenizer.decode(token_cache)
        if current_decode and '\ufffd' not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in (token_cache if isinstance(token_cache, list) else [token_cache])]
            print(f'Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')
            token_cache = []

if __name__ == '__main__':
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
    eval_tokenizer(TOKENIZER_DIR)
