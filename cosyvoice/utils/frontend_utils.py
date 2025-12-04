# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import regex
chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')

# 英文单词边界检测的正则表达式
word_boundary_pattern = re.compile(r'\b')


def find_english_word_boundary(text: str, start_pos: int, end_pos: int) -> int:
    """
    在指定范围内找到最佳的英文单词边界位置
    返回最接近end_pos但不截断单词的位置
    """
    # 首先尝试在空格、标点符号等明显分隔符处分割
    for i in range(end_pos, start_pos, -1):
        if text[i] in [' ', '\t', '\n', '.', ',', ';', ':', '!', '?', '-', '_']:
            return i + 1
    
    # 如果没有找到明显分隔符，尝试在单词边界分割
    # 寻找字母到非字母的转换点
    for i in range(end_pos, start_pos, -1):
        if i > 0 and i < len(text) - 1:
            # 当前字符是字母，下一个字符不是字母
            if text[i].isalpha() and not text[i + 1].isalpha():
                return i + 1
            # 当前字符不是字母，下一个字符是字母
            elif not text[i].isalpha() and text[i + 1].isalpha():
                return i + 1
    
    # 如果都找不到合适的边界，返回原始位置
    return end_pos


# whether contain chinese character
def contains_chinese(text):
    return bool(chinese_char_pattern.search(text))


# replace special symbol
def replace_corner_mark(text):
    text = text.replace('²', '平方')
    text = text.replace('³', '立方')
    return text


# remove meaningless symbol
def remove_bracket(text):
    text = text.replace('（', '').replace('）', '')
    text = text.replace('【', '').replace('】', '')
    text = text.replace('`', '').replace('`', '')
    text = text.replace("——", " ")
    return text


# spell Arabic numerals
def spell_out_number(text: str, inflect_parser):
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st: i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return ''.join(new_text)


# split paragrah logic：
# 1. per sentence max len token_max_n, min len token_min_n, merge if last sentence len less than merge_len
# 2. cal sentence len according to lang
# 3. split sentence according to puncatation
def split_paragraph(text: str, tokenize, lang="zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False):
    def calc_utt_length(_text: str):
        if lang == "zh":
            return len(_text)
        else:
            return len(tokenize(_text))

    def should_merge(_text: str):
        if lang == "zh":
            return len(_text) < merge_len
        else:
            return len(tokenize(_text)) < merge_len

    # 处理空文本
    if not text or not text.strip():
        return []

    # 参数验证
    if token_min_n > token_max_n:
        token_min_n, token_max_n = token_max_n, token_min_n

    if lang == "zh":
        pounc = ['。', '？', '！', '；', '：', '、', '.', '?', '!', ';']
        default_split_punc = '。'
    else:
        pounc = ['.', '?', '!', ';', ':']
        default_split_punc = '.'
    if comma_split:
        pounc.extend(['，', ','])

    def force_split_segment(segment: str):
        """强制切分单条过长的语句，避免超出 token_max_n。"""
        if not segment:
            return []
        tail_punc = segment[-1] if segment[-1] in pounc else ""
        body = segment[:-1] if tail_punc else segment
        if not body:
            return [segment]

        pieces = []
        current_pos = 0
        while current_pos < len(body):
            end_pos = min(current_pos + token_max_n, len(body))

            if end_pos < len(body):
                search_start = max(current_pos + token_min_n, end_pos - 20)
                if search_start < end_pos:
                    if lang == "en":
                        end_pos = find_english_word_boundary(body, search_start, end_pos)
                    else:
                        for i in range(end_pos, search_start, -1):
                            if body[i] in [' ', '，', ',', '、', '\t', '\n']:
                                end_pos = i + 1
                                break
                if end_pos == current_pos + token_max_n:
                    end_pos = min(current_pos + token_max_n, len(body))

            if end_pos <= current_pos:
                end_pos = min(current_pos + token_max_n, len(body))
                if end_pos <= current_pos:
                    end_pos = current_pos + 1

            pieces.append(body[current_pos:end_pos])
            current_pos = end_pos

        if not pieces:
            return [segment]

        result = []
        for idx, piece in enumerate(pieces):
            if idx == len(pieces) - 1:
                current_punc = tail_punc if tail_punc else default_split_punc
            else:
                current_punc = default_split_punc
            result.append(piece + current_punc)
        return result

    # 检查原始文本是否包含标点符号
    original_text = text.strip()
    if not original_text:  # 防止空字符串访问 [-1]
        return []
        
    has_punctuation = any(p in original_text for p in pounc)
    
    # 确保文本以标点符号结尾
    if original_text[-1] not in pounc:
        if lang == "zh":
            text = original_text + "。"
        else:
            text = original_text + "."
    else:
        text = original_text

    st = 0
    utts = []
    for i, c in enumerate(text):
        if c in pounc:
            # 修复：确保即使连续标点也不会丢失内容
            segment_text = text[st: i]
            if len(segment_text) > 0:
                utts.append(segment_text + c)
                st = i + 1
            elif len(utts) > 0:
                # 如果当前段为空（连续标点），将标点追加到上一个段
                utts[-1] = utts[-1] + c
                st = i + 1
            else:
                # 如果这是第一个标点且前面没有内容，至少保留标点
                utts.append(c)
                st = i + 1
            # 处理引号
            if i + 1 < len(text) and text[i + 1] in ['"', '”']:
                if len(utts) > 0:
                    tmp = utts.pop(-1)
                    utts.append(tmp + text[i + 1])
                    st = i + 2
                else:
                    # 如果utts为空，直接添加引号
                    utts.append(text[i + 1])
                    st = i + 2

    # 如果没有标点符号，需要按长度强制分割
    if not has_punctuation and len(utts) == 1:
        original_text = utts[0][:-1]  # 去掉添加的句号
        utts = []
        current_pos = 0
        
        while current_pos < len(original_text):
            # 计算当前段的最大长度
            max_len = token_max_n
            end_pos = min(current_pos + max_len, len(original_text))
            
            # 如果不是最后一段，尝试在合适的位置分割
            if end_pos < len(original_text):
                # 修复搜索范围计算
                search_start = max(current_pos + token_min_n, end_pos - 20)
                if search_start < end_pos:  # 确保搜索范围有效
                    # 改进的英文单词边界检测
                    if lang == "en":
                        # 使用专门的英文单词边界检测函数
                        end_pos = find_english_word_boundary(original_text, search_start, end_pos)
                    else:
                        # 中文保持原有逻辑
                        for i in range(end_pos, search_start, -1):
                            if original_text[i] in [' ', '，', ',', '、', '\t', '\n']:
                                end_pos = i + 1
                                break
                # 如果找不到合适的分割点，就按最大长度分割
                if end_pos == current_pos + max_len:
                    end_pos = min(current_pos + max_len, len(original_text))
            
            # 防止无限循环
            if end_pos <= current_pos:
                end_pos = current_pos + 1
                
            utts.append(original_text[current_pos:end_pos] + "。")
            current_pos = end_pos

    # 对于包含标点但仍然过长的语句，继续切分
    processed_utts = []
    for utt in utts:
        if calc_utt_length(utt) > token_max_n * 1:
            processed_utts.extend(force_split_segment(utt))
        else:
            processed_utts.append(utt)
    utts = processed_utts

    final_utts = []
    cur_utt = ""
    for utt in utts:
        combined_length = calc_utt_length(cur_utt + utt)
        cur_length = calc_utt_length(cur_utt)
        # 如果合并后会明显超长（超过1.5倍），即使cur_utt很小也要分割
        if combined_length > token_max_n * 1.5:
            if cur_length > 0:
                final_utts.append(cur_utt)
            cur_utt = utt
        # 如果合并后超过token_max_n且cur_utt已达到最小长度，则分割
        elif combined_length > token_max_n and cur_length > token_min_n:
            final_utts.append(cur_utt)
            cur_utt = utt
        else:
            cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if should_merge(cur_utt) and len(final_utts) != 0:
            # 检查合并后是否会明显超长，如果会则不再合并
            merged_length = calc_utt_length(final_utts[-1] + cur_utt)
            if merged_length <= token_max_n * 1.5:
                final_utts[-1] = final_utts[-1] + cur_utt
            else:
                final_utts.append(cur_utt)
        else:
            final_utts.append(cur_utt)

    return final_utts


# remove blank between chinese character
def replace_blank(text: str):
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if ((text[i + 1].isascii() and text[i + 1] != " ") and
                    (text[i - 1].isascii() and text[i - 1] != " ")):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def is_only_punctuation(text):
    # Regular expression: Match strings that consist only of punctuation marks or are empty.
    punctuation_pattern = r'^[\p{P}\p{S}]*$'
    return bool(regex.fullmatch(punctuation_pattern, text))
