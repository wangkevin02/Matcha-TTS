#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的用户画像生成系统
整合对话属性提取和画像生成功能，输出统一的JSONL格式
"""

import os
import sys
import json
import argparse
import jsonlines
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

from chatgpt_helper import GPTHelper
from config import zh_persona_prompt, zh_character_prompt, zh_merge_prompt


class ProfileGenerator:
    """用户画像生成器"""
    
    def __init__(self, gpt_helper: GPTHelper, num_workers: int = 16):
        self.gpt_helper = gpt_helper
        self.num_workers = num_workers
    
    @staticmethod
    def read_jsonl(file_path: str) -> List[Dict]:
        """读取JSONL文件"""
        with jsonlines.open(file_path) as reader:
            return [obj for obj in reader]
    
    @staticmethod
    def write_jsonl(data: List[Dict], file_path: str) -> None:
        """写入JSONL文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with jsonlines.open(file_path, mode='w') as writer:
            for item in data:
                writer.write(item)
    
    @staticmethod
    def transfer_utterances_to_string(dialog_lst: List[Dict], order="A-B") -> str:
        """将对话转换为字符串格式"""
        return_str = ""
        if order == "A-B":
            user_role = "A"
            assistant_role = "B"
        elif order == "B-A":
            user_role = "B"
            assistant_role = "A"
        else:
            raise ValueError(f"不支持的对话顺序: {order}")
        for dialog in dialog_lst:
            if dialog['role'] == "user":
                return_str += f"[{user_role}]: {dialog['content']}\n"
            elif dialog['role'] == "assistant":
                return_str += f"[{assistant_role}]: {dialog['content']}\n"
        return return_str.strip()
    
    @staticmethod
    def extract_json_from_string(string: str) -> dict:
        """从字符串中提取JSON对象"""
        try:
            # 使用正则表达式提取JSON对象
            first_left_brace_index = string.find("{")
            last_right_brace_index = string.rfind("}")
            json_str = string[first_left_brace_index:last_right_brace_index+1]
            if json_str:
                json_str = json_str.strip()
                return json.loads(json_str)
            else:
                return None
        except Exception as e:
            print(f"提取JSON对象时发生错误: {e}")
            return None

    @staticmethod
    def random_dict(input_dict: dict) -> dict:
        """随机打乱字典键值对顺序"""
        keys = list(input_dict.keys())
        random.shuffle(keys)
        return {key: input_dict[key] for key in keys}

    def filter_character_output(self, character_output: str) -> Tuple[Dict, Dict]:
        """过滤character_output中的无效信息"""
        json_obj = self.extract_json_from_string(character_output)
        if not json_obj:
            return {}, {}
            
        character_name = json_obj.get("name", "")
        if character_name:
            return_dict = {
                "name": character_name
            }
            json_obj.pop("name")
        else:
            return_dict = {}
            
        new_dict = {}
        for key, value in json_obj.items():
            if not value:
                pass
            else:
                new_dict[key] = value
        new_dict = self.random_dict(new_dict)
        return_dict.update(new_dict)
        
        interlocutor_info = return_dict.get("interlocutor_info", "")
        if interlocutor_info:
            return_dict.pop("interlocutor_info")
        return return_dict, {"interlocutor_info": interlocutor_info}

    def filter_persona_output(self, persona_output: str) -> Dict:
        """过滤persona_output中的无效信息"""
        json_obj = self.extract_json_from_string(persona_output)
        if not json_obj:
            return {}
            
        new_return = {}
        for key, value in json_obj.items():
            if value and value.get("score", "Inconclusive") == "Inconclusive":
                pass
            elif value and value.get("score", "不确定") == "不确定":
                pass
            else:
                new_return[key] = value["conclusion"]
        return self.random_dict(new_return)
    
    def generate_attribute(self, dialog: str, attribute_type: str) -> Tuple[bool, str]:
        """生成单个属性"""
        prompts = {
            'character': zh_character_prompt,
            'persona': zh_persona_prompt, 
        }
        
        if attribute_type not in prompts:
            raise ValueError(f"不支持的属性类型: {attribute_type}")
        
        return self.gpt_helper.request_text_api(
            user_input=dialog,
            system_messages=prompts[attribute_type]
        )
    
    def generate_profile(self, filtered_character_output: Dict, filtered_persona_output: Dict, 
                        interlocutor_info: Dict) -> Tuple[bool, str]:
        """生成最终用户画像"""
        if not filtered_character_output and not filtered_persona_output:
            return False, "知识或性格数据为空"
        
        character_name = filtered_character_output.get("name", "B")
        if character_name == "B":
            character_name = ""
        interlocutor_name = interlocutor_info.get("interlocutor_info", {}).get("name_or_title", "")
        if not interlocutor_name or interlocutor_name == "A":
            interlocutor_name = "对方"

        merge_input = f"""
**客观事实属性**
{filtered_character_output}

**主观特性属性**
{filtered_persona_output}

**当前对话场景**
{interlocutor_info}
"""
        # 替换A、B为实际姓名
        merge_input = merge_input.replace("A", interlocutor_name).replace("B", character_name)

        print(f"merge_input: {merge_input}")
        return self.gpt_helper.request_text_api(
            system_messages=zh_merge_prompt, 
            user_input=merge_input
        )
    
    def process_single_dialog(self, dialog_data: Dict, index: int, order="A-B", save_dir: str = None) -> Dict[str, Any]:
        """处理单个对话，生成完整的用户画像数据"""
        result = {
            'id': index,
            'original_dialog': dialog_data,
            'attributes': {},
            'filtered_attributes': {},
            'profile': None,
            'status': 'processing'
        }
        save_path = os.path.join(save_dir, f"{index}.json")
        if os.path.exists(save_path):
            result['status'] = 'skipped'
            return result
        try:
            # 如果dialog_data是嵌套结构，提取conversation部分
            if isinstance(dialog_data, dict):
                conversation = dialog_data['conversation']
            else:
                conversation = dialog_data
                
            dialog_str = self.transfer_utterances_to_string(conversation, order)
            
            # 生成两种属性
            attribute_types = ['character', 'persona']
            for attr_type in attribute_types:
                flag, response = self.generate_attribute(dialog_str, attr_type)
                if flag:
                    result['attributes'][attr_type] = response
                else:
                    result['attributes'][attr_type] = None
                    print(f"Warning: 生成 {attr_type} 属性失败 (ID: {index})")
            
            # 过滤属性
            if result['attributes'].get('character'):
                filtered_character, interlocutor_info = self.filter_character_output(
                    result['attributes']['character']
                )
                result['filtered_attributes']['character'] = filtered_character
                result['filtered_attributes']['interlocutor_info'] = interlocutor_info
            else:
                result['filtered_attributes']['character'] = {}
                result['filtered_attributes']['interlocutor_info'] = {}
                
            if result['attributes'].get('persona'):
                filtered_persona = self.filter_persona_output(result['attributes']['persona'])
                result['filtered_attributes']['persona'] = filtered_persona
            else:
                result['filtered_attributes']['persona'] = {}
            
            # 如果有足够的属性，生成最终画像
            if (result['filtered_attributes'].get('character') or 
                result['filtered_attributes'].get('persona')):
                flag, profile = self.generate_profile(
                    result['filtered_attributes']['character'], 
                    result['filtered_attributes']['persona'],
                    result['filtered_attributes']['interlocutor_info']
                )
                if flag:
                    result['profile'] = profile
                    result['status'] = 'success'
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=4)
                else:
                    result['status'] = 'profile_generation_failed'
                    print(f"Warning: 生成最终画像失败 (ID: {index})")
            else:
                result['status'] = 'insufficient_attributes'
                print(f"Warning: 属性不足，无法生成画像 (ID: {index})")
        
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            print(f"Error: 处理对话失败 (ID: {index}): {e}")
        
        return result
    
    def process_batch(self, dialog_list: List[Dict], max_count: Optional[int] = None, order="A-B", save_dir: str = None) -> List[Dict[str, Any]]:
        """批量处理对话"""
        if max_count:
            dialog_list = dialog_list[:max_count]
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self.process_single_dialog, dialog, i, order, save_dir): i 
                for i, dialog in enumerate(dialog_list)
            }
            
            # 收集结果
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="生成用户画像"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"任务执行失败: {e}")
        
        # 按ID排序
        results.sort(key=lambda x: x['id'])
        return results
    
    def save_results(self, results: List[Dict], output_file: str) -> None:
        """保存结果到JSONL文件"""
        self.write_jsonl(results, output_file)
        print(f"结果已保存到: {output_file}")
        
        # 打印统计信息
        total = len(results)
        success = sum(1 for r in results if r['status'] == 'success')
        print(f"处理统计: 总计 {total}, 成功 {success}, 成功率 {success/total*100:.2f}%")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="完整的用户画像生成系统")
    
    # 输入输出参数
    parser.add_argument('--input_file', type=str, default="./openai_format/train.jsonl",
                       help="输入的JSONL文件路径")
    parser.add_argument('--output_dir', type=str, default="./profile_data",
                       help="输出的JSONL文件路径")
    
    # 处理参数
    parser.add_argument('--max_count', type=int, default=50,
                       help="最大处理数量，默认处理全部")
    parser.add_argument('--num_workers', type=int, default=4,
                       help="并行工作线程数")
    parser.add_argument('--order', type=str, default="A-B", choices=["A-B", "B-A"],
                       help="对话角色顺序，A-B表示user是A，assistant是B")
    
    # 示例参数
    parser.add_argument('--show_examples', type=int, default=3,
                       help="显示示例数量")

    parser.add_argument('--model', type=str, default="gpt-4o",
                       help="使用的模型")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化
    print("=" * 60)
    print("用户画像生成系统启动")
    print(f"输入文件: {args.input_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"最大处理数量: {args.max_count or '全部'}")
    print(f"并行线程数: {args.num_workers}")
    print(f"对话顺序: {args.order}")
    print("=" * 60)
    
    # 初始化GPT助手和生成器
    try:
        gpt_helper = GPTHelper(model_config={"model": args.model})
        generator = ProfileGenerator(gpt_helper, args.num_workers)
        print(f"GPT助手初始化成功: {gpt_helper}")
    except Exception as e:
        print(f"GPT助手初始化失败: {e}")
        return
    
    # 读取数据
    try:
        dialog_list = generator.read_jsonl(args.input_file)
        print(f"成功读取 {len(dialog_list)} 条对话数据")
    except Exception as e:
        print(f"读取输入文件{args.input_file}失败: {e}")
        return
    
    # 处理数据
    print("\n开始处理对话数据...")
    results = generator.process_batch(
        dialog_list, 
        max_count=args.max_count,
        order=args.order,
        save_dir=args.output_dir
    )
    
    # 保存结果
    generator.save_results(results, os.path.join(args.output_dir, "profile_data.jsonl"))
    
    print("\n处理完成!")


if __name__ == "__main__":
    main() 