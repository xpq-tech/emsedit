from typing import Dict, Union, List
import torch
import json
import numpy as np

from data.base import BaseDataset


class RIPPLE_EFFECTDataset(BaseDataset):
    
    def __getitem__(self, idx) -> Dict[str, Dict[str, torch.LongTensor]]:
        row = self.data[idx]
        
        # 判断是训练数据还是测试数据
        if "rephrase_prompt" in row:  # 训练数据格式
            prompt = row["prompt"]
            equiv_prompt = row["rephrase_prompt"]
            answer = row["target_new"]
            unrel_prompt = row["locality_prompt"]
            unrel_answer = row["locality_ground_truth"]
        
            return {
                "edit_tuples": self.tok_tuples(prompt, answer),
                "equiv_tuples": self.tok_tuples(equiv_prompt, answer),
                "unrel_tuples": self.tok_tuples(unrel_prompt, unrel_answer)
            }
        else:  # 测试数据格式 - 返回与训练数据一致的格式
            return self._process_test_sample(row)
    
    def _process_test_sample(self, s: Dict) -> Dict[str, Dict[str, torch.LongTensor]]:
        """处理测试样本，返回与训练数据一致的格式"""
        prompt = s["prompt"]
        answer = s["target_new"]
        
        # 对于测试数据，我们需要构建 equiv_prompt 和 unrel_prompt
        # equiv_prompt: 从 Subject_Aliasing 中选择第一个可用的 rephrase
        equiv_prompt = prompt  # 默认使用原始 prompt
        if "Subject_Aliasing" in s and s["Subject_Aliasing"]:
            for alias_item in s["Subject_Aliasing"]:
                if alias_item["prompt"]:
                    equiv_prompt = alias_item["prompt"]
                    break
        
        # unrel_prompt 和 unrel_answer: 从 Relation_Specificity 中选择
        unrel_prompt = prompt  # 默认使用原始 prompt
        unrel_answer = answer  # 默认使用原始 answer
        if "Relation_Specificity" in s and s["Relation_Specificity"]:
            for rel_item in s["Relation_Specificity"]:
                if rel_item["prompt"] and rel_item["targets"]:
                    unrel_prompt = rel_item["prompt"]
                    # 选择第一个非空的 target
                    for target in rel_item["targets"]:
                        if target.strip():
                            unrel_answer = target
                            break
                    break
        
        return {
            "edit_tuples": self.tok_tuples(prompt, answer),
            "equiv_tuples": self.tok_tuples(equiv_prompt, answer),
            "unrel_tuples": self.tok_tuples(unrel_prompt, unrel_answer)
        }

    def tok_tuples(
        self,
        prompt: str,
        answer: str
    ) -> Dict[str, torch.LongTensor]:
        """与 zsre 相同的 tokenization 处理"""
        answer = " " + answer
        tok_prompt = self.tok(
            prompt,
            return_tensors="pt",
        )
        tok_answer = self.tok(
            answer,
            return_tensors="pt",
            add_special_tokens=False
        )

        tok_tuples = {
            key: torch.cat((value, tok_answer[key][:, :-1]), -1)
            for key, value in tok_prompt.items()
        }
        
        tok_tuples["labels"] = torch.cat((
            torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
            tok_answer["input_ids"]
        ), -1)

        return tok_tuples
    
    @staticmethod
    def load_and_select_data(path: str, test_i: Union[List, int], shuffle: bool, seed: int):
        """加载和选择数据的静态方法"""
        with open(path, 'r') as f:
            data = json.load(f)
        idx = list(range(len(data)))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
        if test_i == None:
            test_i = idx
        elif type(test_i) == int:
            test_i = idx[:test_i]
        elif type(test_i) == list:
            test_i = [idx[i] for i in test_i]
        else:
            raise ValueError("test_i must be None, int, or list")
        return [data[i] for i in test_i]
    
    @staticmethod
    def ripple_effect(path='data/evaluation/ripple_effect/ripple_effect.json', 
                      test_i: Union[List, int] = None, shuffle=True, seed=0):
        """静态方法：处理 ripple effect 测试数据"""
        data = RIPPLE_EFFECTDataset.load_and_select_data(path, test_i, shuffle, seed)
        test_sample_list = []
        for s in data:
            ns = {}
            ns['example_type'] = s['example_type']
            ns['request'] = {
                'prompt': s['prompt'], 
                'target_new': s['target_new'], 
                'subject': s['subject'],
            }
            if s['example_type'] == 'recent':
                ns['request']['ground_truth'] = s['target_new']
            else:
                ns['request']['ground_truth'] = s['ground_truth']
            gen_types = ['Logical_Generalization', 'Compositionality_I', 
                            'Compositionality_II', 'Subject_Aliasing']
            ns['generality'] = {}
            for gen_type in gen_types:
                ns['generality'][gen_type] = []
                if gen_type in s:
                    for i in s[gen_type]:
                        for t in i['targets']:
                            if t != "":
                                ns['generality'][gen_type].append({'prompt': i['prompt'], 'target': t})
                                break
            loc_types = ['Relation_Specificity', 'Forgetfulness']
            ns['locality'] = {}
            for loc_type in loc_types:
                ns['locality'][loc_type] = []
                if loc_type in s:
                    for i in s[loc_type]:
                        for t in i['targets']:
                            if t != "":
                                ns['locality'][loc_type].append({'prompt': i['prompt'], 'target': t})
                                break
            test_sample_list.append(ns)
        return test_sample_list