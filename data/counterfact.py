from typing import Union, Tuple, List, Dict

import numpy as np
# import scipy
import torch
import random
import math
# import unicodedata
# import nltk
# from sklearn.feature_extraction.text import TfidfVectorizer
# from transformers import AutoModelForCausalLM, AutoTokenizer

from data.base import BaseDataset



class COUNTERFACTDataset(BaseDataset):
    
    def __getitem__(self, idx) -> Dict[str, Dict[str, torch.LongTensor]]:
        row = self.data[idx]
        prompt = row["requested_rewrite"]["prompt"].format(row["requested_rewrite"]["subject"])
        equiv_prompts = row["paraphrase_prompts"]
        answer = row["requested_rewrite"]["target_new"]["str"]
        unrel_prompts = row["neighborhood_prompts"]
        unrel_answer = row["requested_rewrite"]["target_true"]["str"]
        # generation_prompts = row["generation_prompts"]
        if self.path.endswith("train.json"):
            return {
                "edit_tuples": self.tok_tuples(prompt, answer, unrel_answer),
                "equiv_tuples": self.tok_tuples(equiv_prompts[0], answer, unrel_answer),
                "unrel_tuples": self.tok_tuples(unrel_prompts[0], unrel_answer, answer),
            }
        else:
            return {
                "edit_tuples": self.tok_tuples(prompt, answer, unrel_answer),
                "equiv_tuples": [self.tok_tuples(equiv_prompt, answer, unrel_answer) for equiv_prompt in equiv_prompts],
                "unrel_tuples": [self.tok_tuples(unrel_prompt, unrel_answer, answer) for unrel_prompt in unrel_prompts],
            }
        
    
    def collate_fn_val(
        self,
        batch: List[Dict[str, Union[Dict[str, torch.LongTensor], List[Dict[str, torch.LongTensor]]]]]
    ) -> Dict[str, List[Dict[str, torch.LongTensor]]]:
        # 展开多层嵌套结构
        collated_dict = {}
        for key in batch[0].keys():
            # 收集所有样本的该键值（可能是单个元素或列表）
            all_items = []
            for sample in batch:
                items = sample[key]
                if isinstance(items, list):
                    all_items.extend(items)  # 展开列表
                else:
                    all_items.append(items)  # 单个元素直接添加
            
            # 按attention_mask长度降序排序
            sorted_items = sorted(
                all_items,
                key=lambda x: x["attention_mask"].sum().item(),
                reverse=True
            )
            
            # 分批次处理
            batch_size = self.config.batch_size
            batched = [
                self.pad_tok_tuples(sorted_items[i*batch_size : (i+1)*batch_size])
                for i in range(math.ceil(len(sorted_items) / batch_size))
            ]
            
            collated_dict[key] = batched

        return collated_dict


    def tok_tuples(
        self,
        prompt: str,
        answer: str,
        old_answer: str
    ) -> Dict[str, torch.LongTensor]:

        answer = " " + answer
        old_answer = " " + old_answer
        tok_prompt = self.tok(
            prompt,
            return_tensors = "pt",
            truncation=False,
            padding=False
        )
        tok_answer = self.tok(
            answer,
            return_tensors = "pt",
            add_special_tokens = False,
            truncation=False,
            padding=False
        )
        tok_old_answer = self.tok(
            old_answer,
            return_tensors = "pt",
            add_special_tokens = False,
            truncation=False,
            padding=False
        )   
        tok_tuples = {
            key: torch.cat((value, tok_answer[key][:, :-1]), -1)
            for key, value in tok_prompt.items()
        }
        tok_tuples["labels"] = torch.cat((
            torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
            tok_answer["input_ids"]
        ), -1)
        tok_tuples["old_labels"] = torch.cat((
            torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
            tok_old_answer["input_ids"]
        ), -1)

        return tok_tuples
    


# def test_generation(
#     model,
#     tok,
#     prefixes: List[str],
#     consistency_texts: List[str],
#     essence_texts: List[str],
#     vec: TfidfVectorizer,
# ):
#     gen_texts = generate_fast(
#         model,
#         tok,
#         prefixes,
#         n_gen_per_prompt=1,
#         max_out_len=100,
#     )

#     ngram_entropy = n_gram_entropy(gen_texts)
#     consistency_tfidf = tfidf_similarity(
#         " ".join(gen_texts), " ".join(consistency_texts), vec
#     )

#     ret = {
#         "ngram_entropy": ngram_entropy,
#         "reference_score": consistency_tfidf,
#         "text": gen_texts,
#     }

#     if len(essence_texts) > 0:
#         ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
#         ret.update({"essence_score": ppl, "essence_text": essence_texts})

#     return ret


# def generate_fast(
#     model: AutoModelForCausalLM,
#     tok: AutoTokenizer,
#     prompts: List[str],
#     n_gen_per_prompt: int = 1,
#     top_k: int = 5,
#     max_out_len: int = 200,
# ):
#     """
#     Fast, parallelized auto-regressive text generation with top-k sampling.
#     Our custom implementation.
#     """

#     # Unroll prompts and tokenize
#     inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
#     inp_tok = tok(inp, padding=True, return_tensors="pt").to(
#         next(model.parameters()).device
#     )
#     input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
#     batch_size = input_ids.size(0)

#     # Setup storage of fast generation with attention caches.
#     # `cur_context` is used to define the range of inputs that are not yet
#     # stored in `past_key_values`. At each step, we are generating the
#     # next token for the index at `cur_context.stop + 1`.
#     past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

#     with torch.no_grad():
#         while input_ids.size(1) < max_out_len:  # while not exceeding max output length
#             model_out = model(
#                 input_ids=input_ids[:, cur_context],
#                 attention_mask=attention_mask[:, cur_context],
#                 past_key_values=past_key_values,
#                 use_cache=True,
#             )
#             logits, past_key_values = model_out.logits, model_out.past_key_values
#             softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

#             # Top-k sampling
#             tk = torch.topk(softmax_out, top_k, dim=1).indices
#             softmax_out_top_k = torch.gather(softmax_out, 1, tk)
#             softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
#             new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
#             new_toks = torch.gather(tk, 1, new_tok_indices)

#             # If we're currently generating the continuation for the last token in `input_ids`,
#             # create a new index so we can insert the new token
#             if cur_context.stop == input_ids.size(1):
#                 attention_mask = torch.cat(
#                     [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
#                 )
#                 input_ids = torch.cat(
#                     [
#                         input_ids,
#                         input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
#                     ],
#                     dim=1,
#                 )

#             last_non_masked = attention_mask.sum(1) - 1
#             for i in range(batch_size):
#                 new_idx = last_non_masked[i] + 1
#                 if last_non_masked[i].item() + 1 != cur_context.stop:
#                     continue

#                 # Stop generating if we've already maxed out for this prompt
#                 if new_idx < max_out_len:
#                     input_ids[i][new_idx] = new_toks[i]
#                     attention_mask[i][new_idx] = 1

#             cur_context = slice(cur_context.stop, cur_context.stop + 1)

#     txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
#     txt = [
#         unicodedata.normalize("NFKD", x)
#         .replace("\n\n", " ")
#         .replace("<|endoftext|>", "")
#         for x in txt
#     ]

#     return txt


# def n_gram_entropy(gen_texts, agg="arith"):
#     assert agg in ["arith", "geom"]

#     return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
#         [compute_n_gram_entropy(txt) for txt in gen_texts]
#     ).item()


# def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
#     if ns is None:
#         ns = [2, 3]
#     if weights is None:
#         weights = [2 / 3, 4 / 3]
#     assert agg in ["arith", "geom"]

#     entropy_list = []
#     for n in ns:
#         fdist = compute_freq(sentence, n)
#         freqs = np.array([freq for _, freq in fdist.items()])
#         freqs = freqs / freqs.sum()

#         entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

#     entropy_list = np.array(entropy_list) * np.array(weights)

#     return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


# def compute_freq(sentence, n=2):
#     tokens = nltk.word_tokenize(sentence)
#     ngrams = nltk.ngrams(tokens, n)
#     return nltk.FreqDist(ngrams)

# def tfidf_similarity(text_a, text_b, vec):
#     encs = vec.transform([text_a, text_b]).A
#     norm = np.linalg.norm
#     return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()

# def perplexity(
#     model: AutoModelForCausalLM,
#     tok: AutoTokenizer,
#     text: str,
#     max_input_length: int = None,
# ):
#     """
#     Computes perplexity of a piece of text, measured on a reference model.
#     Text is truncated to max_input_length tokens.
#     """

#     inputs = tok(
#         [text], return_tensors="pt", max_length=max_input_length, truncation=True
#     ).to("cuda")

#     logits = torch.nn.functional.log_softmax(model(**inputs).logits, dim=2)
#     log_probs = torch.gather(logits[:, :-1, :], 2, inputs["input_ids"][:, 1:, None])[0]

#     # Perplexity = exp(-1/N * log P(x_1, ..., x_n))
#     return torch.exp(-1 / inputs["input_ids"].size(1) * log_probs.sum()).item()