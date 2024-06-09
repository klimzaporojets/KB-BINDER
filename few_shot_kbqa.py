import gc
import os
import pickle
from typing import List

import openai
import json

import psutil
import spacy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer

from sparql_exe import execute_query, get_types, get_2hop_relations, lisp_to_sparql
from utils import process_file, process_file_node, process_file_rela, process_file_test
from rank_bm25 import BM25Okapi
from time import sleep
import re
import logging
from collections import Counter
import argparse
from pyserini.search import FaissSearcher, LuceneSearcher
from pyserini.search.hybrid import HybridSearcher
from pyserini.search.faiss import AutoQueryEncoder
import random
import itertools
from ollama import generate

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("time recoder")
import torch


def select_shot_prompt_train(train_data_in, shot_number):
    random.shuffle(train_data_in)
    compare_list = ["le", "ge", "gt", "lt", "ARGMIN", "ARGMAX"]
    if shot_number == 1:
        selected_quest_compose = [train_data_in[0]["question"]]
        selected_quest_compare = [train_data_in[0]["question"]]
        selected_quest = [train_data_in[0]["question"]]
    else:
        selected_quest_compose = []
        selected_quest_compare = []
        each_type_num = shot_number // 2
        for data in train_data_in:
            if any([x in data['s_expression'] for x in compare_list]):
                selected_quest_compare.append(data["question"])
                if len(selected_quest_compare) == each_type_num:
                    break
        for data in train_data_in:
            if not any([x in data['s_expression'] for x in compare_list]):
                selected_quest_compose.append(data["question"])
                if len(selected_quest_compose) == each_type_num:
                    break
        mix_type_num = each_type_num // 3
        selected_quest = selected_quest_compose[:mix_type_num] + selected_quest_compare[:mix_type_num]
    logger.info("selected_quest_compose: {}".format(selected_quest_compose))
    logger.info("selected_quest_compare: {}".format(selected_quest_compare))
    logger.info("selected_quest: {}".format(selected_quest))
    return selected_quest_compose, selected_quest_compare, selected_quest


def sub_mid_to_fn(question, string, question_to_mid_dict):
    seg_list = string.split()
    mid_to_start_idx_dict = {}
    for seg in seg_list:
        if seg.startswith("m.") or seg.startswith("g."):
            mid = seg.strip(')(')
            start_index = string.index(mid)
            mid_to_start_idx_dict[mid] = start_index
    if len(mid_to_start_idx_dict) == 0:
        return string
    start_index = 0
    new_string = ''
    for key in mid_to_start_idx_dict:
        b_idx = mid_to_start_idx_dict[key]
        e_idx = b_idx + len(key)
        new_string = new_string + string[start_index:b_idx] + question_to_mid_dict[question][key]
        start_index = e_idx
    new_string = new_string + string[start_index:]
    return new_string


def get_llm_output(prompt, api_key, LLM_engine, nr_choices, temperature, type_output,
                   is_cuda_available) -> List[str]:
    print(f'invoked get_llm_output with type_output: {type_output} and prompt: \n {prompt}')
    if LLM_engine == 'dummy' and type_output == 'type_generator':
        gene_exp = """Type of the question: Composition
    Answer: Glynne Polan is an opera designer who designed the telephone / the medium.
    """
        gene_exp = gene_exp.lower()
        gene_exp = gene_exp[gene_exp.index('question:') + len('question:'):gene_exp.index('answer:')].strip()
        return [gene_exp]
    elif LLM_engine == 'dummy' and type_output == 'ep_generator':
        gene_exp = "(AND occupation.opera_designer (JOIN occupation.opera_designer.roles " \
                   "(JOIN opera_designer.opera_designer_role.opera_name Telephone/The Medium)))" \
                   "\n\nNote: The logical form for the last question is based on the assumption that the opera " \
                   "designer\'s role in the specific opera \")Telephone/The Medium\" can be linked to their " \
                   "occupation. However, since \"Telephone/The Medium\" is not a known opera, the logical form may" \
                   " need to be adjusted based on the actual opera or production in question."
        # an extract from example of prompt for ep_generator:
        # -----
        # Question: which rocket engine has the isp (sea level) of 243.6?
        # Logical Form: (AND spaceflight.rocket_engine (JOIN spaceflight.rocket_engine.isp_sea_level 243.6^^http://www.w3.org/2001/XMLSchema#float))
        # Question: who is the basketball coach that has 51 playoff losses throughout his career?
        # Logical Form: (AND basketball.basketball_coach (JOIN basketball.basketball_coach.playoff_losses 51^^http://www.w3.org/2001/XMLSchema#integer))
        # Question: what is the role of opera designer gig who designed the telephone / the medium?
        # Logical Form:
        #
        #
        return [gene_exp]
    elif LLM_engine == 'ollama:phi':
        response = generate('phi', 'Why is the sky blue?')
        print(response['response'])
        exit()
    elif LLM_engine == 'huggingface:Phi-3-mini-4k-instruct':
        print(f'invoking the hugging face engine: {LLM_engine}')
        device_map = 'cuda' if is_cuda_available else 'cpu'
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            device_map=device_map,
            torch_dtype="auto",
            trust_remote_code=True
        )
        print('type of object model: ', type(model))
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        print('type of object tokenizer: ', type(tokenizer))

        # messages = [
        #     {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        #     {"role": "assistant",
        #      "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
        #     {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
        # ]
        messages = [
            {"role": "user", "content": prompt},
        ]

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        if temperature > 0.0:
            generation_args = {
                "max_new_tokens": 256,
                "return_full_text": False,
                "temperature": temperature,
                "do_sample": True
                # "top_p": 1
                # "do_sample": False,
            }
        else:
            generation_args = {
                "max_new_tokens": 256,
                "return_full_text": False,
                "temperature": temperature,
                "do_sample": False
                # "top_p": 1
                # "do_sample": False,
            }
        print('starting running pipe')
        to_ret = list()
        for i in range(nr_choices):
            print('--------------')
            output = pipe(messages, **generation_args)
            # print('huggingface generated text: ', output[0]['generated_text'])
            print(f'{i} huggingface_generated {type_output} raw text: {output}')
            gene_exp = output[0]['generated_text']
            gene_exp_l = gene_exp.lower()
            if type_output == 'type_generator':
                if 'answer:' in gene_exp_l:
                    gene_exp = gene_exp[
                               gene_exp_l.index('question:') + len('question:'):gene_exp_l.index('answer:')].strip()
                elif 'response:' in gene_exp:
                    gene_exp = gene_exp[
                               gene_exp_l.index('question:') + len('question:'):gene_exp_l.index('response:')].strip()
            elif type_output == 'ep_generator':
                if 'question:' in gene_exp:
                    gene_exp = gene_exp[:gene_exp_l.index('question:')].strip()

                if '\n\n' in gene_exp:
                    gene_exp = gene_exp[:gene_exp.index('\n\n')].strip()

                # if '\n' in gene_exp:
                #     gene_exp = gene_exp[:gene_exp.index('\n')].strip()
                if 'Logical Form:' in gene_exp:
                    gene_exp = gene_exp[:gene_exp.index('Logical Form:')].strip()

            to_ret.append(gene_exp)
            print('***')
            print(f'{i} huggingface_generated {type_output} cleaned text: {gene_exp}')
            print('--------------')
        del tokenizer
        del model
        gc.collect()
        print(f'to_ret returned: {to_ret}')
        return to_ret
    else:
        got_result = False
        while not got_result:
            try:
                openai.api_key = api_key
                answer_modi = openai.Completion.create(
                    engine=LLM_engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=256,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=["Question: "],
                    n=nr_choices
                )
                got_result = True
            except Exception as e:
                # sleep(3)
                e.args = ("Exception while accessing the API. The original message:\n%s" % e.args,)
                raise
        # gene_exp = answer_modi["choices"][0]["text"].strip()
        gene_exp = [exp["text"].strip() for exp in answer_modi["choices"]]
        return gene_exp


def type_generator(question, prompt_type, api_key, LLM_engine, is_cuda_available) -> str:
    sleep(1)

    prompt = prompt_type
    prompt = prompt + " Question: " + question + "Type of the question: "
    # got_result = False
    to_ret = get_llm_output(prompt, api_key, LLM_engine, nr_choices=1, temperature=0.0, type_output='type_generator',
                            is_cuda_available=is_cuda_available)
    return to_ret[0]


def ep_generator(question, selected_examples, temp, que_to_s_dict_train, question_to_mid_dict, api_key, LLM_engine,
                 retrieval=False, corpus=None, nlp_model=None, bm25_train_full=None, retrieve_number=100,
                 is_cuda_available=False):
    if retrieval:
        tokenized_query = nlp_model(question)
        tokenized_query = [token.lemma_ for token in tokenized_query]
        top_ques = bm25_train_full.get_top_n(tokenized_query, corpus, n=retrieve_number)
        doc_scores = bm25_train_full.get_scores(tokenized_query)
        top_score = max(doc_scores)
        logger.info("top_score: {}".format(top_score))
        logger.info("top related questions: {}".format(top_ques))
        selected_examples = top_ques
    prompt = ""
    for que in selected_examples:
        if not que_to_s_dict_train[que]:
            continue
        prompt = prompt + "Question: " + que + "\n" + "Logical Form: " + sub_mid_to_fn(que, que_to_s_dict_train[que],
                                                                                       question_to_mid_dict) + "\n"
    prompt = prompt + "Question: " + question + "\n" + "Logical Form: "
    # got_result = False

    # TODO kzaporoj: here rewrite
    # while got_result != True:
    #     try:
    #         openai.api_key = api_key
    #         answer_modi = openai.Completion.create(
    #             engine=LLM_engine,
    #             prompt=prompt,
    #             temperature=temp,
    #             max_tokens=256,
    #             top_p=1,
    #             frequency_penalty=0,
    #             presence_penalty=0,
    #             stop=["Question: "],
    #             n=7
    #         )
    #         got_result = True
    #     except:
    #         sleep(3)
    # gene_exp = [exp["text"].strip() for exp in answer_modi["choices"]]
    gene_exp = get_llm_output(prompt, api_key, LLM_engine, nr_choices=7, temperature=temp, type_output='ep_generator',
                              is_cuda_available=is_cuda_available)
    return gene_exp


def convert_to_frame(s_exp):
    phrase_set = ["(JOIN", "(ARGMIN", "(ARGMAX", "(R", "(le", "(lt", "(ge", "(gt", "(COUNT", "(AND", "(TC", "(CONS"]
    seg_list = s_exp.split()
    after_filter_list = []
    for seg in seg_list:
        for phrase in phrase_set:
            if phrase in seg:
                after_filter_list.append(phrase)
        if ")" in seg:
            after_filter_list.append(''.join(i for i in seg if i == ')'))
    return ''.join(after_filter_list)


def find_friend_name(gene_exp, org_question):
    seg_list = gene_exp.split()
    phrase_set = ["(JOIN", "(ARGMIN", "(ARGMAX", "(R", "(le", "(lt", "(ge", "(gt", "(COUNT", "(AND"]
    temp = []
    reg_ents = []
    for i, seg in enumerate(seg_list):
        if not any([ph in seg for ph in phrase_set]):
            if seg.lower() in org_question:
                temp.append(seg.lower())
            if seg.endswith(')'):
                stripped = seg.strip(')')
                stripped_add = stripped + ')'
                if stripped_add.lower() in org_question:
                    temp.append(stripped_add.lower())
                    reg_ents.append(" ".join(temp).lower())
                    temp = []
                elif stripped.lower() in org_question:
                    temp.append(stripped.lower())
                    reg_ents.append(" ".join(temp).lower())
                    temp = []
    if len(temp) != 0:
        reg_ents.append(" ".join(temp))
    return reg_ents


def get_right_mid_set(fn, id_dict, question):
    type_to_mid_dict = {}
    type_list = []
    for mid in id_dict:
        types = get_types(mid)
        for cur_type in types:
            if not cur_type.startswith("common.") and not cur_type.startswith("base."):
                if cur_type not in type_to_mid_dict:
                    type_to_mid_dict[cur_type] = {}
                    type_to_mid_dict[cur_type][mid] = id_dict[mid]
                else:
                    type_to_mid_dict[cur_type][mid] = id_dict[mid]
                type_list.append(cur_type)
    tokenized_type_list = [re.split('\.|_', doc) for doc in type_list]
    #     tokenized_question = tokenizer.tokenize(question)
    tokenized_question = question.split()
    bm25 = BM25Okapi(tokenized_type_list)
    top10_types = bm25.get_top_n(tokenized_question, type_list, n=10)
    selected_types = top10_types[:3]
    selected_mids = []
    for any_type in selected_types:
        # logger.info("any_type: {}".format(any_type))
        # logger.info("type_to_mid_dict[any_type]: {}".format(type_to_mid_dict[any_type]))
        selected_mids += list(type_to_mid_dict[any_type].keys())
    return selected_mids


def from_fn_to_id_set(fn_list, question, name_to_id_dict, bm25_all_fns, all_fns):
    return_mid_list = []
    for fn_org in fn_list:
        drop_dot = fn_org.split()
        drop_dot = [seg.strip('.') for seg in drop_dot]
        drop_dot = " ".join(drop_dot)
        if fn_org.lower() not in question and drop_dot.lower() in question:
            fn_org = drop_dot
        if fn_org.lower() not in name_to_id_dict:
            logger.info("fn_org: {}".format(fn_org.lower()))
            tokenized_query = fn_org.lower().split()
            fn = bm25_all_fns.get_top_n(tokenized_query, all_fns, n=1)[0]
            logger.info("sub fn: {}".format(fn))
        else:
            fn = fn_org
        if fn.lower() in name_to_id_dict:
            id_dict = name_to_id_dict[fn.lower()]
        if len(id_dict) > 15:
            mids = get_right_mid_set(fn.lower(), id_dict, question)
        else:
            mids = sorted(id_dict.items(), key=lambda x: x[1], reverse=True)
            mids = [mid[0] for mid in mids]
        return_mid_list.append(mids)
    return return_mid_list


def convz_fn_to_mids(gene_exp, found_names, found_mids):
    if len(found_names) == 0:
        return gene_exp
    start_index = 0
    new_string = ''
    for name, mid in zip(found_names, found_mids):
        b_idx = gene_exp.lower().index(name)
        e_idx = b_idx + len(name)
        new_string = new_string + gene_exp[start_index:b_idx] + mid
        start_index = e_idx
    new_string = new_string + gene_exp[start_index:]
    return new_string


def add_reverse(org_exp):
    final_candi = [org_exp]
    total_join = 0
    list_seg = org_exp.split(" ")
    for seg in list_seg:
        if "JOIN" in seg:
            total_join += 1
    for i in range(total_join):
        final_candi = final_candi + add_reverse_index(final_candi, i + 1)
    return final_candi


def add_reverse_index(list_of_e, join_id):
    added_list = []
    list_of_e_copy = list_of_e.copy()
    for exp in list_of_e_copy:
        list_seg = exp.split(" ")
        count = 0
        for i, seg in enumerate(list_seg):
            if "JOIN" in seg and "." in list_seg[i + 1]:
                count += 1
                if count != join_id:
                    continue
                list_seg[i + 1] = "(R " + list_seg[i + 1] + ")"
                added_list.append(" ".join(list_seg))
                break
            if "JOIN" in seg and "(R" in list_seg[i + 1]:
                count += 1
                if count != join_id:
                    continue
                list_seg[i + 1] = ""
                list_seg[i + 2] = list_seg[i + 2][:-1]
                added_list.append(" ".join(" ".join(list_seg).split()))
                break
    return added_list


def bound_to_existed(question, s_expression, found_mids, two_hop_rela_dict,
                     relationship_to_enti, hsearcher, rela_corpus, relationships):
    possible_relationships_can = []
    possible_relationships = []
    # logger.info("before 2 hop rela")
    updating_two_hop_rela_dict = two_hop_rela_dict.copy()
    for mid in found_mids:
        if mid in updating_two_hop_rela_dict:
            relas = updating_two_hop_rela_dict[mid]
            possible_relationships_can += list(set(relas[0]))
            possible_relationships_can += list(set(relas[1]))
        else:
            relas = get_2hop_relations(mid)
            updating_two_hop_rela_dict[mid] = relas
            possible_relationships_can += list(set(relas[0]))
            possible_relationships_can += list(set(relas[1]))
    # logger.info("after 2 hop rela")
    for rela in possible_relationships_can:
        if not rela.startswith('common') and not rela.startswith('base') and not rela.startswith('type'):
            possible_relationships.append(rela)
    if not possible_relationships:
        possible_relationships = relationships.copy()
    expression_segment = s_expression.split(" ")
    # print("possible_relationships: ", possible_relationships)
    possible_relationships = list(set(possible_relationships))
    relationship_replace_dict = {}
    lemma_tags = {"NNS", "NNPS"}
    for i, seg in enumerate(expression_segment):
        processed_seg = seg.strip(')')
        if '.' in seg and not seg.startswith('m.') and not seg.startswith('g.') and not (
                expression_segment[i - 1].endswith("AND") or expression_segment[i - 1].endswith("COUNT") or
                expression_segment[i - 1].endswith("MAX") or expression_segment[i - 1].endswith("MIN")) and (
                not any(ele.isupper() for ele in seg)):
            tokenized_query = re.split('\.|_', processed_seg)
            tokenized_query = " ".join(tokenized_query)
            tokenized_question = question.strip(' ?')
            tokenized_query = tokenized_query + ' ' + tokenized_question
            searched_results = hsearcher.search(tokenized_query, k=1000)
            top3_ques = []
            for hit in searched_results:
                if len(top3_ques) > 7:
                    break
                cur_result = json.loads(rela_corpus.doc(str(hit.docid)).raw())
                cur_rela = cur_result['rel_ori']
                if not cur_rela.startswith("base.") and not cur_rela.startswith("common.") and \
                        not cur_rela.endswith("_inv.") and len(cur_rela.split('.')) > 2 and \
                        cur_rela in possible_relationships:
                    top3_ques.append(cur_rela)
            logger.info("top3_ques rela: {}".format(top3_ques))
            relationship_replace_dict[i] = top3_ques[:7]
    if len(relationship_replace_dict) > 5:
        return None, updating_two_hop_rela_dict, None
    elif len(relationship_replace_dict) >= 3:
        for key in relationship_replace_dict:
            relationship_replace_dict[key] = relationship_replace_dict[key][:4]
    combinations = list(relationship_replace_dict.values())
    all_iters = list(itertools.product(*combinations))
    rela_index = list(relationship_replace_dict.keys())
    # logger.info("all_iters: {}".format(all_iters))
    for iters in all_iters:
        expression_segment_copy = expression_segment.copy()
        possible_entities_set = []
        for i in range(len(iters)):
            suffix = ""
            for k in range(len(expression_segment[rela_index[i]].split(')')) - 1):
                suffix = suffix + ')'
            expression_segment_copy[rela_index[i]] = iters[i] + suffix
            if iters[i] in relationship_to_enti:
                possible_entities_set += relationship_to_enti[iters[i]]
        if not possible_entities_set:
            continue
        enti_replace_dict = {}
        for j, seg in enumerate(expression_segment):
            processed_seg = seg.strip(')')
            if '.' in seg and not seg.startswith('m.') and not seg.startswith('g.') and (
                    expression_segment[j - 1].endswith("AND") or expression_segment[j - 1].endswith("COUNT") or
                    expression_segment[j - 1].endswith("MAX") or expression_segment[j - 1].endswith("MIN")) and (
                    not any(ele.isupper() for ele in seg)):
                tokenized_enti = [re.split('\.|_', doc) for doc in possible_entities_set]
                tokenized_query = re.split('\.|_', processed_seg)
                bm25 = BM25Okapi(tokenized_enti)
                top3_ques = bm25.get_top_n(tokenized_query, possible_entities_set, n=3)
                enti_replace_dict[j] = list(set(top3_ques))
        combinations_enti = list(enti_replace_dict.values())
        all_iters_enti = list(itertools.product(*combinations_enti))
        enti_index = list(enti_replace_dict.keys())
        for iter_ent in all_iters_enti:
            for k in range(len(iter_ent)):
                suffix = ""
                for h in range(len(expression_segment[enti_index[k]].split(')')) - 1):
                    suffix = suffix + ')'
                expression_segment_copy[enti_index[k]] = iter_ent[k] + suffix
            final = " ".join(expression_segment_copy)
            added = add_reverse(final)
            for exp in added:
                try:
                    answer = generate_answer([exp])
                except:
                    answer = None
                if answer is not None:
                    return answer, updating_two_hop_rela_dict, exp
    return None, updating_two_hop_rela_dict, None


def generate_answer(list_exp):
    for exp in list_exp:
        try:
            sparql = lisp_to_sparql(exp)
        except:
            continue
        try:
            re = execute_query(sparql)
        except:
            continue
        if re:
            if re[0].isnumeric():
                if re[0] == '0':
                    continue
                else:
                    return re
            else:
                return re
    return None


def number_of_join(exp):
    count = 0
    seg_list = exp.split()
    for seg in seg_list:
        if "JOIN" in seg:
            count += 1
    return count


def process_file_codex_output(filename_before, filename_after):
    codex_eps_dict_before = json.load(open(filename_before, 'r'), strict=False)
    codex_eps_dict_after = json.load(open(filename_after, 'r'), strict=False)
    for key in codex_eps_dict_after:
        codex_eps_dict_before[key] = codex_eps_dict_after[key]
    return codex_eps_dict_before


def all_combiner_evaluation(data_batch, selected_quest_compare, selected_quest_compose, selected_quest,
                            prompt_type, hsearcher, rela_corpus, relationships, temp, que_to_s_dict_train,
                            question_to_mid_dict, api_key, LLM_engine, name_to_id_dict, bm25_all_fns, all_fns,
                            relationship_to_enti, retrieval=False, corpus=None, nlp_model=None, bm25_train_full=None,
                            retrieve_number=100, is_cuda_available=False):
    correct = [0] * 6
    total = [0] * 6
    no_ans = [0] * 6
    for data in data_batch:
        logger.info("==========")
        logger.info("data[id]: {}".format(data["id"]))
        logger.info("data[question]: {}".format(data["question"]))
        logger.info("data[exp]: {}".format(data["s_expression"]))
        label = []
        for ans in data["answer"]:
            label.append(ans["answer_argument"])
        if not retrieval:
            gene_type = type_generator(data["question"], prompt_type, api_key, LLM_engine,
                                       is_cuda_available=is_cuda_available)
            logger.info("gene_type: {}".format(gene_type))
        else:
            gene_type = None

        # if gene_type == "Comparison":
        if gene_type.lower() == "comparison":
            gene_exps = ep_generator(data["question"],
                                     list(set(selected_quest_compare) | set(selected_quest)),
                                     temp, que_to_s_dict_train, question_to_mid_dict, api_key, LLM_engine,
                                     retrieval=retrieval, corpus=corpus, nlp_model=nlp_model,
                                     bm25_train_full=bm25_train_full, retrieve_number=retrieve_number,
                                     is_cuda_available=is_cuda_available)
        else:
            gene_exps = ep_generator(data["question"],
                                     list(set(selected_quest_compose) | set(selected_quest)),
                                     temp, que_to_s_dict_train, question_to_mid_dict, api_key, LLM_engine,
                                     retrieval=retrieval, corpus=corpus, nlp_model=nlp_model,
                                     bm25_train_full=bm25_train_full, retrieve_number=retrieve_number,
                                     is_cuda_available=is_cuda_available)
        two_hop_rela_dict = {}
        answer_candi = []
        removed_none_candi = []
        answer_to_grounded_dict = {}
        logger.info("gene_exps: {}".format(gene_exps))
        scouts = gene_exps[:6]
        for idx, gene_exp in enumerate(scouts):
            try:
                logger.info("gene_exp: {}".format(gene_exp))
                join_num = number_of_join(gene_exp)
                if join_num > 5:
                    continue
                if join_num > 3:
                    top_mid = 5
                else:
                    top_mid = 15
                found_names = find_friend_name(gene_exp, data["question"])
                found_mids = from_fn_to_id_set(found_names, data["question"], name_to_id_dict, bm25_all_fns, all_fns)
                found_mids = [mids[:top_mid] for mids in found_mids]
                mid_combinations = list(itertools.product(*found_mids))
                logger.info("all_iters: {}".format(mid_combinations))
                for mid_iters in mid_combinations:
                    logger.info("mid_iters: {}".format(mid_iters))
                    replaced_exp = convz_fn_to_mids(gene_exp, found_names, mid_iters)

                    answer, two_hop_rela_dict, bounded_exp = bound_to_existed(data["question"], replaced_exp, mid_iters,
                                                                              two_hop_rela_dict, relationship_to_enti,
                                                                              hsearcher, rela_corpus, relationships)
                    answer_candi.append(answer)
                    if answer is not None:
                        answer_to_grounded_dict[tuple(answer)] = bounded_exp
                for ans in answer_candi:
                    if ans != None:
                        removed_none_candi.append(ans)
                if not removed_none_candi:
                    answer = None
                else:
                    count_dict = Counter([tuple(candi) for candi in removed_none_candi])
                    logger.info("count_dict: {}".format(count_dict))
                    answer = max(count_dict, key=count_dict.get)
            except:
                if not removed_none_candi:
                    answer = None
                else:
                    count_dict = Counter([tuple(candi) for candi in removed_none_candi])
                    logger.info("count_dict: {}".format(count_dict))
                    answer = max(count_dict, key=count_dict.get)
            answer_to_grounded_dict[None] = ""
            logger.info("predicted_answer: {}".format(answer))
            logger.info("label: {}".format(label))
            if answer is None:
                no_ans[idx] += 1
            elif set(answer) == set(label):
                correct[idx] += 1
            total[idx] += 1
            em_score = correct[idx] / total[idx]
            logger.info("================================================================")
            logger.info("consistent candidates number: {}".format(idx + 1))
            logger.info("em_score: {}".format(em_score))
            logger.info("correct: {}".format(correct[idx]))
            logger.info("total: {}".format(total[idx]))
            logger.info("no_ans: {}".format(no_ans[idx]))
            logger.info(" ")
            logger.info("================================================================")


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--shot_num', type=int, metavar='N',
                        default=40, help='the number of shots used in in-context demo')
    parser.add_argument('--temperature', type=float, metavar='N',
                        default=0.3, help='the temperature of LLM')
    parser.add_argument('--api_key', type=str, metavar='N',
                        default=None, help='the api key to access LLM')
    parser.add_argument('--engine', type=str, metavar='N',
                        default="huggingface:Phi-3-mini-4k-instruct", help='engine name of LLM')
    # default="dummy", help='engine name of LLM')
    # default="huggingface:Phi-3-mini-4k-instruct", help='engine name of LLM')
    # default="code-davinci-002", help='engine name of LLM')
    parser.add_argument('--retrieval', action='store_true', help='whether to use retrieval-augmented KB-BINDER')
    parser.add_argument('--train_data_path', type=str, metavar='N',
                        default="data/GrailQA/grailqa_v1.0_train.json", help='training data path')
    parser.add_argument('--eva_data_path', type=str, metavar='N',
                        default="data/GrailQA/grailqa_v1.0_dev.json", help='evaluation data path')
    parser.add_argument('--fb_roles_path', type=str, metavar='N',
                        default="data/GrailQA/fb_roles", help='freebase roles file path')
    parser.add_argument('--surface_map_path', type=str, metavar='N',
                        default="data/surface_map_file_freebase_complete_all_mention", help='surface map file path')
    parser.add_argument('--grailqa_cache_path', type=str, metavar='N',
                        default="data/GrailQA/cache/", help='cache where things will be saved')
    parser.add_argument('--surface_lines_cache_path', type=str, metavar='N',
                        default="data/surface_lines.pickle", help='cache where the surface_lines will be pickled')
    parser.add_argument('--debug_nr_surface_lines', type=int, metavar='N',
                        default=-1, help='the number of shots used in in-context demo')

    args = parser.parse_args()
    return args


def print_available_memory(prefix: str = ''):
    available_ram = psutil.virtual_memory().available

    # Print the available RAM memory
    # f"{original_num:.2f}"
    print('-----------------------')
    print(f'{prefix} - free RAM memory: {available_ram / 1024 / 1024 / 1024:.2f} Gb')
    print('-----------------------')


def main():
    is_cuda_available = torch.cuda.is_available()

    args = parse_args()
    print_available_memory('before spacy.load')
    nlp = spacy.load("en_core_web_sm")
    print_available_memory('after spacy.load')

    bm25_searcher = LuceneSearcher('contriever_fb_relation/index_relation_fb')
    print_available_memory('after LuceneSearcher')

    query_encoder = AutoQueryEncoder(encoder_dir='facebook/contriever', pooling='mean')
    print_available_memory('after AutoQueryEncoder')

    contriever_searcher = FaissSearcher('contriever_fb_relation/freebase_contriever_index', query_encoder)
    print_available_memory('after FaissSearcher')

    hsearcher = HybridSearcher(contriever_searcher, bm25_searcher)
    print_available_memory('after HybridSearcher')

    rela_corpus = LuceneSearcher('contriever_fb_relation/index_relation_fb')
    print_available_memory('after LuceneSearcher')

    dev_data = process_file(args.eva_data_path)
    print_available_memory('after for dev_data')

    train_data = process_file(args.train_data_path)
    print_available_memory('after train_data')

    que_to_s_dict_train = {data["question"]: data["s_expression"] for data in train_data}
    print_available_memory('after que_to_s_dict_train')

    question_to_mid_dict = process_file_node(args.train_data_path)
    print_available_memory('after question_to_mid_dict')
    if not args.retrieval:
        selected_quest_compose, selected_quest_compare, selected_quest = select_shot_prompt_train(train_data,
                                                                                                  args.shot_num)
        print_available_memory('after select_shot_prompt_train')
    else:
        selected_quest_compose = []
        selected_quest_compare = []
        selected_quest = []

    all_ques = selected_quest_compose + selected_quest_compare
    print_available_memory('after all_ques')
    corpus = [data["question"] for data in train_data]
    print_available_memory('after corpus')
    tokenized_train_data = []
    grailqa_cache_path = args.grailqa_cache_path
    os.makedirs(grailqa_cache_path, exist_ok=True)
    pickle_tokenized_train_path = os.path.join(grailqa_cache_path, 'grailqa_tokenized_train.pickle')
    if os.path.exists(pickle_tokenized_train_path):
        print('pickle_tokenized_train_path exists, loading')
        tokenized_train_data = pickle.load(open(pickle_tokenized_train_path, 'rb'))
        print('loaded from pickle_tokenized_train_path')
    else:
        for doc in tqdm(corpus):
            nlp_doc = nlp(doc)
            tokenized_train_data.append([token.lemma_ for token in nlp_doc])
        print('pickling the tokenized_train_data')
        with open(pickle_tokenized_train_path, 'wb') as outfile:
            pickle.dump(tokenized_train_data, outfile)
        print('finished pickling the tokenized_train_data')
    print_available_memory()
    bm25_train_full = BM25Okapi(tokenized_train_data)
    print('finished loading bm25_train_full')
    print_available_memory()

    if not args.retrieval:
        prompt_type = ''
        random.shuffle(all_ques)
        for que in all_ques:
            prompt_type = prompt_type + "Question: " + que + "\nType of the question: "
            if que in selected_quest_compose:
                prompt_type += "Composition\n"
            else:
                prompt_type += "Comparison\n"
    else:
        prompt_type = ''
    with open(args.fb_roles_path) as f:
        lines = f.readlines()
    print('finished loading f.readlines')
    print_available_memory()

    relationships = []
    entities_set = []
    relationship_to_enti = {}
    for line in lines:
        info = line.split(" ")
        relationships.append(info[1])
        entities_set.append(info[0])
        entities_set.append(info[2])
        relationship_to_enti[info[1]] = [info[0], info[2]]
    print('finished loading everything in lines')
    print_available_memory()

    with open(args.surface_map_path) as f:
        lines = f.readlines()
    print('surface_map_path after f.readlines()')
    print(f'number of lines in surface_map_path: {len(lines)}')
    name_to_id_dict = {}

    # sl_cache_path = args.surface_lines_cache_path
    sl_cache_path = args.surface_lines_cache_path + (f'_{args.debug_nr_surface_lines}'
                                                     if args.debug_nr_surface_lines > -1 else f'_all')
    print(f'subsampling debug_nr_surface_lines to {args.debug_nr_surface_lines}')
    if os.path.exists(sl_cache_path):
        print(f'found in pickle: {sl_cache_path}, loading')
        name_to_id_dict = pickle.load(open(sl_cache_path, 'rb'))
        print(f'finished loading from pickle: {sl_cache_path}')
    else:
        print(f'NOT found in pickle: {sl_cache_path}, parsing')
        if args.debug_nr_surface_lines > -1:
            lines = random.sample(lines, k=args.debug_nr_surface_lines)

        print(f'reading {len(lines)} of surface into memory')
        for line in tqdm(lines):
            info = line.split("\t")
            name = info[0]
            score = float(info[1])
            mid = info[2].strip()
            if name in name_to_id_dict:
                name_to_id_dict[name][mid] = score
            else:
                name_to_id_dict[name] = {}
                name_to_id_dict[name][mid] = score
        print('finished parsing, saving to pickle')
        pickle.dump(name_to_id_dict, open(sl_cache_path, 'wb'))
        print('finished parsing, finished saving to pickle')
    # debug_nr_surface_lines
    print_available_memory('after surface_map_path')
    # exit()
    all_fns = list(name_to_id_dict.keys())
    print_available_memory('after all_fns')
    tokenized_all_fns = [fn.split() for fn in all_fns]
    print_available_memory('after tokenized_all_fns')
    bm25_all_fns = BM25Okapi(tokenized_all_fns)
    print_available_memory('after bm25_all_fns')
    print('about to run all_combiner_evaluation')
    # exit()
    all_combiner_evaluation(dev_data, selected_quest_compose, selected_quest_compare, selected_quest, prompt_type,
                            hsearcher, rela_corpus, relationships, args.temperature, que_to_s_dict_train,
                            question_to_mid_dict, args.api_key, args.engine, name_to_id_dict, bm25_all_fns,
                            all_fns, relationship_to_enti, retrieval=args.retrieval, corpus=corpus, nlp_model=nlp,
                            bm25_train_full=bm25_train_full, retrieve_number=args.shot_num,
                            is_cuda_available=is_cuda_available)


if __name__ == "__main__":
    main()
