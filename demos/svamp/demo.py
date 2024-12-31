#!/usr/bin/env python
# coding: utf-8

# #### Set environment variables in [.env](.env) for LLM API calling

# ### Import Dependencies



import sys
sys.path.insert(0, "../../")
import os
import glue
from glue.promptopt.instantiate import GluePromptOpt
from glue.common_logic import DatasetSpecificProcessing
from glue.utils.file import save_jsonlist
from typing import Any
from tqdm import tqdm
import json
from datasets import load_dataset

from dotenv import load_dotenv
load_dotenv(override = True)


# ### Create a dataset specific class and define the required functions 



def extract_between(start, end, text):
    """
    Extracts the substring from 'text' that is between 'start' and 'end' strings.
    
    Parameters:
    - start (str): The starting delimiter string.
    - end (str): The ending delimiter string.
    - text (str): The text to search within.
    
    Returns:
    - str: The extracted substring between the start and end delimiters.
    """
    start_index = text.find(start)
    if start_index == -1:
        return '' 
    
    start_index += len(start)
    
    end_index = text.find(end, start_index)
    if end_index == -1:
        return ''  
    return text[start_index:end_index]

class SVAMP(DatasetSpecificProcessing):

    def dataset_to_jsonl(self, dataset_jsonl: str, **kwargs: Any) -> None:
        def extract_answer_from_output(completion):

                return completion

        examples_set = []

        for _, sample in tqdm(enumerate(kwargs["dataset"]), desc="Evaluating samples"):
            example = {
              DatasetSpecificProcessing.QUESTION_LITERAL: sample['question'],
              DatasetSpecificProcessing.ANSWER_WITH_REASON_LITERAL: sample['answer'],
              DatasetSpecificProcessing.FINAL_ANSWER_LITERAL: extract_answer_from_output(sample["answer"])
            }
            examples_set.append(example)

        save_jsonlist(dataset_jsonl, examples_set, "w")

    def extract_final_answer(self, answer: str):
        
        final_answer = extract_between(text=answer,start="<ANS_START>",end="<ANS_END>")
        return final_answer
    
    def access_answer(self, llm_output: str, gt_answer: str):

        predicted_answer = self.extract_final_answer(llm_output)
        is_correct = False
        if predicted_answer and (predicted_answer.lower() == gt_answer.lower()):
            is_correct = True

        return is_correct, predicted_answer




svamp_processor = SVAMP()




if not os.path.exists("data"):
    os.mkdir("data")

dataset = load_dataset("ChilleD/SVAMP")

for dataset_type in ['train','test']:
    data_list = []
    num_samples = 0
    for data in dataset[dataset_type]:
        data_list.append({"question": data['question_concat'], "answer": data['Answer']})
        if dataset_type == 'train' and num_samples == 100: # We sample only 100 train examples and use 25 out them for training randomly
            break
        num_samples += 1
    svamp_processor.dataset_to_jsonl("data/"+ dataset_type+'.jsonl', dataset=data_list)


# ### Set paths



train_file_name = os.path.join("data", "train.jsonl")
test_file_name = os.path.join("data", "test.jsonl")
path_to_config = "demos/svamp/configs"
llm_config_path = os.path.join(path_to_config, "llm_config.yaml")
promptopt_config_path = os.path.join(path_to_config, "promptopt_config.yaml")
setup_config_path = os.path.join(path_to_config, "setup_config.yaml")


# ### Create an object for calling prompt optimization and inference functionalities



gp = GluePromptOpt(promptopt_config_path,
                   setup_config_path,
                   train_file_name,
                   svamp_processor)


# ### Call prompt optmization function
# 1. ```use_examples``` can be used when there are training samples and a mixture of real and synthetic in-context examples are required in the final prompt. When set to ```False``` all the in-context examples will be real
# 2. ```generate_synthetic_examples``` can be used when there are no training samples and we want to generate synthetic examples 
# 3. ```run_without_train_examples``` can be used when there are no training samples and in-context examples are not required in the final prompt 



# Function call to generate optimal prompt and expert profile 
best_prompt, expert_profile = gp.get_best_prompt(use_examples=True,run_without_train_examples=False,generate_synthetic_examples=False)


# ### Save the optimized prompt and expert profile



import pickle 

if not os.path.exists("results"):
    os.system("mkdir results")

with open("results/best_prompt.pkl", 'wb') as f:
    pickle.dump(best_prompt, f)
with open("results/expert_profile.pkl", 'wb') as f:
    pickle.dump(expert_profile, f)

print(f"Best prompt: {best_prompt} \nExpert profile: {expert_profile}")


# ### Evaluate the optimized prompt



gp.EXPERT_PROFILE = expert_profile
gp.BEST_PROMPT = best_prompt

# Function call to evaluate the prompt
accuracy = gp.evaluate(test_file_name)

print(f"Final Accuracy: {accuracy}")

