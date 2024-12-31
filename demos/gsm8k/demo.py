from glue.promptopt.instantiate import GluePromptOpt
from glue.common_logic import DatasetSpecificProcessing
from glue.utils.file import save_jsonlist
from typing import Any
from tqdm import tqdm
from re import compile, findall
import os
from datasets import load_dataset

from dotenv import load_dotenv
load_dotenv(override = True)

class GSM8k(DatasetSpecificProcessing):

    def dataset_to_jsonl(self, dataset_jsonl: str, **kwargs: Any) -> None:
        def extract_answer_from_output(completion):
            # Your functions for metrics and prompt building
            ans_re = compile(r"#### (\-?[0-9\.\,]+)")
            self.INVALID_ANS = "[invalid]"

            match = ans_re.search(completion)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                return match_str
            else:
                return self.INVALID_ANS

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
        
        if not answer:
            return self.INVALID_ANS

        model_pred = answer.lower()
        preds = model_pred.split(self.ANSWER_START.lower())
        answer_flag = True if len(preds) > 1 else False

        pred = preds[-1].replace(",", "")
        pred = [s for s in findall(r'-?\d+\.?\d*', pred)]

        if len(pred) == 0:
            return self.INVALID_ANS

        if answer_flag:
            # choose the first element in list
            pred = pred[0]
        else:
            # choose the last element in list
            pred = pred[-1]

        # (For arithmetic tasks) if a word ends with period, it will be omitted ...
        if pred[-1] == ".":
            pred = pred[:-1]
        return pred




gsm8k_processor = GSM8k()


# ### Load and save the dataset 



if not os.path.exists("data"):
    os.mkdir("data")
    
dataset = load_dataset("openai/gsm8k", "main")
num_samples = 0
for dataset_type in ['train','test']:
    data_list = []
    for data in dataset[dataset_type]:
        data_list.append({"question": data['question'], "answer": data['answer']})
        if num_samples == 100 and dataset_type == 'train': # We sample only 100 train examples and use 25 out them for training randomly
            break
        num_samples += 1
    gsm8k_processor.dataset_to_jsonl("data/"+ dataset_type+'.jsonl', dataset=data_list)


# ### Set paths



train_file_name = os.path.join("data", "train.jsonl")
test_file_name = os.path.join("data", "test.jsonl")
path_to_config = "demos/gsm8k/configs"
promptopt_config_path = os.path.join(path_to_config, "promptopt_config.yaml")
setup_config_path = os.path.join(path_to_config, "setup_config.yaml")


# ### Create an object for calling prompt optimization and inference functionalities



gp = GluePromptOpt(promptopt_config_path,
                   setup_config_path,
                   train_file_name,
                   gsm8k_processor)


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

