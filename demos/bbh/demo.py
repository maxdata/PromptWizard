from glue.promptopt.instantiate import GluePromptOpt
from glue.promptopt.techniques.common_logic import DatasetSpecificProcessing
from glue.common.utils.file import save_jsonlist
from typing import Any
from tqdm import tqdm
import json
import os
from azure.identity import get_bearer_token_provider, AzureCliCredential
from openai import AzureOpenAI

from dotenv import load_dotenv
load_dotenv(override = True)


# ### Below code can be used for LLM-as-a-judge eval



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

def call_api(messages):
    
    token_provider = get_bearer_token_provider(
            AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
        )
    client = AzureOpenAI(
        api_version="<OPENAI_API_VERSION>",
        azure_endpoint="<AZURE_ENDPOINT>",
        azure_ad_token_provider=token_provider
        )
    response = client.chat.completions.create(
        model="<MODEL_DEPLOYMENT_NAME>",
        messages=messages,
        temperature=0.0,
    )
    prediction = response.choices[0].message.content
    return prediction

def llm_eval(predicted_answer,gt_answer):
    
    EVAL_PROMPT = f"""Given the Predicted_Answer and Reference_Answer, compare them and check they mean the same.
                    If they mean the same then return True between <ANS_START> and <ANS_END> tags , 
                    If they differ in the meaning then return False between <ANS_START> and <ANS_END> tags 
                    Following are the given :
                    Predicted_Answer: {predicted_answer}
                    Reference_Answer: {gt_answer}"""
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": EVAL_PROMPT}
    ]

    response = call_api(messages)
    final_judgement = extract_between(start="<ANS_START>", end="<ANS_END>", text=response)
    return final_judgement == "True"


# ### Create a dataset specific class and define the required functions 



llm_as_judge_eval = True

class BBH(DatasetSpecificProcessing):

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

        if llm_as_judge_eval:
            predicted_answer = self.extract_final_answer(llm_output)
            is_correct = False
            if llm_eval(predicted_answer,gt_answer):
                is_correct = True
        else:
            predicted_answer = self.extract_final_answer(llm_output)
            is_correct = False
            if predicted_answer and (predicted_answer.lower() == gt_answer.lower()):
                is_correct = True

            return is_correct, predicted_answer




bbh_processor = BBH()


# ### Load and save the dataset . 
# Set the ```dataset_to_run``` variable to choose 1 among the 19 datasets of BBII to run the optimization on



if not os.path.exists("data"):
    os.mkdir("data")
dataset_list = ['informal_to_formal','letters_list','negation','orthography_starts_with','rhymes','second_word_letter','sum','diff','sentence_similarity','taxonomy_animal','auto_categorization','object_counting','odd_one_out','antonyms','word_unscrambling','cause_and_effect','common_concept','word_sorting','synonyms']

# Set the dataset on which to run optimization out of the 19 
dataset_to_run = 'second_word_letter'

if not os.path.exists("data/"+dataset_to_run):
    os.mkdir("data/"+dataset_to_run)
    
os.system("git clone https://github.com/xqlin98/INSTINCT")


for mode in ['execute','induce']:
    for dataset in dataset_list:

        if dataset_to_run == dataset:
            data_list = []

            file_path = 'INSTINCT/Induction/experiments/data/instruction_induction/raw/'+mode+'/'+dataset+'.json'  
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            save_file_path = 'test.jsonl'
            if mode == 'execute':
                save_file_path = 'train.jsonl'

            for key,sample in data['examples'].items():
                task = dataset
                if(task == 'cause_and_effect'):
                    cause = sample["cause"]
                    effect = sample["effect"]
                    import random
                    pair = [cause, effect]
                    random.shuffle(pair)
                    question = f"Sentence 1: {pair[0]} Sentence 2: {pair[1]}",
                    answer = cause,
                elif(task == 'antonyms'):
                    
                        question = sample["input"],
                        answer = sample["output"],

                elif(task == 'common_concept'):
                    concept = sample["concept"]
                    items = sample["items"]
                    input = ", ".join(items)
                    question = f"Objects: {input}"
                    answer = f"{concept}"

                elif(task == 'diff'):
                    input = sample["input"]
                    output = sample["output"]
                    question = f"{input}"
                    answer = f"{output}"

                elif(task == 'informal_to_formal'):
                    informal = sample["input"]
                    formal = sample["output"]
                    question = f"{informal}"
                    answer = f"{formal}"

                elif(task == 'synonyms' or task == 'word_unscrambling' or task == 'word_sorting' or task == 'letters_list' or task == 'negation' or task == 'orthography_starts_with' or task == 'second_word_letter' or task == 'sentence_similarity' or task == 'sum' or task == 'taxonomy_animal' or task == 'auto_categorization' or task == 'object_counting' or task == 'odd_one_out'):
                    informal = sample["input"]
                    formal = sample["output"] 
                    question = f"{informal}"
                    answer = f"{formal}"

                elif(task == 'rhymes'):
                    input = sample["input"]
                    output = sample["other_rhymes"]
                    output = ", ".join(output)
                    question = f"{input}"
                    answer = f"{output}"
            
                data_list.append({"question":question,"answer":answer})
            bbh_processor.dataset_to_jsonl("data/"+dataset +"/"+save_file_path, dataset=data_list)

os.system("rm -r INSTINCT")
           


# ### Set paths



train_file_name = os.path.join("data/"+dataset_to_run, "train.jsonl")
test_file_name = os.path.join("data/"+dataset_to_run, "test.jsonl")
path_to_config = "configs"
llm_config_path = os.path.join(path_to_config, "llm_config.yaml")
promptopt_config_path = os.path.join(path_to_config, "promptopt_config.yaml")
setup_config_path = os.path.join(path_to_config, "setup_config.yaml")


# ### Create an object for calling prompt optimization and inference functionalities



gp = GluePromptOpt(promptopt_config_path,
                   setup_config_path,
                   train_file_name,
                   bbh_processor)


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

