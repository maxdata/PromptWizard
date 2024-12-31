from glue.promptopt.instantiate import GluePromptOpt
from glue.common_logic import DatasetSpecificProcessing
from glue.utils.file import save_jsonlist
from typing import Any
from tqdm import tqdm
from re import compile, findall
import os
from datasets import load_dataset
import yaml
from dotenv import load_dotenv
load_dotenv(override = True)

def update_yaml_file(file_path,config_dict):

    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)


    for field,value in config_dict.items():
        data[field] = value

    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    print("YAML file updated successfully!")

path_to_config = "configs"
promptopt_config_path = os.path.join(path_to_config, "promptopt_config.yaml")
setup_config_path = os.path.join(path_to_config, "setup_config.yaml")

# ### Now let us consider the three scenarios with respect to availability of training data

# #### Scenario 1 : We have no training data , but we also don't want in-context examples in final prompt

# Set the configurations to generate mutations



file_path = 'configs/promptopt_config.yaml' 
# Set the following based on the use case
config_dict = {
                "task_description": "You are a mathematics expert. You will be given a mathematics problem which you need to solve",
                "base_instruction": "Lets think step by step.",
                "mutation_rounds": 5
               }
update_yaml_file(file_path,config_dict)


# Create an object for calling prompt optimization and inference functionalities



gp = GluePromptOpt(promptopt_config_path,
                   setup_config_path,
                   dataset_jsonl=None,
                   data_processor=None)


# Call the optimization function



best_prompt, expert_profile = gp.get_best_prompt(use_examples=False,run_without_train_examples=True,generate_synthetic_examples=False)


# Output : Five mutated prompts are printed on the termial as shown below :



OUTPUT = """
Variations 1:
Expert Profile:
You are a mathematician with a strong background in various fields of mathematics, including algebra, calculus, geometry, and statistics. You have a deep understanding of mathematical theories and principles, and you are skilled at solving complex problems with precision and clarity. Your expertise allows you to approach mathematical problems methodically, breaking them down into manageable steps and applying appropriate techniques to find solutions. You are familiar with both theoretical and applied mathematics, and you can explain your reasoning and solutions in a clear and concise manner. Your ability to solve mathematical problems efficiently and accurately makes you an invaluable resource for anyone seeking help with mathematics.:
Prompt:
You are a mathematics expert. You will be given a mathematics problem which you need to solve
Lets think step by step.


For each question present the reasoning followed by the correct answer.
Keywords: mathematics, problem-solving, step-by-step, logical reasoning, expert
_______________________________________________________________________

Variations 2:
Expert Profile:
You are a mathematician with a strong background in various fields of mathematics, including algebra, calculus, geometry, and statistics. You have a deep understanding of mathematical theories and principles, and you are skilled at solving complex problems with precision and clarity. Your expertise allows you to approach mathematical problems methodically, breaking them down into manageable steps and applying appropriate techniques to find solutions. You are familiar with both theoretical and applied mathematics, and you can explain your reasoning and solutions in a clear and concise manner. Your ability to solve mathematical problems efficiently and accurately makes you an invaluable resource for anyone seeking help with mathematics.:
Prompt:
Let's break this problem down step by step and devise an experiment to help solve it.


For each question present the reasoning followed by the correct answer.
Keywords: mathematics, problem-solving, step-by-step, logical reasoning, expert
_______________________________________________________________________

Variations 3:
Expert Profile:
You are a mathematics expert with a strong background in various fields of mathematics, including algebra, calculus, geometry, and statistics. You have a deep understanding of mathematical theories and principles, and you are skilled at solving complex problems with precision and clarity. Your expertise allows you to break down intricate problems into manageable steps, making it easier to find solutions. You are familiar with a wide range of mathematical techniques and tools, and you can apply them effectively to solve problems. Whether the problem involves solving equations, proving theorems, or analyzing data, you can provide a clear and accurate solution. Your ability to explain your reasoning and methodology ensures that others can follow and understand your approach, making you an invaluable resource for tackling challenging mathematical problems.:
Prompt:
Let's think through this problem step by step and make a list of ideas to solve it.


For each question present the reasoning followed by the correct answer.
Keywords: mathematics, problem-solving, step-by-step, logical reasoning, expert
_______________________________________________________________________

Variations 4:
Expert Profile:
You are a mathematics expert with a strong background in various fields of mathematics, including algebra, calculus, geometry, and statistics. You have a deep understanding of mathematical theories and principles, and you are skilled at solving complex problems with precision and clarity. Your expertise allows you to break down intricate problems into manageable steps, making it easier for others to follow your reasoning. You are familiar with a wide range of mathematical techniques and tools, and you can apply them effectively to find solutions. Whether the problem involves solving equations, proving theorems, or analyzing data, you can provide a clear, accurate, and well-explained solution. Your ability to communicate complex mathematical concepts in an understandable way makes you an invaluable resource for anyone seeking to solve mathematical problems.:
Prompt:
Let's approach this problem step by step and measure our progress as we go.


For each question present the reasoning followed by the correct answer.
Keywords: mathematics, problem-solving, step-by-step, logical reasoning, expert
Iterations completed:   0%|          | 0/3 [00:24<?, ?it/s]
Time taken to find best prompt: 24.79972267150879 sec
_______________________________________________________________________

Variations 5:
Expert Profile:
You are a mathematics expert with a strong background in various fields of mathematics, including algebra, calculus, geometry, and statistics. You have a deep understanding of mathematical theories and principles, and you are skilled at solving complex problems with precision and clarity. Your expertise allows you to approach problems methodically, breaking them down into manageable steps and applying appropriate mathematical techniques to find solutions. You are also adept at explaining your reasoning and methods in a clear and concise manner, making it easy for others to follow your thought process. Whether the problem involves solving equations, proving theorems, or analyzing data, you have the knowledge and skills to tackle it effectively. Your proficiency in mathematics is highly valuable in both academic and practical applications, and you are well-equipped to provide accurate and insightful solutions to a wide range of mathematical problems.:
Prompt:
Let's simplify this problem step by step to make it easier to solve.


For each question present the reasoning followed by the correct answer.
Keywords: mathematics, problem-solving, step-by-step, logical reasoning, expert"""


# #### Scenario 2 : We have no training data , but we also want in-context examples in final prompt

# This scenario has two steps 
# - Genrate synthetic data
# - Optimize prompts using synthetic data

# STEP 1 : Generate synthetic data

# Set the configurations to first generate synthetic training data. \
# Any number of synthetic examples can be generated and then used for optimizing prompts as mentioned in STEP 2



file_path = 'configs/promptopt_config.yaml' 
# Set the number of synthetic training examples to be generated
config_dict = {
                "num_train_examples":20
               }
update_yaml_file(file_path,config_dict)




gp = GluePromptOpt(promptopt_config_path,
                   setup_config_path,
                   dataset_jsonl=None,
                   data_processor=None)


# Call the function to generate synthetic examples, which are saved in train.jsonl



best_prompt, expert_profile = gp.get_best_prompt(use_examples=False,run_without_train_examples=False,generate_synthetic_examples=True)


# STEP 2 : Optimize prompts using synthetic data

# Create a dataset specific class and define the required functions 



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


# Set the configurations to optimize the prompt on the synthetic data



file_path = 'configs/promptopt_config.yaml' 
config_dict = {
                "task_description": "You are a mathematics expert. You will be given a mathematics problem which you need to solve",
                "base_instruction": "Lets think step by step.",
                "mutation_rounds": 2,
                "few_shot_count": 5,
                "generate_reasoning": True,
                "mutate_refine_iterations" : 3,
                "seen_set_size":20
               }
update_yaml_file(file_path,config_dict)


# Call the optimization function 



gp = GluePromptOpt(promptopt_config_path,
                   setup_config_path,
                   dataset_jsonl = "train_synthetic.jsonl",
                   data_processor=gsm8k_processor)
best_prompt, expert_profile = gp.get_best_prompt(use_examples=True,run_without_train_examples=False,generate_synthetic_examples=False)


# Output : Following Prompt and Expert Profile are generated 



OUTPUT = """
Generating Expert Identity....
Expert Identity: You are a mathematician with a strong background in various fields of mathematics, including algebra, calculus, geometry, and statistics. You have a deep understanding of mathematical theories and principles, and you are skilled at solving complex problems with precision and clarity. Your analytical skills and logical reasoning enable you to break down problems into manageable steps and find accurate solutions efficiently. You are familiar with a wide range of mathematical techniques and tools, and you can apply them to solve problems in both theoretical and applied contexts. Your expertise allows you to explain your solutions clearly and concisely, making complex concepts accessible to others. Whether the problem involves solving equations, proving theorems, or analyzing data, you are well-equipped to provide a thorough and correct solution.
Final best prompt: Provide a clear and detailed solution, breaking down all necessary steps. Ensure that the final answer is clearly marked and separated from the solution steps. Use proper mathematical notation and formatting throughout. Verify the final answer by checking the solution steps for accuracy. Simplify all expressions and fractions where possible. Handle special cases or edge cases appropriately, and clearly state any assumptions or conditions applied during the solution process. Finally, review the entire solution to ensure logical consistency and correct formatting.

[Question] Solve for \( x \) in the equation \( 2x + 3 = 11 \).
[Answer] To solve for \( x \) in the equation \( 2x + 3 = 11 \), we will follow these steps:

1. **Isolate the term containing \( x \)**:
   We start by isolating the term with \( x \) on one side of the equation. To do this, we need to eliminate the constant term on the left side of the equation.

   \[
   2x + 3 = 11
   \]

   Subtract 3 from both sides of the equation:

   \[
   2x + 3 - 3 = 11 - 3
   \]

   Simplifying this, we get:

   \[
   2x = 8
   \]

2. **Solve for \( x \)**:
   Now, we need to solve for \( x \) by isolating \( x \) itself. Since \( x \) is multiplied by 2, we will divide both sides of the equation by 2 to solve for \( x \).

   \[
   \frac{2x}{2} = \frac{8}{2}
   \]

   Simplifying this, we get:

   \[
   x = 4
   \]

3. **Verify the solution**:
   To ensure our solution is correct, we substitute \( x = 4 \) back into the original equation and check if both sides are equal.

   Original equation:

   \[
   2x + 3 = 11
   \]

   Substitute \( x = 4 \):

   \[
   2(4) + 3 = 11
   \]

   Simplifying this, we get:

   \[
   8 + 3 = 11
   \]

   \[
   11 = 11
   \]

   Since both sides of the equation are equal, our solution is verified to be correct.

**Final Answer**: \( x = 4 \) <ANS_START> \( x = 4 \) <ANS_END>

[Question] Solve for \( x \) in the equation \( x^2 - 4x + 4 = 0 \).
[Answer] To solve the quadratic equation \( x^2 - 4x + 4 = 0 \), we will follow these steps:

1. **Identify the quadratic equation**: The given equation is \( x^2 - 4x + 4 = 0 \).

2. **Recognize the standard form**: The standard form of a quadratic equation is \( ax^2 + bx + c = 0 \). Here, \( a = 1 \), \( b = -4 \), and \( c = 4 \).

3. **Factor the quadratic expression**: We need to factor the quadratic expression on the left-hand side of the equation. We look for two numbers that multiply to \( c \) (which is 4) and add up to \( b \) (which is -4). These numbers are -2 and -2.

4. **Write the factored form**: The quadratic expression \( x^2 - 4x + 4 \) can be factored as \( (x - 2)(x - 2) \) or \( (x - 2)^2 \).

5. **Set the factored form equal to zero**: We now have \( (x - 2)^2 = 0 \).

6. **Solve for \( x \)**: To find the value of \( x \), we take the square root of both sides of the equation:
   \[
   \sqrt{(x - 2)^2} = \sqrt{0}
   \]
   This simplifies to:
   \[
   x - 2 = 0
   \]

7. **Isolate \( x \)**: Add 2 to both sides of the equation to solve for \( x \):
   \[
   x = 2
   \]

8. **Verify the solution**: Substitute \( x = 2 \) back into the original equation to ensure it satisfies the equation:
   \[
   (2)^2 - 4(2) + 4 = 4 - 8 + 4 = 0
   \]
   Since the left-hand side equals the right-hand side (0), the solution \( x = 2 \) is verified.

**Final Answer**: \( x = 2 \) <ANS_START> \( x = 2 \) <ANS_END>

[Question] Find the derivative of \( f(x) = 3x^2 \cdot \sin(x) \).
[Answer] To find the derivative of the function \( f(x) = 3x^2 \cdot \sin(x) \), we will use the product rule of differentiation. The product rule states that if we have a function \( f(x) = u(x) \cdot v(x) \), then its derivative \( f'(x) \) is given by:

\[ f'(x) = u'(x) \cdot v(x) + u(x) \cdot v'(x) \]

Here, we identify \( u(x) = 3x^2 \) and \( v(x) = \sin(x) \).

Step 1: Differentiate \( u(x) = 3x^2 \)
\[ u'(x) = \frac{d}{dx}(3x^2) = 3 \cdot 2x = 6x \]

Step 2: Differentiate \( v(x) = \sin(x) \)
\[ v'(x) = \frac{d}{dx}(\sin(x)) = \cos(x) \]

Step 3: Apply the product rule
\[ f'(x) = u'(x) \cdot v(x) + u(x) \cdot v'(x) \]
\[ f'(x) = (6x) \cdot \sin(x) + (3x^2) \cdot \cos(x) \]

Step 4: Simplify the expression
\[ f'(x) = 6x \sin(x) + 3x^2 \cos(x) \]

Thus, the derivative of the function \( f(x) = 3x^2 \cdot \sin(x) \) is:

\[ \boxed{f'(x) = 6x \sin(x) + 3x^2 \cos(x)} \]

To verify the final answer, we can recheck each step to ensure accuracy:
- The derivative of \( 3x^2 \) is correctly calculated as \( 6x \).
- The derivative of \( \sin(x) \) is correctly calculated as \( \cos(x) \).
- The product rule is correctly applied, and the terms are correctly combined and simplified.

Therefore, the final answer is confirmed to be correct. <ANS_START> \( f'(x) = 3x^2 \cos(x) + 6x \sin(x) \) <ANS_END>

[Question] Evaluate the definite integral \( \int_{0}^{1} (4x^3 - 2x + 1) \, dx \).
[Answer] To evaluate the definite integral \( \int_{0}^{1} (4x^3 - 2x + 1) \, dx \), we will follow these steps:

1. **Find the antiderivative** of the integrand \( 4x^3 - 2x + 1 \).
2. **Evaluate the antiderivative** at the upper limit of integration (1).
3. **Evaluate the antiderivative** at the lower limit of integration (0).
4. **Subtract the value** of the antiderivative at the lower limit from the value at the upper limit to find the definite integral.

### Step-by-Step Solution:

1. **Find the antiderivative**:
   - The antiderivative of \( 4x^3 \) is \( \frac{4x^4}{4} = x^4 \).
   - The antiderivative of \( -2x \) is \( -\frac{2x^2}{2} = -x^2 \).
   - The antiderivative of \( 1 \) is \( x \).

   Therefore, the antiderivative of \( 4x^3 - 2x + 1 \) is:
   \[
   F(x) = x^4 - x^2 + x
   \]

2. **Evaluate the antiderivative at the upper limit (1)**:
   \[
   F(1) = 1^4 - 1^2 + 1 = 1 - 1 + 1 = 1
   \]

3. **Evaluate the antiderivative at the lower limit (0)**:
   \[
   F(0) = 0^4 - 0^2 + 0 = 0
   \]

4. **Subtract the value at the lower limit from the value at the upper limit**:
   \[
   \int_{0}^{1} (4x^3 - 2x + 1) \, dx = F(1) - F(0) = 1 - 0 = 1
   \]

### Final Answer:
\[
\boxed{1}
\] <ANS_START> \( 1 \) <ANS_END>

[Question] Solve the system of equations:
\[ \begin{cases} 
x + 2y + z = 6 \\
2x - y + 3z = 14 \\
3x + y - z = 2 
\end{cases} \]
[Answer] To solve the system of equations:
\[ \begin{cases} 
x + 2y + z = 6 \\
2x - y + 3z = 14 \\
3x + y - z = 2 
\end{cases} \]

we will use the method of elimination and substitution to find the values of \(x\), \(y\), and \(z\).

**Step 1: Eliminate \(z\) from the first two equations.**

First, we multiply the first equation by 3 to align the coefficients of \(z\):
\[ 3(x + 2y + z) = 3 \cdot 6 \]
\[ 3x + 6y + 3z = 18 \]

Now, we subtract the second equation from this result:
\[ (3x + 6y + 3z) - (2x - y + 3z) = 18 - 14 \]
\[ 3x + 6y + 3z - 2x + y - 3z = 4 \]
\[ x + 7y = 4 \]
\[ \text{(Equation 4)} \]

**Step 2: Eliminate \(z\) from the first and third equations.**

Next, we multiply the first equation by 1 and the third equation by 1 to align the coefficients of \(z\):
\[ 1(x + 2y + z) = 1 \cdot 6 \]
\[ x + 2y + z = 6 \]

\[ 1(3x + y - z) = 1 \cdot 2 \]
\[ 3x + y - z = 2 \]

Now, we add these two equations:
\[ (x + 2y + z) + (3x + y - z) = 6 + 2 \]
\[ x + 2y + z + 3x + y - z = 8 \]
\[ 4x + 3y = 8 \]
\[ \text{(Equation 5)} \]

**Step 3: Solve the system of equations formed by Equation 4 and Equation 5.**

We now have:
\[ \begin{cases} 
x + 7y = 4 \\
4x + 3y = 8 
\end{cases} \]

First, we solve Equation 4 for \(x\):
\[ x = 4 - 7y \]

Substitute \(x = 4 - 7y\) into Equation 5:
\[ 4(4 - 7y) + 3y = 8 \]
\[ 16 - 28y + 3y = 8 \]
\[ 16 - 25y = 8 \]
\[ -25y = 8 - 16 \]
\[ -25y = -8 \]
\[ y = \frac{8}{25} \]

**Step 4: Substitute \(y\) back into Equation 4 to find \(x\).**

\[ x + 7\left(\frac{8}{25}\right) = 4 \]
\[ x + \frac{56}{25} = 4 \]
\[ x = 4 - \frac{56}{25} \]
\[ x = \frac{100}{25} - \frac{56}{25} \]
\[ x = \frac{44}{25} \]

**Step 5: Substitute \(x\) and \(y\) back into the first original equation to find \(z\).**

\[ x + 2y + z = 6 \]
\[ \frac{44}{25} + 2\left(\frac{8}{25}\right) + z = 6 \]
\[ \frac{44}{25} + \frac{16}{25} + z = 6 \]
\[ \frac{60}{25} + z = 6 \]
\[ \frac{60}{25} = 2.4 \]
\[ 2.4 + z = 6 \]
\[ z = 6 - 2.4 \]
\[ z = 3.6 \]

**Final Answer:**
\[ x = \frac{44}{25}, y = \frac{8}{25}, z = 3.6 \]

We have verified each step and simplified all expressions. The solution is logically consistent and correctly formatted. <ANS_START> \( x = \frac{44}{25}, y = \frac{8}{25}, z = 3.6 \) <ANS_END>


For each question present the reasoning followed by the correct answer.
"""


# #### Scenario 3 : We have training data and also want in-context examples in final prompt

# Load and save the dataset 



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


# Set the configurations



file_path = 'configs/promptopt_config.yaml' 
config_dict = {
                "task_description": "You are a mathematics expert. You will be given a mathematics problem which you need to solve",
                "base_instruction": "Lets think step by step.",
                "mutation_rounds": 2,
                "few_shot_count": 5,
                "generate_reasoning": True,
                "mutate_refine_iterations" : 3,
                "seen_set_size":20
               }
update_yaml_file(file_path,config_dict)


# Create an object for calling prompt optimization and inference functionalities



gp = GluePromptOpt(promptopt_config_path,
                   setup_config_path,
                   dataset_jsonl = os.path.join("data", "train.jsonl"),
                   data_processor = gsm8k_processor)


# Call the optimization function 



best_prompt, expert_profile = gp.get_best_prompt(use_examples=True,run_without_train_examples=False,generate_synthetic_examples=False)


# Output : Following Prompt and Expert Profile are generated 



OUTPUT = """Expert Identity: You are a mathematics expert with a strong background in various fields of mathematics, including algebra, calculus, geometry, and statistics. You have a deep understanding of mathematical theories and principles, and you are skilled at solving complex problems with precision and clarity. Your expertise allows you to break down intricate problems into manageable steps, making it easier for others to follow your reasoning. You are familiar with a wide range of mathematical techniques and tools, and you can apply them effectively to find solutions. Whether the problem involves solving equations, proving theorems, or analyzing data, you can provide a clear, accurate, and well-explained solution. Your ability to communicate complex mathematical concepts in an understandable way makes you an invaluable resource for anyone seeking help with mathematics.

Final best prompt: 

You are a mathematics expert. Your task is to solve a given mathematics problem accurately and provide a clear, detailed explanation of your solution process. Follow these steps to ensure a comprehensive and well-structured solution:

1. **Understand the Problem**: Carefully read and comprehend the problem statement. Identify the key components and what is being asked.

2. **Identify Components**: Break down the problem into its fundamental components, such as variables, constants, and relevant quantities (e.g., base pay, overtime pay, distances, speeds, etc.).

3. **Apply Relevant Principles**: Use appropriate mathematical principles, formulas, and methods to solve the problem step by step.

4. **Logical Reasoning**: Employ logical reasoning to explain each step of your solution process. Ensure that each step follows logically from the previous one.

5. **Detailed Explanations**: Provide detailed explanations for each step to ensure clarity and understanding. Include intermediate results and how they contribute to the final solution.

6. **Explicit Calculation Steps**: Show each calculation step in detail, including intermediate results. Use proper mathematical notation and symbols.

7. **Verify Each Step**: Recheck each intermediate step of your calculation to verify the correctness of the final answer. Ensure that all arithmetic and algebraic operations are accurate.

8. **Combine Results**: Clearly combine different components of the problem (e.g., base pay and overtime pay) before arriving at the final answer.

9. **Simplify and Notate**: Simplify the final answer where possible, and use proper mathematical notation and symbols.

10. **Mark the Final Answer**: Clearly mark the final answer within <ANS_START> and <ANS_END> tags.

Ensure that your approach is tailored to the specific type of mathematical problem being solved, whether it involves arithmetic, algebra, geometry, calculus, or any other area of mathematics. Present the solutions in a clear and organized manner.

**Additional Guidelines:**
- **Contextual Understanding**: Pay close attention to the context of the problem to ensure that all relationships and quantities are correctly interpreted.
- **Correct Application of Arithmetic Operations**: Double-check that all arithmetic operations are applied correctly and align with the problem's requirements.
- **Verification of Final Answer**: Dedicate a step to verify the final answer by rechecking all intermediate steps and ensuring they logically lead to the correct final result.
- **Clarity in Marking Final Answer**: Use the <ANS_START> and <ANS_END> tags to clearly mark the final answer.

By following these steps and additional guidelines, you will ensure that the solution is accurate, well-explained, and clearly presented.


[Question] Bella bought stamps at the post office. Some of the stamps had a snowflake design, some had a truck design, and some had a rose design. Bella bought 11 snowflake stamps. She bought 9 more truck stamps than snowflake stamps, and 13 fewer rose stamps than truck stamps. How many stamps did Bella buy in all?
[Answer] 1. **Understand the Problem**: Bella bought three types of stamps: snowflake, truck, and rose. We need to determine the total number of stamps she bought, given the relationships between the quantities of each type.

2. **Identify Components**:
   - Number of snowflake stamps: 11.
   - Number of truck stamps: 9 more than the number of snowflake stamps.
   - Number of rose stamps: 13 fewer than the number of truck stamps.

3. **Apply Relevant Principles**: Use basic arithmetic operations to find the quantities of truck and rose stamps, and then sum all the quantities to find the total number of stamps.

4. **Logical Reasoning**:
   - Number of snowflake stamps: 11.
   - Number of truck stamps: 11 (snowflake stamps) + 9 = 20.
   - Number of rose stamps: 20 (truck stamps) - 13 = 7.

5. **Detailed Explanations**:
   - Calculate the number of truck stamps: 11 (snowflake stamps) + 9 = 20.
   - Calculate the number of rose stamps: 20 (truck stamps) - 13 = 7.
   - Calculate the total number of stamps: 11 (snowflake) + 20 (truck) + 7 (rose) = 38.

6. **Explicit Calculation Steps**:
   - Truck stamps: 11 + 9 = $<11+9=20>20.
   - Rose stamps: 20 - 13 = $<20-13=7>7.
   - Total stamps: 11 + 20 + 7 = $<11+20+7=38>38.

7. **Verify Each Step**: Recheck each calculation step to ensure correctness:
   - Truck stamps: 11 + 9 = 20.
   - Rose stamps: 20 - 13 = 7.
   - Total stamps: 11 + 20 + 7 = 38.

8. **Combine Results**: Combine the number of each type of stamp correctly to find the total number of stamps.

9. **Simplify and Notate**: The final answer is already simplified.

10. **Mark the Final Answer**: <ANS_START>38<ANS_END>

By following these steps, we ensure that the solution is accurate, well-explained, and clearly presented. <ANS_START>38<ANS_END>

[Question] It takes Roque two hours to walk to work and one hour to ride his bike to work. Roque walks to and from work three times a week and rides his bike to and from work twice a week. How many hours in total does he take to get to and from work a week with walking and biking?
[Answer] 1. **Understand the Problem**: Roque has two modes of transportation to work: walking and biking. We need to calculate the total time he spends traveling to and from work in a week, considering the different times and frequencies for each mode.

2. **Identify Components**:
   - Time to walk to work: 2 hours (one way).
   - Time to bike to work: 1 hour (one way).
   - Frequency of walking: 3 times a week (to and from work).
   - Frequency of biking: 2 times a week (to and from work).

3. **Apply Relevant Principles**: Use basic arithmetic to calculate the total time spent walking and biking separately, then sum these times to get the total weekly travel time.

4. **Logical Reasoning**:
   - Calculate the total walking time for a week:
     - One round trip (to and from work) by walking takes 2 hours (to work) + 2 hours (from work) = 4 hours.
     - Roque walks to and from work 3 times a week, so the total walking time is 4 hours per round trip * 3 round trips = 12 hours.
   - Calculate the total biking time for a week:
     - One round trip (to and from work) by biking takes 1 hour (to work) + 1 hour (from work) = 2 hours.
     - Roque bikes to and from work 2 times a week, so the total biking time is 2 hours per round trip * 2 round trips = 4 hours.

5. **Detailed Explanations**:
   - Walking time calculation:
     - One round trip walking: 2 hours (to work) + 2 hours (from work) = 4 hours.
     - Total walking time for the week: 4 hours per round trip * 3 round trips = 12 hours.
   - Biking time calculation:
     - One round trip biking: 1 hour (to work) + 1 hour (from work) = 2 hours.
     - Total biking time for the week: 2 hours per round trip * 2 round trips = 4 hours.
   - Combine the total walking and biking times to get the total weekly travel time:
     - Total weekly travel time: 12 hours (walking) + 4 hours (biking) = 16 hours.

6. **Explicit Calculation Steps**:
   - Walking time: 2 hours (one way) * 2 (round trip) * 3 (times a week) = $<2*2*3=12>12 hours.
   - Biking time: 1 hour (one way) * 2 (round trip) * 2 (times a week) = $<1*2*2=4>4 hours.
   - Total time: 12 hours (walking) + 4 hours (biking) = $<12+4=16>16 hours.

7. **Verify Each Step**: Recheck each calculation step to ensure correctness. Confirm that the arithmetic operations and logic used are accurate.

8. **Combine Results**: Combine the total walking and biking times correctly to ensure the final answer is accurate.

9. **Simplify and Notate**: The final answer is already simplified and clearly presented.

10. **Mark the Final Answer**: <ANS_START>16<ANS_END>

By following these steps, we ensure that the solution is accurate, well-explained, and clearly presented. <ANS_START>16<ANS_END>

[Question] Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
[Answer] 1. **Understand the Problem**: Betty is saving money for a wallet that costs $100. She currently has half of the money she needs. Her parents and grandparents are contributing additional amounts to help her reach her goal. We need to determine how much more money Betty needs to buy the wallet.

2. **Identify Components**:
   - Total cost of the wallet: $100.
   - Amount Betty currently has: half of $100.
   - Contribution from parents: $15.
   - Contribution from grandparents: twice the amount given by parents.

3. **Apply Relevant Principles**: Use basic arithmetic to calculate the total amount of money Betty will have after receiving contributions from her parents and grandparents, and then determine how much more she needs to reach $100.

4. **Logical Reasoning**:
   - Calculate the amount Betty currently has: $100 / 2 = $50.
   - Calculate the contribution from grandparents: 2 * $15 = $30.
   - Calculate the total amount of money Betty will have: $50 (current amount) + $15 (parents' contribution) + $30 (grandparents' contribution).

5. **Detailed Explanations**:
   - Betty currently has $50 because half of $100 is $50.
   - Her parents give her $15.
   - Her grandparents give her twice the amount her parents give, which is 2 * $15 = $30.
   - The total amount of money Betty will have is $50 (current amount) + $15 (parents' contribution) + $30 (grandparents' contribution) = $95.

6. **Explicit Calculation Steps**:
   - Current amount: $100 / 2 = $<100/2=50>50.
   - Grandparents' contribution: 2 * $15 = $<2*15=30>30.
   - Total amount: $50 + $15 + $30 = $<50+15+30=95>95.

7. **Verify Each Step**: Recheck each calculation step to ensure correctness.
   - Current amount: $100 / 2 = $50.
   - Grandparents' contribution: 2 * $15 = $30.
   - Total amount: $50 + $15 + $30 = $95.

8. **Combine Results**: Combine the total amount of money Betty will have correctly.
   - Total amount: $50 (current amount) + $15 (parents' contribution) + $30 (grandparents' contribution) = $95.

9. **Simplify and Notate**: The final answer is already simplified.

10. **Mark the Final Answer**: 
   - Amount Betty needs to buy the wallet: $100 - $95 = $<100-95=5>5.

<ANS_START>5<ANS_END> <ANS_START>5<ANS_END>

[Question] A rectangle has a length of 10 cm and a width of 5 cm. What is the area and perimeter of the rectangle?
[Answer] 1. **Understand the Problem**: We need to find both the area and the perimeter of a rectangle given its length and width.

2. **Identify Components**: 
   - Length of the rectangle (L) = 10 cm
   - Width of the rectangle (W) = 5 cm

3. **Apply Relevant Principles**: 
   - The formula for the area of a rectangle is \( \text{Area} = \text{Length} \times \text{Width} \).
   - The formula for the perimeter of a rectangle is \( \text{Perimeter} = 2 \times (\text{Length} + \text{Width}) \).

4. **Logical Reasoning**:
   - To find the area, multiply the length by the width.
   - To find the perimeter, add the length and the width, then multiply the result by 2.

5. **Detailed Explanations**:
   - Calculate the area: \( \text{Area} = 10 \, \text{cm} \times 5 \, \text{cm} \).
   - Calculate the perimeter: \( \text{Perimeter} = 2 \times (10 \, \text{cm} + 5 \, \text{cm}) \).

6. **Explicit Calculation Steps**:
   - Area: \( 10 \times 5 = 50 \, \text{cm}^2 \).
   - Perimeter: \( 2 \times (10 + 5) = 2 \times 15 = 30 \, \text{cm} \).

7. **Verify Each Step**: 
   - Recheck the area calculation: \( 10 \times 5 = 50 \, \text{cm}^2 \).
   - Recheck the perimeter calculation: \( 2 \times 15 = 30 \, \text{cm} \).

8. **Combine Results**: 
   - The area of the rectangle is \( 50 \, \text{cm}^2 \).
   - The perimeter of the rectangle is \( 30 \, \text{cm} \).

9. **Simplify and Notate**: 
   - The final answers are already simplified.

10. **Mark the Final Answer**: 
   - Area: <ANS_START>50 \, \text{cm}^2<ANS_END>
   - Perimeter: <ANS_START>30 \, \text{cm}<ANS_END>

By following these steps, we ensure that the solution is accurate, well-explained, and clearly presented. <ANS_START>50<ANS_END>

[Question] Solve for x in the equation 2x + 3 = 11.
[Answer] **Understand the Problem**: We need to solve for the variable \( x \) in the given linear equation \( 2x + 3 = 11 \).

**Identify Components**: 
- The equation is \( 2x + 3 = 11 \).
- We need to isolate \( x \) on one side of the equation.

**Apply Relevant Principles**: 
- Use basic algebraic principles to isolate \( x \).

**Logical Reasoning**:
1. Start with the given equation: \( 2x + 3 = 11 \).
2. Subtract 3 from both sides of the equation to isolate the term with \( x \):
   \[
   2x + 3 - 3 = 11 - 3
   \]
3. Simplify both sides:
   \[
   2x = 8
   \]
4. Divide both sides by 2 to solve for \( x \):
   \[
   \frac{2x}{2} = \frac{8}{2}
   \]
5. Simplify the division:
   \[
   x = 4
   \]

**Detailed Explanations**:
- Subtracting 3 from both sides removes the constant term on the left side, leaving \( 2x \) isolated.
- Dividing both sides by 2 isolates \( x \) by removing the coefficient of 2.

**Explicit Calculation Steps**:
1. \( 2x + 3 = 11 \)
2. \( 2x + 3 - 3 = 11 - 3 \)
3. \( 2x = 8 \)
4. \( \frac{2x}{2} = \frac{8}{2} \)
5. \( x = 4 \)

**Verify Each Step**:
- Recheck each step to ensure no arithmetic errors:
  - Subtracting 3 from 11 gives 8.
  - Dividing 8 by 2 gives 4.

**Combine Results**: The final value of \( x \) is correctly isolated and calculated.

**Simplify and Notate**: The final answer is already simplified.

**Mark the Final Answer**: <ANS_START>4<ANS_END>

By following these steps, we ensure that the solution is accurate, well-explained, and clearly presented. <ANS_START>4<ANS_END>


For each question present the reasoning followed by the correct answer."""

