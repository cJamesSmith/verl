system_prompts = """<|im_start|>system\nYou are a helpful assistant help user solve problems. \
<|im_end|>\n<|im_start|>user\nYou need to think first then write {{language}} script. {{special_requirements}}
This is the problem:\n{{problem}}\n\n<|im_end|>\n<|im_start|>assistant\n\n"""

system_case_prompts = """<|im_start|>system\nYou are a helpful assistant help user generate test examples for coding tasks. \
<|im_end|>\n<|im_start|>user\nGiven a coding task, instead of providing the final script, your task is to generate a new test example (both input, output and explanation).
This is the problem:\n{{problem}}\n
{{example_intro}}
You need to provide a new test example. A good test example should be completely accurate and conform to the problem's format requirements, while also possessing enough discriminative power to distinguish correct code from incorrect code.\n\nBefore providing a test example, you must think carefully and reason step by step to derive an input and output you are very confident are correct. For example, start by designing an input you can reliably handle, then compute the output step by step. If you're unsure about the output, revise or re-design the input to ensure accuracy. Directly providing input/output pairs without this process is discouraged, as it often results in low accuracy.\n\nFinally, after completing these previous thinking and derivation steps (you should not write the final test example unless you have gone through these steps very thoroughly), you MUST put your final test example in the following format:\n\n**Test Input:**\n\n```input

```\n\n**Test Output:**\n\n```output

```\n\n**Explanation:**\n\n```explanation

```.\n\n<|im_end|>\n<|im_start|>assistant\n\n"""

# some special requirements for code generation
special_requirements = """You should use input() to input and print() to output in your script. """
