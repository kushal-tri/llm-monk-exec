from pathlib import Path
from multiprocessing import Pool
import threading
from tqdm import tqdm
import yaml
import time
import concurrent.futures
import re
from absl import app, flags
import datasets
import pandas as pd
import os

FLAGS = flags.FLAGS
# flags.DEFINE_string("dataset", "Asap7772/code_contests", "Directory to load data from")
flags.DEFINE_string("dataset", "Asap7772/code_contests_llamasft1e-7_mc_passk_train-part1-of-1", "Directory to load data from")
flags.DEFINE_integer("num_workers", 512, "Number of workers to use for grading")
flags.DEFINE_string("save_dir", "results", "Directory to save results in")
# flags.DEFINE_string("split", "valid", "Split to evaluate on")
flags.DEFINE_string("split", "train", "Split to evaluate on")
# flags.DEFINE_string('solution_col', 'solutions', 'Column name for solutions')
flags.DEFINE_string('solution_col', 'responses', 'Column name for solutions')
flags.DEFINE_integer('max_solutions', 256, 'Maximum number of solutions to evaluate')
flags.DEFINE_float('per_testcases', -1.0, 'Percentage of testcases to evaluate')

from llmonk.evaluate.code_contests_utils.schema import ExecuteCodeRequest, ExecuteCodeResult
MAX_CONCURRENT_REQUESTS = os.cpu_count() * 2
NUM_RETRIES = 1
RETRY_BACKOFF = 3

import tempfile
import math
import subprocess
from llmonk.evaluate.code_contests_utils.compare_results import outputs_match

def execute_with_input(
    code_file_name: str, input_str: str, timeout: float, memory_limit_bytes: int
):
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_input_file:
        temp_input_file.write(input_str)
        temp_input_file.flush()  # Ensure the data is written to disk

        # Rewind the file so the subprocess reads it from the beginning
        temp_input_file.seek(0)

        try:
            memory_limit_kb = math.ceil(memory_limit_bytes / 1024)
            cmd = f"ulimit -v {memory_limit_kb} && python {code_file_name}"

            try:
                result = subprocess.run(
                    ["bash", "-c", cmd],
                    capture_output=True,
                    stdin=temp_input_file.fileno(),
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return None
            
            return result.stdout.decode()
        except Exception as e:
            return None

def execute_python_code(request: ExecuteCodeRequest):
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=True) as temp:
            
            with open(temp.name, "w") as f:
                f.write(request.code)

            for input_str, expected_output in request.input_expected_output_pairs:
                # Intentionally serial - most programs fail.
                actual_output = execute_with_input(
                    code_file_name=temp.name,
                    input_str=input_str,
                    timeout=request.timeout,
                    memory_limit_bytes=request.memory_limit_bytes,
                )

                if actual_output is None or not outputs_match(expected_output, actual_output):
                    return ExecuteCodeResult(correct=False)

            return ExecuteCodeResult(correct=True)
    except Exception as e:
        print(f"Failed to execute code: \n {e}")

def is_valid_python(snippet):
    try:
        compile(snippet, "<string>", "exec")
        return True
    except SyntaxError:
        return False
           
def extract_first_code(output_string: str):
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # sometimes the block of code is ```python ... ``` instead of ``` ... ```
        # in this case strip the python out

        if code.startswith("python"):
            code = code[len("python") :].strip()

        return code

    if is_valid_python(trimmed):
        return trimmed

    return None

def solution_is_correct(
    code: str | None,
    problem: dict,
):
    if code is None:
        return False

    assert len(problem["test_cases"]["input"]) == len(problem["test_cases"]["output"])

    input_expected_output_pairs = list(
        zip(problem["test_cases"]["input"], problem["test_cases"]["output"])
    )
    is_correct = False
    code = extract_first_code(code)
    if code is not None:
        is_correct = execute_python_code(
            ExecuteCodeRequest(
                code=code,
                input_expected_output_pairs=input_expected_output_pairs,
                timeout=problem["timeout"] + 10,  # buffer for 10
                memory_limit_bytes=2_000_000_000_000,  # double max limit
        ))
    return is_correct


def grade_problems(
    solutions_data: dict,
    output_dir: Path,
):
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_REQUESTS
    ) as executor:
        futures = []
        for idx, solution_data in enumerate(solutions_data):
            for code in solution_data["solutions"]:
                future = executor.submit(
                            solution_is_correct,
                            code=code,
                            problem=solution_data,
                        )
                futures.append((future, idx))

        is_corrects_list = [[]] * len(solutions_data) 
        for i, (future, idx) in tqdm(enumerate(futures), total=len(futures), desc="Running tests on problem"):
            is_corrects_list[idx].append(future.result())
        
        for idx, is_corrects in enumerate(is_corrects_list):
            solutions_data[idx]["is_corrects"] = is_corrects

def load_data_from_dataset(ds, solution_col="responses", max_solutions=16, per_testcases=-1):
    keys = ['test_cases', 'solutions', 'name', 'timeout']
    all_prompts = sorted(list(set(ds['prompt'])))
    prompt_to_idx = {prompt: i for i, prompt in enumerate(all_prompts)}
    
    def map_fn(examples):
        return_dict = {k:[] for k in keys}
        for i in range(len(examples['prompt'])):
            curr_prompt = examples['prompt'][i]
            which_prompt = prompt_to_idx[curr_prompt]
            name = f"problem{which_prompt}"
            return_dict['name'].append(name)
            if per_testcases > 0:
                num_testcases = len(examples['test_cases'][i])
                num_testcases_to_keep = int(num_testcases * per_testcases)
                test_cases = examples['test_cases'][i][:num_testcases_to_keep]
                return_dict['test_cases'].append(test_cases)
            else:
                return_dict['test_cases'].append(examples['test_cases'][i])
            return_dict['solutions'].append(examples[solution_col][i][:max_solutions])
            return_dict['timeout'].append(examples['timeout'][i])
        return return_dict
    all_cols = list(ds.column_names)
    rm_cols = [col for col in all_cols if col not in keys]
    ds_mapped = ds.map(map_fn, batched=True, num_proc=8, remove_columns=rm_cols)
    # convert to dict of lists
    df = ds_mapped.data.to_pandas()
    return df.to_dict(orient='records')


def main(_):
    ds = datasets.load_dataset(FLAGS.dataset, split=FLAGS.split)
    solutions_data = load_data_from_dataset(ds, solution_col=FLAGS.solution_col, max_solutions=FLAGS.max_solutions, per_testcases=FLAGS.per_testcases)
    grade_problems(solutions_data, FLAGS.save_dir)

    hf_token = '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'
    os.environ['HF_TOKEN'] = hf_token
    
    df = pd.DataFrame(solutions_data)
    ds = datasets.Dataset.from_pandas(df)
    if FLAGS.per_testcases > 0:
        output_name = f"{FLAGS.dataset}_graded_{FLAGS.per_testcases}"
    else:
        output_name = f"{FLAGS.dataset}_graded"
    
    ds.push_to_hub(output_name)
    
if __name__ == "__main__":
    app.run(main)
