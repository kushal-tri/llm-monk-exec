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

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "Asap7772/code_contests", "Directory to load data from")
# flags.DEFINE_string("dataset", "Asap7772/code_contests_llamasft1e-5_mc_passk-part1-of-1", "Directory to load data from")
flags.DEFINE_integer("num_workers", 16, "Number of workers to use for grading")
flags.DEFINE_string("save_dir", "results", "Directory to save results in")
# flags.DEFINE_string("split", "train", "Split to evaluate on")
flags.DEFINE_string("split", "valid", "Split to evaluate on")
flags.DEFINE_string('solution_col', 'solutions', 'Column name for solutions')
# flags.DEFINE_string('solution_col', 'responses', 'Column name for solutions')
flags.DEFINE_integer('max_solutions', 16, 'Maximum number of solutions to evaluate')

from llmonk.evaluate.code_contests_utils import execution_server_client

MAX_CONCURRENT_REQUESTS = 512
semaphore = threading.Semaphore(value=MAX_CONCURRENT_REQUESTS)
NUM_RETRIES = 3
RETRY_BACKOFF = 3


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
    client: execution_server_client.ExecutionServerClient,
):
    if code is None:
        return False

    assert len(problem["test_cases"]["input"]) == len(problem["test_cases"]["output"])

    input_expected_output_pairs = list(
        zip(problem["test_cases"]["input"], problem["test_cases"]["output"])
    )

    with semaphore:
        for i in range(NUM_RETRIES):
            try:
                is_correct = client.execute_code(
                    extract_first_code(code),
                    input_expected_output_pairs,
                    timeout=problem["timeout"] + 10,  # buffer for 10
                    memory_limit_bytes=2_000_000_000_000,  # double max limit
                )
                break
            except:
                if i == NUM_RETRIES - 1:
                    raise
                time.sleep(RETRY_BACKOFF**i)

    return is_correct


def grade_problems(
    solutions_data: dict,
    output_dir: Path,
    client: execution_server_client.ExecutionServerClient,
):
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_REQUESTS // 2
    ) as executor:
        is_corrects_futures = [
            executor.submit(
                solution_is_correct,
                code=code,
                problem=solutions_data,
                client=client,
            )
            for code in solutions_data["solutions"]
        ]

        is_corrects = []
        for i, future in enumerate(is_corrects_futures):
            if i % 100 == 0:
                print("Progress being made...")
            is_corrects.append(future.result())

    solutions_data["is_corrects"] = is_corrects

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / solutions_data["name"], "w") as f:
        yaml.dump(
            solutions_data,
            f,
            sort_keys=True,
        )


def load_data_from_dataset(ds, solution_col="responses", max_solutions=16):
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
            return_dict['test_cases'].append(examples['test_cases'][i])
            return_dict['solutions'].append(examples[solution_col][i][:max_solutions])
            return_dict['timeout'].append(examples['timeout'][i])
        return return_dict
    all_cols = list(ds.column_names)
    rm_cols = [col for col in all_cols if col not in keys]
    ds_mapped = ds.map(map_fn, batched=True, num_proc=FLAGS.num_workers, remove_columns=rm_cols)
    # convert to dict of lists
    df = ds_mapped.data.to_pandas()
    return df.to_dict(orient='records')


def main(_):
    ds = datasets.load_dataset(FLAGS.dataset, split=FLAGS.split)
    solutions_data = load_data_from_dataset(ds, solution_col=FLAGS.solution_col, max_solutions=FLAGS.max_solutions)

    # multiprocessing pool is used to load data
    with execution_server_client.ExecutionServerClient() as client:
        # threads are used to run code in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=FLAGS.num_workers
        ) as executor:
            futures = [
                executor.submit(
                    grade_problems,
                    solutions_data=solution_data,
                    output_dir=FLAGS.save_dir,
                    client=client,
                )
                for solution_data in solutions_data
            ]

            for future in tqdm(futures, desc="Running tests on problem"):
                future.result()

    df = pd.DataFrame(solutions_data)
    ds = datasets.Dataset.from_pandas(df)
    ds.push_to_hub(f"{FLAGS.dataset}_graded", hf_token = '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')
    
if __name__ == "__main__":
    app.run(main)
