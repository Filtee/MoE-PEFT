import copy
import json
import os
from encodings.punycode import selective_len

import fire
from yaml import compose

from moe_peft.tasks import task_dict

WORK_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_SAVING_PATH = f"{WORK_PATH}{os.sep}save_configs"
LOG_SAVING_PATH = f"{WORK_PATH}{os.sep}save_logs"

BASE_EDITABLE_DICT = {
    "lr": int,
    "scheduler_type": str,
    "warmup_steps": int,
    "batch_size": int,
    "micro_batch_size": int,
    "evaluate_steps": int,
    "evaluate_batch_size": int,
    "num_epochs": int,
}

PEFT_EDITABLE_DICT = {
    "lora": {},
    "dora": {
        "use_dora",
    },
    "lora_plus": {
        "loraplus_lr_ratio",
    },
    "rslora": {
        "use_rslora",
    },
    "dynmole": {
        "entropy_threshold",
        "entropy_index",
        "keep_top_k",
        "router_dyn_loss_coef",
    },
}


def print_red(message):
    print('\033[0;31m' + message + '\033[0m')


def print_green(message):
    print('\033[0;32m' + message + '\033[0m')


def get_tasks():
    input_tasks = []
    while True:
        task = input("Tasks? (Only one task for each input round): ").strip()

        if task == "" and input_tasks != []:
            break
        elif task == "" and input_tasks == []:
            print_red("Please enter at least one task!")
            continue
        elif task not in task_dict.keys():
            print_red(f"Wrong input! The task {task} is not in the task dict!")
            continue
        elif task in input_tasks:
            print_red(f"Wrong input! Task {task} has already input!")
            continue

        input_tasks.append(task)
    return input_tasks


def get_peft_method():
    while True:
        peft_method = input("Plz input the peft method: ").strip()
        if peft_method in PEFT_EDITABLE_DICT.keys():
            break
        else:
            print_red("Wrong Input!")
    return null


def edit_params(lora_config, editable_list):
    for param in editable_list:
        while True:
            input_param = input(f">>> Input {param} (DEFAULT {lora_config[param]}): ").strip()

            if input_param == "":
                break
            # Try to cast the input data type.
            try:
                target_type = type(lora_config[param])
                lora_config[param] = target_type(input_param)
                break
            except ValueError as _:
                print_red("Wrong input data type!")
                continue


def edit_template_config(
    tasks: list,
    peft_method: str,
):
    template_dir = f"{WORK_PATH}{os.sep}templates{os.sep}{peft_method}.json"
    with open(template_dir, "r", encoding="utf8") as fp:
        template = json.load(fp)

    lora_template = template["lora"].pop()
    lora_config_pool, cnt = {}, 0

    # Edit params for each task.
    for task in tasks:
        lora_config = copy.deepcopy(lora_template)
        lora_config["task_name"] = task + f"_{cnt}"
        print_green(f"Input parameters for {task}...")

        # Edit base template params.
        while True:
            choice = input(f"? Edit base config params: [y/N] ").strip().lower()
            choice = "n" if choice == "" else choice
            if choice in {"y", "n", "yes", "no"}:
                break
        if choice == "y" or choice == "yes":
            edit_params(lora_config, BASE_EDITABLE_DICT)

        # Edit template params basing on different PEFT methods.
        edit_params(lora_config, PEFT_EDITABLE_DICT[peft_method])

        # Add this lora config to the POOL.
        lora_config_pool[task] = lora_config
        cnt += 1

    # TODO: Ask the user again to confirm the edit.

    for lora_config in lora_config_pool.values():
        template["lora"].append(lora_config)
    return template


def generate_config():
    if not os.path.exists(CONFIG_SAVING_PATH):
        os.makedirs(CONFIG_SAVING_PATH)
    if not os.path.exists(LOG_SAVING_PATH):
        os.makedirs(LOG_SAVING_PATH)

    # Get task_name.
    task_name = input("Plz input the name of config file: ").strip()

    # Get config.
    tasks = get_tasks()

    peft_method = get_peft_method()
    template = edit_template_config(tasks, peft_method)

    # Save config.
    config_dir = f"{CONFIG_SAVING_PATH}{task_name}.json"
    with open(config_dir, "w", encoding="utf8") as fp:
        json.dump(template, fp, indent=4)

    print_green(f"Configuration file saved to {config_dir}")


def compose_command(
    base_model: str,
    cuda_device: int = 0,
    config: str = "moe_peft.json",
    log_file: str = "moe_peft.log",
    random_seed: int = 42,
    attn_impl: str = "eager",
    dtype: str = "bf16",
    quantize: str = None,
):
    assert base_model is not None
    assert dtype in ("bf16", "bf32", "fp32")
    assert quantize in (None, "4bit", "8bit")

    command = f"""
        CUDA_VISIBLE_DEVICES={cuda_device}
        python moe_peft.py \
            --base_model {base_model}
            --config {config}
            --log_file {log_file}
            --seed {random_seed}
            --attn_impl {attn_impl}
            --dtype {dtype}
    """


def choose_file():
    config_files = os.listdir(CONFIG_SAVING_PATH)
    file_names = [
        f for f in config_files if os.path.isfile(os.path.join(CONFIG_SAVING_PATH, f))
                                   and f.endswith('.json')
    ]

    if len(file_names) == 0:
        print_red("No config files currently!")
        selected_file = None
    else:
        print_green("Current config files:")
        for index, file_name in enumerate(file_names):
            print(f"{index}: {file_name}")

        while True:
            try:
                user_input = input(f"\n? Which file you choose to run: ")
                selected_index = int(user_input)
                if 0 <= selected_index < len(file_names):
                    selected_file = file_names[selected_index]
                    break
                else:
                    print_red(f"Plz input a valid index between 0 and {len(file_names) - 1}!")
            except ValueError:
                print_red("Invalid input type!")
    return selected_file


def run_command():
    config_file = choose_file()
    if config_file is None:
        return

    command = compose(...)
    os.system(command)


def show_help(*args, **kwargs):
    # TODO: Complete func::show_help.
    return


command_map = {
    "gen": generate_config,
    "run": run_command,
    "help": show_help,
}


def main(command: str = "help", *args, **kwargs):
    command_map[command](*args, **kwargs)


if __name__ == '__main__':
    fire.Fire(main)