import argparse
import os
import random

model_dir = "/workspace/models/"
# task_list = {"arc-e", "arc-c", "boolq", "obqa", "piqa", "siqa", "hellaswag", "winogrande"}
task_list = {"arc-e", "arc-c", "boolq", "obqa", "piqa", "siqa", "hellaswag", "winogrande"}
model_list = {"microsoft/Phi-3-mini-128k-instruct"}

peft_methods = {"loraplus", "rslora"}

config_command = f"python launch.py gen"
run_command = f"python launch.py run"

config_suffix = {
    "lora": "--template lora",
    "dora": "--template lora --use_dora",
    "qlora": "--template lora",
    "loraplus": "--template lora --loraplus_lr_ratio 20.0",
    "rslora": "--template lora --use_rslora",
    "mixlora": "--template mixlora",
    "mixdora": "--template mixlora --use_dora",
    "qmixlora": "--template mixlora",
    "loramoe": "--template loramoe_phi3",
    "mola": "--template mola_phi3",
}

run_suffix = {
    "qlora": "--quantize 4bit",
    "qmixlora": "--quantize 4bit",
}


def call_gen(peft_method, tasks, name_prefix, multi_task):
    if multi_task:
        config = (f"{config_command} --tasks \"{tasks}\" --adapter_name multi_{peft_method} --batch_size 64 --micro_batch_size 32"
                  f" --file_name multi_{peft_method}.json {config_suffix.get(peft_method, '')} --multi_task")
    else:
        config = (f"{config_command} --tasks {tasks} --adapter_name {name_prefix} --batch_size 64 --micro_batch_size 32"
                  f" --file_name {name_prefix}.json {config_suffix.get(peft_method, '')}")
    os.system(config)
    print(config)


def call_run(peft_method, model, name_prefix, multi_task):
    if multi_task:
        run = (f"{run_command} --base_model {model_dir}{model} --config multi_{peft_method}.json"
               f" --cuda_device {args.cuda} --log_file multi_{peft_method}.log"
               f" --overwrite false --attn_impl eager {run_suffix.get(peft_method, '')}")
    else:
        run = (f"{run_command} --base_model {model_dir}{model} --config {name_prefix}.json"
               f" --cuda_device {args.cuda} --log_file {name_prefix}.log"
               f" --overwrite false --attn_impl eager {run_suffix.get(peft_method, '')}")
    os.system(run)
    print(run)


def main(args):
    for model in model_list:
        for method in peft_methods:
            for task in task_list:
                name_prefix = f"{model.split('/')[-1]}_{method}_{task}"
                if args.run:
                    call_run(method, model, name_prefix, args.multi_task)
                elif args.gen:
                    call_gen(method, task, name_prefix, args.multi_task)
                else:
                    call_gen(method, task, name_prefix, args.multi_task)
                    call_run(method, model, name_prefix, args.multi_task)


    # model = "THUDM/chatglm3-6b"
    # method = "lora"    # for task in task_list:
    #     sys_call(method, task, model)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch run tasks")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device number")
    parser.add_argument("--run", action="store_true", help="Run the tasks")
    parser.add_argument("--gen", action="store_true", help="Generate the config")
    parser.add_argument("--multi_task", action="store_true", help="Run multiple tasks at once")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

# python launch.py run --base_model /workspace/models/microsoft/Phi-3-mini-128k-instruct --config multi_qmixlora.json --cuda_device 3 --log_file multi_qmixlora.log --attn_impl eager --quantize 4bit