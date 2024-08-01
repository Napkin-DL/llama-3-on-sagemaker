import logging
from dataclasses import dataclass, field
import os
import random
import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import  TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,

)
from trl import setup_chat_format
from peft import LoraConfig


from trl import SFTTrainer, SFTConfig

os.environ['ACCELERATE_USE_FSDP'] = '1'
os.environ['FSDP_CPU_RAM_EFFICIENT_LOADING'] = '1'
os.environ['NCCL_DEBUG'] = 'INFO'


import mlflow
from  mlflow.tracking import MlflowClient

# Set the tracking server URI using the ARN of the tracking server you created
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_ARN'])


# Comment in if you want to use the Llama 3 instruct template but make sure to add modules_to_save
# LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

# Anthropic/Vicuna like template without the need for special tokens
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)


# ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=4 ./scripts/run_fsdp_qlora.py --config llama_3_70b_fsdp_qlora.yaml

@dataclass
class ScriptArguments:
    train_dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset"
        },
    )
    test_dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset"
        },
    )
    model_name_or_path: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )
    
    model_uri: str = field(
        default=None, metadata={"help": "S3 URI to save the merged_mode"}
    )

    wb_token: str = field(
        default=None, metadata={"help": "wb_token"}
    )

def wandb_login(args):
    # wb_token = userdata.get('wandb')
    print(f"args.wb_token : {args.wb_token}")
    wandb.login(key=args.wb_token)
    
    
def training_function(script_args, sft_config):
    exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
    run_name = os.environ['MLFLOW_RUN_NAME']
    
    
    if script_args.rank == 0:
        client = MlflowClient()
        filter_string = f"name='{exp_name}'"
        experiment_id = mlflow.search_experiments(filter_string=filter_string)[0].experiment_id
        run = client.create_run(experiment_id=experiment_id , run_name=run_name)

        # log parameter to this specific run
        for key,val in sft_config.to_dict().items():
            client.log_param(run.info.run_id, key, val)

    ################
    # Dataset
    ################


    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.train_dataset_path, "train_dataset.json"),
        split="train",
    )
    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.test_dataset_path, "test_dataset.json"),
        split="train",
    )

    ################
    # Model & Tokenizer
    ################

    # Tokenizer        
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE

    # print random sample
    with sft_config.main_process_first(
        desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(train_dataset)), 2):
            print(train_dataset[index]["text"])

    # Model    
    torch_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=torch_dtype,
        )


    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        # attn_implementation="sdpa", # use sdpa, alternatively use "flash_attention_2"
        attn_implementation="flash_attention_2", # use sdpa, alternatively use "flash_attention_2"
        torch_dtype=torch_dtype,
        use_cache=False if sft_config.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )

    if sft_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ################
    # PEFT
    ################

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        # modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
    )

    import evaluate
    
    def compute_metrics(eval_preds):
        
        metric = evaluate.load("glue", "mrpc")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    ################
    # Training
    ################
    print(f"sft_config.output_dir : {sft_config.output_dir}")

    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False,  # No need to add additional separator token
    }

    sft_config.dataset_text_field="text"
    sft_config.packing=True
    sft_config.dataset_kwargs=dataset_kwargs


    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics
    )
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    ##########################
    # Train model
    ##########################
    checkpoint = None
    if sft_config.resume_from_checkpoint is not None:
        print(f"sft_config.resume_from_checkpoint : {sft_config.resume_from_checkpoint}")
        checkpoint = sft_config.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # compute train results
    metrics = train_result.metrics
    
    if script_args.rank == 0:
        for key, val in metrics.items():
            client.log_metric(
                run_id=run.info.run_id,
                key=key,
                value=val
            )

    ##########################
    # SAVE MODEL FOR SAGEMAKER
    ##########################
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model()

    if script_args.rank == 0:
        mv = mlflow.register_model(script_args.model_uri, "llama-3-1-kor-bllossom-8b")
        mlflow.end_run()

        

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig))
    script_args, sft_config = parser.parse_args_and_config()  
    # config_args = yaml.load(open(script_args.config, 'r'), Loader = yaml.Loader)
    print(f"os.environ : {os.environ}")
    # print(f"sft_config.report_to : {sft_config.report_to[0]}")
    if sft_config.report_to[0] == 'wandb':
        wandb_login(script_args)

    # print(f"script_args : {script_args}, sft_config : {sft_config}")
    # set use reentrant to False
    if sft_config.gradient_checkpointing:
        sft_config.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(sft_config.seed)
    
    script_args.rank = int(os.environ['RANK'])
  
    # launch training
    training_function(script_args, sft_config)
