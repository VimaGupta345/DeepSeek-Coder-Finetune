#!/bin/bash
#SBATCH -interactive                    # Job name
#SBATCH --account=gts-apadmanabh3                 # charge account
#SBATCH -N2 --ntasks-per-node=4                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=80G                        # Memory per core
#SBATCH -t420                                    # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                               # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=vgupta345@gatech.edu        # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory

conda activate ian6-myenv

srun deepspeed finetune_lora_deepseekcoder.py --model_name_or_path deepseek-ai/deepseek-coder-6.7b-instruct --data_path data/rust_instruct_format.json --output_dir output --per_device_train_batch_size 16 --gradient_accumulation_steps 4 --learning_rate 2e-5 --bf16 True --gradient_checkpointing False --deepspeed configs/ds_config_zero3.json --report_to wandb --run_name deepseek-lora-r16-longrun --logging_dir ./wandb_logs --logging_steps 25 --evaluation_strategy steps --eval_steps 200 --save_strategy steps --save_steps 200 --save_total_limit 3 --max_steps 1500 --num_train_epochs 3 --model_max_length 1024
