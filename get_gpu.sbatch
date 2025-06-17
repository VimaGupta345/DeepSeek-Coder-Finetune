#!/bin/bash
#SBATCH -JFineTuningOCaml2                    # Job name
#SBATCH --account=gts-ag117                 # charge account
#SBATCH -N1 -n1                                 # Number of nodes and cores required
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=80G                        # Memory per core
#SBATCH -t11:00:00                                    # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                               # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=xliang304@gatech.edu        # E-mail address for notifications

cd $SLURM_SUBMIT_DIR                          # Change to working directory

echo "hello_world"
echo "Job is still running. You can SSH into the node: $SLURM_JOB_NODELIST"

# Keep the job alive:
sleep infinity
