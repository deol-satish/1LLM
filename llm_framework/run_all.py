import subprocess

# Function to generate commands for different models and configurations
def generate_command(plm_type, plm_size, mode='adapt', device='cuda:0', grad_accum_steps=32, 
                     lr=0.0001, warmup_steps=2000, num_epochs=200, eval_per_epoch=2, rank=128):
    return f"python run_plm.py --{mode} --grad-accum-steps {grad_accum_steps} --plm-type {plm_type} --plm-size {plm_size} --rank {rank} --device {device} --lr {lr} --warmup-steps {warmup_steps} --num-epochs {num_epochs} --eval-per-epoch {eval_per_epoch}"

# List of PLM types and sizes
plm_configurations = [
    ("llama3", "base"),
    ("opt", "xs"),
    ("t5", "base"),
    ("gpt2", "small"),
    ("llama2", "base"),
]

# Modes for each experiment
# modes = ['adapt', 'test', 'eval']
modes = ['eval']

# Generate all commands dynamically
commands = [generate_command(plm_type, plm_size, mode) 
            for plm_type, plm_size in plm_configurations 
            for mode in modes]

# Run each command in sequence
for command in commands:
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully ran: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running: {command}")
        print(f"Error details: {e}")
        break  # Stop execution if a command fails
