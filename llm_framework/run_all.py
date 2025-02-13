import subprocess

# Define the commands
commands = [
    "python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama3 --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 200 --eval-per-epoch 2",
    "python run_plm.py --adapt --grad-accum-steps 32 --plm-type opt --plm-size xs --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 200 --eval-per-epoch 2",
    "python run_plm.py --adapt --grad-accum-steps 32 --plm-type t5-lm --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 200 --eval-per-epoch 2",
    "python run_plm.py --adapt --grad-accum-steps 32 --plm-type gpt2 --plm-size small --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 200 --eval-per-epoch 2",
    "python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama2 --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 200 --eval-per-epoch 2"
]

# Run each command in sequence
for command in commands:
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully ran: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running: {command}")
        print(f"Error details: {e}")
        break  # Stop execution if a command fails
