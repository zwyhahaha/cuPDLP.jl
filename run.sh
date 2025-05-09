CUDA_VISIBLE_DEVICES=1 ~/julia-1.10/bin/julia --project scripts/solve.jl --instance_name=all --dataset=netlib --experiment_name=PDLP --tolerance=1e-4 --learning_rate=0.0 --time_sec_limit=3600

CUDA_VISIBLE_DEVICES=1 ~/julia-1.10/bin/julia --project scripts/solve.jl --instance_name=all --dataset=MIPLIB --experiment_name=PDLP --tolerance=1e-4 --learning_rate=0.0 --time_sec_limit=3600

CUDA_VISIBLE_DEVICES=1 ~/julia-1.10/bin/julia --project scripts/solve.jl --instance_name=all --dataset=netlib --experiment_name=won-f-adaPDLP --tolerance=1e-4 --time_sec_limit=3600 --learning_rate=0.00001 --normalize=false --online_scaling_frequency=1

CUDA_VISIBLE_DEVICES=1 ~/julia-1.10/bin/julia --project scripts/solve.jl --instance_name=all --dataset=netlib --experiment_name=wn-f-adaPDLP --tolerance=1e-4 --time_sec_limit=3600 --learning_rate=0.00001 --normalize=true --online_scaling_frequency=1

CUDA_VISIBLE_DEVICES=1 ~/julia-1.10/bin/julia --project scripts/solve.jl --instance_name=all --dataset=netlib --experiment_name=won-if-adaPDLP --tolerance=1e-4 --time_sec_limit=3600 --learning_rate=0.00001 --normalize=false --online_scaling_frequency=20

CUDA_VISIBLE_DEVICES=1 ~/julia-1.10/bin/julia --project scripts/solve.jl --instance_name=all --dataset=netlib --experiment_name=won-if-adaPDLP --tolerance=1e-4 --time_sec_limit=3600 --learning_rate=0.00001 --normalize=true --online_scaling_frequency=20