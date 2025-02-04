import ArgParse
import GZip
import JSON3
import JSON
import CSV
import DataFrames

include("/home/shanshu/opt/cuPDLP.jl/src/cuPDLP.jl")
include("json2csv.jl")


function write_vector_to_file(filename, vector)
    open(filename, "w") do io
      for x in vector
        println(io, x)
      end
    end
end

function solve_instance_and_output(
    parameters::cuPDLP.PdhgParameters,
    output_dir::String,
    instance_path::String,
)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
  
    instance_name = replace(basename(instance_path), r"\.(mps|MPS|qps|QPS)(\.gz)?$" => "")
  
    function inner_solve()
        lower_file_name = lowercase(basename(instance_path))
        if endswith(lower_file_name, ".mps") ||
            endswith(lower_file_name, ".mps.gz") ||
            endswith(lower_file_name, ".qps") ||
            endswith(lower_file_name, ".qps.gz")
            lp = cuPDLP.qps_reader_to_standard_form(instance_path)
        else
            error(
                "Instance has unrecognized file extension: ", 
                basename(instance_path),
            )
        end
    
        if parameters.verbosity >= 1
            println("Instance: ", instance_name)
        end

        output::cuPDLP.SaddlePointOutput = cuPDLP.optimize(parameters, lp)
    
        log = cuPDLP.SolveLog()
        log.instance_name = instance_name
        log.command_line_invocation = join([PROGRAM_FILE; ARGS...], " ")
        log.termination_reason = output.termination_reason
        log.termination_string = output.termination_string
        log.iteration_count = output.iteration_count
        log.solve_time_sec = output.iteration_stats[end].cumulative_time_sec
        log.solution_stats = output.iteration_stats[end]
        log.solution_type = cuPDLP.POINT_TYPE_AVERAGE_ITERATE
    
        summary_output_path = joinpath(output_dir, instance_name * "_summary.json")
        open(summary_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end
    
        log.iteration_stats = output.iteration_stats
        full_log_output_path =
            joinpath(output_dir, instance_name * "_full_log.json.gz")
        GZip.open(full_log_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end
    
        primal_output_path = joinpath(output_dir, instance_name * "_primal.txt")
        write_vector_to_file(primal_output_path, output.primal_solution)
    
        dual_output_path = joinpath(output_dir, instance_name * "_dual.txt")
        write_vector_to_file(dual_output_path, output.dual_solution)
    end     

    inner_solve()
   
    return
end

"""
Warm up the GPU by pre-solving the LP for 100 iterations.
"""
function warm_up(lp::cuPDLP.QuadraticProgrammingProblem)
    restart_params = cuPDLP.construct_restart_parameters(
        cuPDLP.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        cuPDLP.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.5,                    # primal_weight_update_smoothing
    )

    termination_params_warmup = cuPDLP.construct_termination_criteria(
        # optimality_norm = L2,
        eps_optimal_absolute = 1.0e-4,
        eps_optimal_relative = 1.0e-4,
        eps_primal_infeasible = 1.0e-8,
        eps_dual_infeasible = 1.0e-8,
        time_sec_limit = Inf,
        iteration_limit = 100,
        kkt_matrix_pass_limit = Inf,
    )

    params_warmup = cuPDLP.PdhgParameters(
        10, # ruiz scaling iterations 
        false, # off l2 scaling
        1.0, # Pock scaling coefficient: l1 norm 
        1.0, # primal importance, control initial primal weight
        true, # scale initial primal weight
        0, # verbosity: no output
        true, # record iteration stats
        64, # termination evaluation frequency
        termination_params_warmup,
        restart_params,
        cuPDLP.AdaptiveStepsizeParams(0.3,0.6), # adaptive stepsize parameters (reduction, growth exponent)
        false, # online scaling
        0., # online learning rate
        1, # online scaling frequency
        true,
    )

    cuPDLP.optimize(params_warmup, lp);
end


function parse_command_line()
    arg_parse = ArgParse.ArgParseSettings()

    ArgParse.@add_arg_table! arg_parse begin
        "--instance_name"
        help = "The name of the instance to solve in .mps.gz or .mps format."
        arg_type = String
        default = "fit1d"
        # required = true


        "--dataset"
        help = "dataset name"
        arg_type = String
        default = "netlib"
        # required = true

        "--experiment_name"
        help = "experiment name."
        arg_type = String
        default = "hyperPDLP"
        # required = true

        "--tolerance"
        help = "KKT tolerance of the solution."
        arg_type = Float64
        default = 1e-4

        "--time_sec_limit"
        help = "Time limit."
        arg_type = Float64
        default = 3600.0

        "--learning_rate"
        help = "Learning rate for online scaling."
        arg_type = Float64
        default = 0.001

        "--online_scaling_frequency"
        help = "Frequency of online scaling."
        arg_type = Int
        default = 1

        "--normalize"
        help = "Normalize the hypergradient."
        arg_type = Bool
        default = true
    end

    return ArgParse.parse_args(arg_parse)
end

function get_iteration_from_json(json_file)
    if isfile(json_file)
        json_string = read(json_file, String)
        data = JSON.parse(json_string)
        if !isa(data, Dict)
            error("Expected a dictionary at the top level of the JSON file")
        end
        return data["iteration_count"]
    end
end

function get_ada_iteration(dataset, instance_name, time_sec_limit, tolerance)
    experiment_name = "adaPDLP_time_$(time_sec_limit)_tol_$(tolerance)_lr_0.0"
    ada_directory = joinpath("output/solver_output", dataset, experiment_name, instance_name)
    if !isdir(ada_directory)
        ada_iter =  typemax(Int32)
    else
        ada_json_file = filter(x -> endswith(x, ".json"), readdir(ada_directory))
        ada_json_file = joinpath(ada_directory, ada_json_file[1])
        ada_iter = get_iteration_from_json(ada_json_file)
    end

    return ada_iter
end

function run_instance(instance_path, args)
    dataset = args["dataset"]
    experiment_name = args["experiment_name"]
    tolerance = args["tolerance"]
    time_sec_limit = args["time_sec_limit"]
    learning_rate = args["learning_rate"]
    online_scaling_frequency = args["online_scaling_frequency"]
    normalize = args["normalize"]

    if experiment_name == "adaPDLP"
        online_scaling = true
        learning_rate = 0.0
    elseif occursin("hyperPDLP", experiment_name)
        online_scaling = true
    else
        error("Invalid experiment name")
    end
    
    if occursin("non", experiment_name)
        normalize = false
    end
    
    experiment_name = "$(experiment_name)_time_$(time_sec_limit)_tol_$(tolerance)_lr_$(learning_rate)"
    instance_name = replace(basename(instance_path), r"\.(mps|MPS|qps|QPS)(\.gz)?$" => "")

    output_directory = joinpath("output/solver_output", dataset, experiment_name,instance_name)
    if !isdir(output_directory)
        mkpath(output_directory)
    else
        if length(readdir(output_directory)) > 0
            println("Skip instance: ", instance_name)
            return
        end
    end

    # lp = cuPDLP.qps_reader_to_standard_form(instance_path)

    # println("warm up start")
    # oldstd = stdout
    # redirect_stdout(devnull) # suppress output
    # warm_up(lp); # solve the LP for 100 iterations
    # redirect_stdout(oldstd)
    # println("warm up end")

    restart_params = cuPDLP.construct_restart_parameters(
        cuPDLP.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        cuPDLP.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.5,                    # primal_weight_update_smoothing
    )

    termination_params = cuPDLP.construct_termination_criteria(
        # optimality_norm = L2,
        eps_optimal_absolute = tolerance,
        eps_optimal_relative = tolerance,
        eps_primal_infeasible = 1.0e-8,
        eps_dual_infeasible = 1.0e-8,
        time_sec_limit = time_sec_limit,
        iteration_limit = 1e6,
        kkt_matrix_pass_limit = Inf,
    )

    step_size_policy_params = cuPDLP.AdaptiveStepsizeParams(0.3,0.6)

    params = cuPDLP.PdhgParameters(
        10, # ruiz scaling iterations
        false, # off l2 scaling
        1.0, # Pock scaling coefficient: l1 norm
        1.0, # primal importance, control initial primal weight
        true, # scale initial primal weight
        3,# verbosity
        true, # record iteration stats
        64, # termination evaluation frequency
        termination_params,
        restart_params,
        step_size_policy_params,
        online_scaling,
        learning_rate,
        online_scaling_frequency,
        normalize,
    )
    
    solve_instance_and_output(
        params,
        output_directory,
        instance_path,
    )
end

function main()
    parsed_args = parse_command_line()
    instance_name = parsed_args["instance_name"]
    dataset = parsed_args["dataset"]
    experiment_name = parsed_args["experiment_name"]
    tolerance = parsed_args["tolerance"]
    time_sec_limit = parsed_args["time_sec_limit"]
    learning_rate = parsed_args["learning_rate"]
    experiment_name = "$(experiment_name)_time_$(time_sec_limit)_tol_$(tolerance)_lr_$(learning_rate)"

    if instance_name != "all"
        instance_path = joinpath("data", dataset, "$(instance_name).mps.gz")
        run_instance(instance_path, parsed_args)
    else
        problem_folder = joinpath("data", dataset)
        all_instances = readdir(problem_folder)
        len = length(all_instances)

        hard_instances = ["bnl1","bore3d","fffff800","greenbea","greenbeb","perold","pilot.ja","pilot4"]

        for i in 1:len
            instance_path = joinpath(problem_folder, all_instances[i])
            try
                instance_name = replace(basename(instance_path), r"\.(mps|MPS|qps|QPS)(\.gz)?$" => "")
                # ada_iter = get_ada_iteration(dataset, instance_name, time_sec_limit, tolerance)
                if instance_name in hard_instances
                    println("Skip instance: ", i, instance_name)
                    continue
                end
                run_instance(instance_path, parsed_args)
            catch e
                println("Error in instance: ", i, instance_path)
                println(e)
                continue
            end
        end
        write_table(joinpath("output/solver_output", dataset, experiment_name))
    end

end

main()
