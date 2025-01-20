import ArgParse
import GZip
import JSON3

import cuPDLP
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
        cuPDLP.ConstantStepsizeParams(), # adaptive stepsize parameters (reduction, growth exponent)
        true, # online scaling
        0.1, # online learning rate
        false, # adaptive primal weight
    )

    cuPDLP.optimize(params_warmup, lp);
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

function get_iteration_limit(dataset, instance_name, time_sec_limit, tolerance)
    experiment_name = "adaPDLP_time_$(time_sec_limit)_tol_$(tolerance)_lr_0.0"
    ada_directory = joinpath("output/solver_output", dataset, experiment_name, instance_name)
    if !isdir(ada_directory)
        ada_iter =  typemax(Int32)
    else
        ada_json_file = filter(x -> endswith(x, ".json"), readdir(ada_directory))
        ada_json_file = joinpath(ada_directory, ada_json_file[1])
        ada_iter = get_iteration_from_json(ada_json_file)
    end

    experiment_name = "basicPDLP_time_$(time_sec_limit)_tol_$(tolerance)_lr_0.0"
    basic_directory = joinpath("output/solver_output", dataset, experiment_name, instance_name)
    if !isdir(basic_directory)
        basic_iter =  typemax(Int32)
    else
        basic_json_file = filter(x -> endswith(x, ".json"), readdir(basic_directory))
        basic_json_file = joinpath(basic_directory, basic_json_file[1])
        basic_iter = get_iteration_from_json(basic_json_file)
    end

    return min(ada_iter, basic_iter) * 5
end

function parse_command_line()
    arg_parse = ArgParse.ArgParseSettings()

    ArgParse.@add_arg_table! arg_parse begin

        "--dataset"
        help = "dataset name"
        arg_type = String
        required = true

        "--experiment_name"
        help = "experiment name."
        arg_type = String
        required = true

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
        default = 0.0

    end

    return ArgParse.parse_args(arg_parse)
end


function main()
    parsed_args = parse_command_line()
    dataset = parsed_args["dataset"]
    tolerance = parsed_args["tolerance"]
    time_sec_limit = parsed_args["time_sec_limit"]
    experiment_name = parsed_args["experiment_name"]
    learning_rate = parsed_args["learning_rate"]

    if experiment_name == "adaPDLP"
        adaptive_step_size = true
        adaptive_primal_weight = true
        online_scaling = false
        learning_rate = 0.0
        iteration_limit = typemax(Int32)
    elseif experiment_name == "osPDLP"
        adaptive_step_size = false
        adaptive_primal_weight = false
        online_scaling = true
        iteration_limit = typemax(Int32)
        
    elseif experiment_name == "basicPDLP"
        adaptive_step_size = false
        adaptive_primal_weight = false
        online_scaling = false
        learning_rate = 0.0
        iteration_limit = typemax(Int32)
    else
        error("Unknown experiment name: ", experiment_name)
    end

    experiment_name = "$(experiment_name)_time_$(time_sec_limit)_tol_$(tolerance)_lr_$(learning_rate)"
    
    problem_folder = joinpath("data", dataset)
    all_instances = readdir(problem_folder)
    len = length(all_instances)

    for i in 1:len
        instance_path = joinpath(problem_folder, all_instances[i])
        instance_name = replace(basename(instance_path), r"\.(mps|MPS|qps|QPS)(\.gz)?$" => "")
        try
            output_directory = joinpath("output/solver_output", dataset, experiment_name,instance_name)

            if !isdir(output_directory)
                mkpath(output_directory)
            else
                println("Instance already solved: ", i, instance_path)
                continue
            end

            # if learning_rate > 0.0
            #     iteration_limit = get_iteration_limit(dataset, instance_name, time_sec_limit, tolerance)
            # end

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
                iteration_limit = iteration_limit, # no iteration limit
                kkt_matrix_pass_limit = Inf,
            )

            if adaptive_step_size
                step_size_policy_params = cuPDLP.AdaptiveStepsizeParams(0.3,0.6)
            else
                step_size_policy_params = cuPDLP.ConstantStepsizeParams()
            end

            params = cuPDLP.PdhgParameters(
                10, # ruiz scaling iterations
                false, # off l2 scaling
                1.0, # Pock scaling coefficient: l1 norm
                1.0, # primal importance, control initial primal weight
                true, # scale initial primal weight
                2,# verbosity
                true, # record iteration stats
                64, # termination evaluation frequency
                termination_params,
                restart_params,
                step_size_policy_params,
                online_scaling,
                learning_rate,
                adaptive_primal_weight,
            )
            
            lp = cuPDLP.qps_reader_to_standard_form(instance_path)
            
            if i == 1
                println("Warm up start")
                oldstd = stdout
                redirect_stdout(devnull) # suppress output
                warm_up(lp); # solve the LP for 100 iterations
                redirect_stdout(oldstd)
                println("Warm up done")
            end
            
            solve_instance_and_output(
                params,
                output_directory,
                instance_path,
            )
        catch e
            println("Error in instance: ", i, instance_path)
            println(e)
            continue
        end
    end
    write_table(joinpath("output/solver_output", dataset, experiment_name))
end

main()
