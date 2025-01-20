import ArgParse
import Printf
# import GZip
# import JSON3
# import JSON
# import CSV
# import DataFrames

include("/home/shanshu/opt/cuPDLP.jl/src/cuPDLP.jl")
include("json2csv.jl")

function test_preconditioner(instance_path)
    original_problem = cuPDLP.qps_reader_to_standard_form(instance_path)
    cuPDLP.validate(original_problem)
    l_inf_ruiz_iterations = 10
    l2_norm_rescaling = false
    pock_chambolle_alpha = 1.0
    verbosity = 2

    Printf.@printf("Rescaling problem on GPU...\n")

    start_gpu_rescaling_time = time()
    buffer_lp = cuPDLP.qp_cpu_to_gpu(original_problem)
    d_scaled_problem_gpu = cuPDLP.rescale_problem(
        l_inf_ruiz_iterations,
        l2_norm_rescaling,
        pock_chambolle_alpha,
        verbosity,
        buffer_lp,
    )
    gpu_rescaling_time = time() - start_gpu_rescaling_time
    Printf.@printf(
        "GPU Preconditioning Time (seconds): %.2e\n",
        gpu_rescaling_time,
    )

    start_cpu_rescaling_time = time()
    d_scaled_problem_cpu = cuPDLP.rescale_problem(
        l_inf_ruiz_iterations,
        l2_norm_rescaling,
        pock_chambolle_alpha,
        verbosity,
        original_problem,
    )
    d_scaled_problem_gpu = cuPDLP.scaledqp_cpu_to_gpu(d_scaled_problem_cpu)
    cpu_rescaling_time = time() - start_cpu_rescaling_time
    Printf.@printf(
        "CPU Preconditioning Time (seconds): %.2e\n",
        cpu_rescaling_time,
    )
    return gpu_rescaling_time, cpu_rescaling_time
end

function log_precondition_time(instance_name, dataset, gpu_rescaling_time, cpu_rescaling_time)
    output_directory = joinpath("output/precondition_test")
    if !isdir(output_directory)
        mkpath(output_directory)
    end

    output_file = joinpath(output_directory, "$(dataset).csv")
    if !isfile(output_file)
        open(output_file, "w") do io
            println(io, "instance_name,gpu_rescaling_time,cpu_rescaling_time")
        end
    end

    open(output_file, "a") do io
        println(io, "$(instance_name),$(gpu_rescaling_time),$(cpu_rescaling_time)")
    end
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
    end

    return ArgParse.parse_args(arg_parse)
end

function main()
    parsed_args = parse_command_line()
    instance_name = parsed_args["instance_name"]
    dataset = parsed_args["dataset"]

    if instance_name != "all"
        instance_path = joinpath("data", dataset, "$(instance_name).mps.gz")
        test_preconditioner(instance_path)
    else
        problem_folder = joinpath("data", dataset)
        all_instances = readdir(problem_folder)
        len = length(all_instances)

        hard_instances = ["bnl1","bore3d","capri","fffff800","greenbea","greenbeb","perold","pilot.ja","pilot4"]

        for i in 1:len
            instance_path = joinpath(problem_folder, all_instances[i])
            try
                instance_name = replace(basename(instance_path), r"\.(mps|MPS|qps|QPS)(\.gz)?$" => "")
                Printf.@printf("Testing instance: %s\n", instance_name)
                gpu_rescaling_time, cpu_rescaling_time = test_preconditioner(instance_path)
                log_precondition_time(instance_name, dataset, gpu_rescaling_time, cpu_rescaling_time)
            catch e
                println("Error in instance: ", i, instance_path)
                println(e)
                continue
            end
        end
    end

end

main()
