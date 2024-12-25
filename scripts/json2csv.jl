import ArgParse
import JSON
import CSV
import DataFrames
"""
write instance_name, termination_string, iteration_count, solve_time_sec,
cumulative_kkt_matrix_passes, cumulative_time_sec, method_specific_stats["time_spent_doing_basic_algorithm"]
output/solver_output/MIPLIB/adaPDLP_time_60.0_tol_0.0001_lr_0.0/acc-tight2/acc-tight2_summary.json
"""
function write_table(json_dir)
    csv_file = replace(json_dir, "solver_output" => "table")
    csv_file = "$(csv_file).csv"
    rows = []
    for subdir in filter(x -> isdir(joinpath(json_dir, x)), readdir(json_dir))
        json_files = filter(x -> endswith(x, ".json"), readdir(joinpath(json_dir, subdir)))
        for json_file in json_files
            json_file = joinpath(json_dir, subdir, json_file)
            if isfile(json_file)
                json_string = read(json_file, String)
                data = JSON.parse(json_string)
                if !isa(data, Dict)
                    error("Expected a dictionary at the top level of the JSON file")
                end

                solution_stats = get(data, "solution_stats", Dict())
                cumulative_kkt_matrix_passes = get(solution_stats, "cumulative_kkt_matrix_passes", missing)
                cumulative_time_sec = get(solution_stats, "cumulative_time_sec", missing)
                method_specific_stats = get(solution_stats, "method_specific_stats", Dict())
                time_spent_doing_basic_algorithm = get(method_specific_stats, "time_spent_doing_basic_algorithm", missing)

                row = Dict(
                    "instance_name" => subdir,
                    "termination_string" => get(data, "termination_string", missing),
                    "iteration_count" => get(data, "iteration_count", missing),
                    "solve_time_sec" => get(data, "solve_time_sec", missing),
                    "cumulative_kkt_matrix_passes" => cumulative_kkt_matrix_passes,
                    "cumulative_time_sec" => cumulative_time_sec,
                    "time_spent_doing_basic_algorithm" => time_spent_doing_basic_algorithm
                )
                push!(rows, row)
            end
        end
    end
    
    # CSV.write(csv_file, rows)
    df = DataFrames.DataFrame(rows)
    column_order = ["instance_name", "termination_string", "iteration_count", "solve_time_sec", "cumulative_kkt_matrix_passes", "cumulative_time_sec", "time_spent_doing_basic_algorithm"]
    df = df[:, column_order]

    CSV.write(csv_file, df)
end

function main()
    json_dirs = [
        # "output/solver_output/MIPLIB/adaPDLP_time_60.0_tol_0.0001_lr_0.0",
        "output/solver_output/MIPLIB/basicPDLP_time_60.0_tol_0.0001_lr_0.0",
        "output/solver_output/MIPLIB/osPDLP_time_60.0_tol_0.0001_lr_0.1",
    ]
    for json_dir in json_dirs
        write_table(json_dir)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end