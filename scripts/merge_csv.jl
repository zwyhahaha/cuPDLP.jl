
using ArgParse
using CSV
using DataFrames

function merge_table(ada_df, basic_df, os_df)
    rename!(ada_df, Dict(
        "termination_string" => "termination_string_adaPDLP",
        "iteration_count" => "iteration_count_adaPDLP",
        "solve_time_sec" => "solve_time_sec_adaPDLP",
        "cumulative_kkt_matrix_passes" => "cumulative_kkt_matrix_passes_adaPDLP",
        "cumulative_time_sec" => "cumulative_time_sec_adaPDLP",
        "time_spent_doing_basic_algorithm" => "time_spent_doing_basic_algorithm_adaPDLP"
    ))

    rename!(basic_df, Dict(
        "termination_string" => "termination_string_basicPDLP",
        "iteration_count" => "iteration_count_basicPDLP",
        "solve_time_sec" => "solve_time_sec_basicPDLP",
        "cumulative_kkt_matrix_passes" => "cumulative_kkt_matrix_passes_basicPDLP",
        "cumulative_time_sec" => "cumulative_time_sec_basicPDLP",
        "time_spent_doing_basic_algorithm" => "time_spent_doing_basic_algorithm_basicPDLP"
    ))

    rename!(os_df, Dict(
        "termination_string" => "termination_string_osPDLP",
        "iteration_count" => "iteration_count_osPDLP",
        "solve_time_sec" => "solve_time_sec_osPDLP",
        "cumulative_kkt_matrix_passes" => "cumulative_kkt_matrix_passes_osPDLP",
        "cumulative_time_sec" => "cumulative_time_sec_osPDLP",
        "time_spent_doing_basic_algorithm" => "time_spent_doing_basic_algorithm_osPDLP"
    ))

    df_merged = outerjoin(ada_df, basic_df, os_df, on=:instance_name)

    column_order = [
        "instance_name",
        "learning_rate",
        "termination_string_basicPDLP", "termination_string_adaPDLP", "termination_string_osPDLP",
        "iteration_count_basicPDLP", "iteration_count_adaPDLP", "iteration_count_osPDLP",
        "solve_time_sec_basicPDLP", "solve_time_sec_adaPDLP", "solve_time_sec_osPDLP",
        "cumulative_kkt_matrix_passes_basicPDLP", "cumulative_kkt_matrix_passes_adaPDLP", "cumulative_kkt_matrix_passes_osPDLP",
        "cumulative_time_sec_basicPDLP", "cumulative_time_sec_adaPDLP", "cumulative_time_sec_osPDLP",
        "time_spent_doing_basic_algorithm_basicPDLP", "time_spent_doing_basic_algorithm_adaPDLP", "time_spent_doing_basic_algorithm_osPDLP"
    ]

    df_merged = df_merged[:, column_order]

    return df_merged

    
end

function merge_os_dfs(dataset::String, time_limit::Float64, tolerance::Float64)
    dir_path = "output/table/$(dataset)"
    pattern  = "osPDLP_time_$(time_limit)_tol_$(tolerance)"
    files    = filter(f -> occursin(pattern, f), readdir(dir_path))

    dfs = DataFrame[]
    for file in files
        lr_match = match(r"lr_([\d\.]+)\.csv", file)
        lr_val   = (lr_match === nothing) ? missing : parse(Float64, lr_match.captures[1])
        df       = CSV.read(joinpath(dir_path, file), DataFrame)
        df[!, :learning_rate] = fill(lr_val, nrow(df))
        push!(dfs, df)
    end

    if isempty(dfs)
        return DataFrame()
    end

    merged_df = dfs[1]
    for i in 2:length(dfs)
        df = dfs[i]
        for row in eachrow(df)
            instance_name = row[:instance_name]
            idx = findfirst(isequal(instance_name), merged_df[!, :instance_name])
            if idx !== nothing
                if row[:iteration_count] < merged_df[idx, :iteration_count]
                    common_cols = intersect(names(merged_df), names(row))
                    for col in common_cols
                        merged_df[idx, col] = row[col]
                    end
                end
            else
                push!(merged_df, row)
            end
        end
    end
    return merged_df
end


function parse_command_line()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--dataset"
        help = "Path to the dataset"
        arg_type = String

        "--time_sec_limit"
        help = "Time limit for the experiments"
        arg_type = Float64

        "--tolerance"
        help = "Tolerance for the experiments"
        arg_type = Float64
    end
    return ArgParse.parse_args(s)
end

args = parse_command_line()
dataset = args["dataset"]
time_limit = args["time_sec_limit"]
tolerance = args["tolerance"]

ada_file = "output/table/$(dataset)/adaPDLP_time_$(time_limit)_tol_$(tolerance)_lr_0.0.csv"
basic_file = "output/table/$(dataset)/basicPDLP_time_$(time_limit)_tol_$(tolerance)_lr_0.0.csv"
ada_df = CSV.read(ada_file, DataFrame)
basic_df = CSV.read(basic_file, DataFrame)
os_df = merge_os_dfs(dataset, time_limit, tolerance)
df_merged = merge_table(ada_df, basic_df, os_df)
CSV.write("output/table/$(dataset)/merged_time_$(time_limit)_tol_$(tolerance).csv", df_merged)