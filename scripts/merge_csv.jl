
using ArgParse
using CSV
using DataFrames

function merge_table(os_df, ada_df, ifos_df)
    rename!(ada_df, Dict(
        "termination_string" => "termination_string_adaPDLP",
        "iteration_count" => "iteration_count_adaPDLP",
        "solve_time_sec" => "solve_time_sec_adaPDLP",
        "cumulative_kkt_matrix_passes" => "cumulative_kkt_matrix_passes_adaPDLP",
        "cumulative_time_sec" => "cumulative_time_sec_adaPDLP",
        "time_spent_doing_basic_algorithm" => "time_spent_doing_basic_algorithm_adaPDLP"
    ))

    rename!(ifos_df, Dict(
        "termination_string" => "termination_string_ifosPDLP",
        "iteration_count" => "iteration_count_ifosPDLP",
        "solve_time_sec" => "solve_time_sec_ifosPDLP",
        "cumulative_kkt_matrix_passes" => "cumulative_kkt_matrix_passes_ifosPDLP",
        "cumulative_time_sec" => "cumulative_time_sec_ifosPDLP",
        "time_spent_doing_basic_algorithm" => "time_spent_doing_basic_algorithm_ifosPDLP",
        "learning_rate" => "learning_rate_ifosPDLP"
    ))

    rename!(os_df, Dict(
        "termination_string" => "termination_string_osPDLP",
        "iteration_count" => "iteration_count_osPDLP",
        "solve_time_sec" => "solve_time_sec_osPDLP",
        "cumulative_kkt_matrix_passes" => "cumulative_kkt_matrix_passes_osPDLP",
        "cumulative_time_sec" => "cumulative_time_sec_osPDLP",
        "time_spent_doing_basic_algorithm" => "time_spent_doing_basic_algorithm_osPDLP",
        "learning_rate" => "learning_rate_osPDLP"
    ))

    df_merged = outerjoin(ada_df, ifos_df, os_df, on=:instance_name)

    column_order = [
        "instance_name",
        "learning_rate_osPDLP", "learning_rate_ifosPDLP",
        "termination_string_adaPDLP", "termination_string_osPDLP", "termination_string_ifosPDLP",
        "iteration_count_adaPDLP", "iteration_count_osPDLP", "iteration_count_ifosPDLP",
        "solve_time_sec_adaPDLP", "solve_time_sec_osPDLP", "solve_time_sec_ifosPDLP",
        "cumulative_kkt_matrix_passes_adaPDLP", "cumulative_kkt_matrix_passes_osPDLP", "cumulative_kkt_matrix_passes_ifosPDLP",
        "cumulative_time_sec_adaPDLP", "cumulative_time_sec_osPDLP", "cumulative_time_sec_ifosPDLP",
        "time_spent_doing_basic_algorithm_adaPDLP", "time_spent_doing_basic_algorithm_osPDLP", "time_spent_doing_basic_algorithm_ifosPDLP"
    ]

    df_merged = df_merged[:, column_order]

    return df_merged

    
end

function merge_os_dfs(experiment::String, dataset::String, time_limit::Float64, tolerance::Float64)
    dir_path = "output/table/$(dataset)"
    pattern  = "$(experiment)_time_$(time_limit)_tol_$(tolerance)"
    files    = filter(f -> occursin(pattern, f), readdir(dir_path))

    dfs = DataFrame[]
    for file in files
        # lr_match = match(r"lr_([\d\.]+)\.csv", file)
        lr_match = match(r"lr_([\d\.eE+-]+)\.csv", file)
        lr_val   = (lr_match === nothing) ? missing : parse(Float64, lr_match.captures[1])
        df       = CSV.read(joinpath(dir_path, file), DataFrame; stringtype=String)
        df[!, :learning_rate] = fill(lr_val, nrow(df))
        push!(dfs, df)
    end

    if isempty(dfs)
        return DataFrame()
    end

    # merged_df = dfs[1]
    # for i in 2:length(dfs)
    #     df = dfs[i]
    #     for row in eachrow(df)
    #         instance_name = row[:instance_name]
    #         idx = findfirst(isequal(instance_name), merged_df[!, :instance_name])
    #         if idx !== nothing
    #             if row[:iteration_count] < merged_df[idx, :iteration_count]
    #                 common_cols = intersect(names(merged_df), names(row))
    #                 for col in common_cols
    #                     merged_df[idx, col] = row[col]
    #                     # if typeof(row[col]) == String && length(row[col]) > 15
    #                     #     merged_df[idx, col] = row[col][1:15]
    #                     # else
    #                     #     merged_df[idx, col] = row[col]
    #                     # end
    #                 end
    #             end
    #         else
    #             # for col in names(row)
    #             #     if typeof(row[col]) == String && length(row[col]) > 15
    #             #         row[col] = row[col][1:15]
    #             #     end
    #             # end
    #             push!(merged_df, row)
    #         end
    #     end
    # end
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
                        merged_df[idx, col] = coalesce(row[col], merged_df[idx, col])
                    end
                end
            else
                # new_row = DataFrame()
                # for col in names(row)
                #     new_row[!, Symbol(col)] = [coalesce(row[col], missing)]
                # end
                # push!(merged_df, new_row[1, :])
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
# basic_file = "output/table/$(dataset)/basicPDLP_time_$(time_limit)_tol_$(tolerance)_lr_0.0.csv"
ada_df = CSV.read(ada_file, DataFrame)
# basic_df = CSV.read(basic_file, DataFrame)
os_df = merge_os_dfs("nonPDLP",dataset, time_limit, tolerance)
ifos_df = merge_os_dfs("hyperPDLP",dataset, time_limit, tolerance)
df_merged = merge_table(os_df, ada_df, ifos_df)
CSV.write("output/table/$(dataset)/nonhyper_merged_time_$(time_limit)_tol_$(tolerance).csv", df_merged)