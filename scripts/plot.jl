using GZip
using JSON3
using Plots

function read_log_file(file_path::String)
    log_data = ""
    GZip.open(file_path, "r") do io
        log_data = read(io, String)
    end
    log_data = replace(log_data, "Infinity" => "\"Infinity\"", "-Infinity" => "\"-Infinity\"")
    return JSON3.read(log_data)
end

function compute_kkt_error(
    primal_objective::Float64,
    dual_objective::Float64,
    l2_primal_residual::Float64,
    l2_dual_residual::Float64,
)
    return sqrt(
    l2_primal_residual^2+
    l2_dual_residual^2+
    max(0,primal_objective-dual_objective)^2
    )
end

function plot_iterates(log_file_path::String)
    log = read_log_file(log_file_path)
    
    iteration_stats = log["iteration_stats"]
    iteration_count = length(iteration_stats)

    kkt_errors = []
    for stat in iteration_stats
        for info in stat["convergence_information"]  # Iterate over the convergence_information vector
            kkt_error = compute_kkt_error(
            Float64(info["primal_objective"]),
            Float64(info["dual_objective"]),
            Float64(info["l2_primal_residual"]),
            Float64(info["l2_dual_residual"])
        )
            push!(kkt_errors, kkt_error)
        end
    end
    iterates = 1:iteration_count
    kkt_plt = Plots.plot()
    Plots.plot!(kkt_plt, 
    iterates, 
    kkt_errors, 
    label="KKT error", 
    xlabel="Iteration", 
    ylabel="KKT error", 
    title="Iteration vs KKT error")
    Plots.savefig(kkt_plt, "output/figure/adlittle.png")
end

plot_iterates("output/solver_output/MIPLIB/neos5_full_log.json.gz")