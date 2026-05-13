# Parity tests against the JSON fixtures (run after building cpp/build/libaompls.so).
using AompLS
using JSON3
using Test

const REF_DIR = joinpath(@__DIR__, "..", "..", "..", "cpp", "tests", "reference")

datasets = ["BEER", "CORN", "ALPINE"]
cases = [("kfold5", false), ("kfold5_oneSE", true), ("spxy5", false)]

@testset "AompLS parity" begin
    isdir(REF_DIR) || @warn "reference directory missing — run scripts/export_reference.py"
    for ds in datasets
        path = joinpath(REF_DIR, "$ds.json")
        isfile(path) || continue
        raw = JSON3.read(read(path, String))
        for (case_name, one_se) in cases
            ref = raw[case_name]
            X = reduce(vcat, [reshape(Float64.(row), 1, :) for row in ref.X])  # (n × p)
            y = Float64.(ref.y)
            folds = [Vector{Int}(f) for f in ref.fold_test_indices]
            cfg = AOMConfig(max_components = ref.max_components,
                            cv_mode = :external,
                            external_folds = folds,
                            one_se_rule = one_se,
                            random_state = UInt(0))
            m = fit_aompls(X, y, cfg = cfg)
            @test aompls_selected_operator_name(m) == ref.selected_operator_name
            @test aompls_n_components(m) == ref.n_components_selected
            @test maximum(abs.(aompls_coef(m) .- Float64.(ref.coef))) < 1e-8
            @test abs(aompls_intercept(m) - ref.intercept) < 1e-8
            pred = predict_aompls(m, X)
            @test maximum(abs.(pred .- Float64.(ref.predictions_train))) < 1e-8
        end
    end
end
