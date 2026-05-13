# SPDX-License-Identifier: CeCILL-2.1
# Julia binding for aompls (AOM-PLS compact, PLS1). Loads libaompls.so/.dylib/.dll
# from cpp/build/ and calls the C ABI declared in cpp/include/aompls/c_api.h.

module AompLS

using Libdl

export AOMConfig, AOMModel, fit_aompls, predict_aompls

# ---------------------------------------------------------------------------
# Library discovery + lazy load
# ---------------------------------------------------------------------------

const _LIB_REF = Ref{Ptr{Cvoid}}(C_NULL)

function _candidates()
    here = @__DIR__
    cpp_build = joinpath(here, "..", "..", "..", "cpp", "build")
    names = [
        joinpath(cpp_build, "libaompls.so"),
        joinpath(cpp_build, "libaompls.dylib"),
        joinpath(cpp_build, "libaompls.dll"),
        joinpath(cpp_build, "aompls.dll"),
    ]
    return filter(isfile, names)
end

function _libhandle()
    if _LIB_REF[] != C_NULL
        return _LIB_REF[]
    end
    cands = _candidates()
    isempty(cands) && error("libaompls not found. Build from cpp/ with:\n" *
                            "  g++ -std=c++17 -O3 -fPIC -shared -I include " *
                            "-DEIGEN_NO_DEBUG -DEIGEN_DONT_PARALLELIZE " *
                            "src/c_api.cpp -o build/libaompls.so")
    _LIB_REF[] = Libdl.dlopen(first(cands))
    return _LIB_REF[]
end

_sym(name::Symbol) = Libdl.dlsym(_libhandle(), name)

# ---------------------------------------------------------------------------
# Layout of aompls_config_t — MUST match cpp/include/aompls/c_api.h.
# Note: C uses native int + unsigned long long (=Culonglong) + double + pointers.
# ---------------------------------------------------------------------------

mutable struct CConfig
    max_components::Cint
    n_folds::Cint
    cv_mode::Cint
    one_se_rule::Cint
    center::Cint
    random_state::Culonglong
    preproc::Cint
    osc_n_components::Cint
    asls_lam::Cdouble
    asls_p::Cdouble
    asls_n_iter::Cint
    external_folds_flat::Ptr{Cint}
    external_fold_sizes::Ptr{Cint}
    n_external_folds::Cint
end

Base.@kwdef struct AOMConfig
    max_components::Int = 15
    n_folds::Int = 5
    cv_mode::Symbol = :kfold      # :kfold | :spxy | :holdout | :external
    one_se_rule::Bool = false
    center::Bool = true
    random_state::UInt = 0
    preproc::Symbol = :none       # :none :snv :msc :osc :asls :snvosc (=:snv+osc) :aslsosc (=:asls+osc)
    osc_n_components::Int = 1
    asls_lam::Float64 = 1e5
    asls_p::Float64 = 0.01
    asls_n_iter::Int = 10
    external_folds::Union{Nothing,Vector{Vector{Int}}} = nothing
end

const _CV_INT = Dict(:kfold => 0, :spxy => 1, :holdout => 2, :external => 3)
const _PREPROC_INT = Dict(:none => 0, :snv => 1, :msc => 2, :osc => 3, :asls => 4,
                          :snvosc => 5, Symbol("snv+osc") => 5,
                          :aslsosc => 6, Symbol("asls+osc") => 6)

function _to_cconfig(cfg::AOMConfig, scratch::Ref{Vector{Cint}}, sizes::Ref{Vector{Cint}})
    cv = get(_CV_INT, cfg.cv_mode, nothing)
    cv === nothing && error("unknown cv_mode: $(cfg.cv_mode)")
    pp = get(_PREPROC_INT, cfg.preproc, nothing)
    pp === nothing && error("unknown preproc: $(cfg.preproc)")
    flat_ptr = Ptr{Cint}(C_NULL)
    sizes_ptr = Ptr{Cint}(C_NULL)
    n_folds = cfg.n_folds
    if cv == 3
        cfg.external_folds === nothing && error("cv_mode=:external requires external_folds")
        flat = Cint[]
        sz = Cint[]
        for f in cfg.external_folds
            push!(sz, Cint(length(f)))
            append!(flat, Cint.(f))
        end
        scratch[] = flat
        sizes[] = sz
        flat_ptr = pointer(scratch[])
        sizes_ptr = pointer(sizes[])
        n_folds = length(cfg.external_folds)
    end
    return CConfig(
        Cint(cfg.max_components), Cint(n_folds), Cint(cv),
        Cint(cfg.one_se_rule), Cint(cfg.center), Culonglong(cfg.random_state),
        Cint(pp), Cint(cfg.osc_n_components),
        Cdouble(cfg.asls_lam), Cdouble(cfg.asls_p), Cint(cfg.asls_n_iter),
        flat_ptr, sizes_ptr, Cint(n_folds))
end

# ---------------------------------------------------------------------------
# Model wrapper with finalizer
# ---------------------------------------------------------------------------

mutable struct AOMModel
    handle::Ptr{Cvoid}
    function AOMModel(handle::Ptr{Cvoid})
        m = new(handle)
        finalizer(m -> begin
            if m.handle != C_NULL
                ccall(_sym(:aompls_free), Cvoid, (Ptr{Cvoid},), m.handle)
                m.handle = C_NULL
            end
        end, m)
        return m
    end
end

function _row_major(X::AbstractMatrix{<:Real})
    Xd = Matrix{Float64}(X)
    n, p = size(Xd)
    out = Vector{Float64}(undef, n * p)
    @inbounds for i in 1:n, j in 1:p
        out[(i - 1) * p + j] = Xd[i, j]
    end
    return out, n, p
end

function _check_err(err_ref::Ref{Ptr{Cchar}})
    if err_ref[] != C_NULL
        msg = unsafe_string(err_ref[])
        ccall(_sym(:aompls_free_string), Cvoid, (Ptr{Cchar},), err_ref[])
        error("aompls: $msg")
    end
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    fit_aompls(X, y; cfg = AOMConfig())

Fit AOM-PLS (compact, PLS1). Returns an `AOMModel` handle.
"""
function fit_aompls(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                    cfg::AOMConfig = AOMConfig())
    Xrm, n, p = _row_major(X)
    yd = Vector{Float64}(y)
    length(yd) == n || throw(ArgumentError("length(y) must match nrow(X)"))
    scratch = Ref(Cint[])
    sizes = Ref(Cint[])
    cc = _to_cconfig(cfg, scratch, sizes)
    err_ref = Ref{Ptr{Cchar}}(C_NULL)
    GC.@preserve scratch sizes Xrm yd begin
        handle = ccall(_sym(:aompls_fit),
                       Ptr{Cvoid},
                       (Ptr{Cdouble}, Cint, Cint, Ptr{Cdouble}, Ref{CConfig}, Ref{Ptr{Cchar}}),
                       Xrm, Cint(n), Cint(p), yd, Ref(cc), err_ref)
        _check_err(err_ref)
        handle == C_NULL && error("aompls_fit returned NULL without an error message")
        return AOMModel(handle)
    end
end

"""
    predict_aompls(model::AOMModel, X)

Predict y for new samples. Returns a `Vector{Float64}` of length `size(X, 1)`.
"""
function predict_aompls(model::AOMModel, X::AbstractMatrix{<:Real})
    Xrm, n, p = _row_major(X)
    p == aompls_n_features(model) || throw(ArgumentError(
        "X has $p features, model expects $(aompls_n_features(model))"))
    out = Vector{Float64}(undef, n)
    err_ref = Ref{Ptr{Cchar}}(C_NULL)
    GC.@preserve Xrm out begin
        ret = ccall(_sym(:aompls_predict),
                    Cint,
                    (Ptr{Cvoid}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ref{Ptr{Cchar}}),
                    model.handle, Xrm, Cint(n), out, err_ref)
        _check_err(err_ref)
        ret == 0 || error("aompls_predict failed")
    end
    return out
end

# Convenience accessors
aompls_n_features(m::AOMModel) = Int(ccall(_sym(:aompls_n_features), Cint, (Ptr{Cvoid},), m.handle))
aompls_n_components(m::AOMModel) = Int(ccall(_sym(:aompls_n_components), Cint, (Ptr{Cvoid},), m.handle))
aompls_selected_operator_index(m::AOMModel) = Int(ccall(_sym(:aompls_selected_operator_index), Cint, (Ptr{Cvoid},), m.handle))

function aompls_selected_operator_name(m::AOMModel)
    cstr = ccall(_sym(:aompls_selected_operator_name), Ptr{Cchar}, (Ptr{Cvoid},), m.handle)
    return unsafe_string(cstr)
end

function aompls_coef(m::AOMModel)
    p = aompls_n_features(m)
    out = Vector{Float64}(undef, p)
    ccall(_sym(:aompls_get_coef), Cvoid, (Ptr{Cvoid}, Ptr{Cdouble}), m.handle, out)
    return out
end

aompls_intercept(m::AOMModel) = ccall(_sym(:aompls_get_intercept), Cdouble, (Ptr{Cvoid},), m.handle)

function aompls_rmse_curves(m::AOMModel)
    n_ops = Ref{Cint}(0); n_k = Ref{Cint}(0)
    ccall(_sym(:aompls_get_rmse_curves), Cint,
          (Ptr{Cvoid}, Ref{Cint}, Ref{Cint}, Ptr{Cdouble}),
          m.handle, n_ops, n_k, C_NULL)
    n_ops[] == 0 && return zeros(0, 0)
    buf = Vector{Float64}(undef, n_ops[] * n_k[])
    ccall(_sym(:aompls_get_rmse_curves), Cint,
          (Ptr{Cvoid}, Ref{Cint}, Ref{Cint}, Ptr{Cdouble}),
          m.handle, n_ops, n_k, buf)
    return permutedims(reshape(buf, Int(n_k[]), Int(n_ops[])))  # → (n_ops × K)
end

export aompls_coef, aompls_intercept, aompls_n_components, aompls_n_features,
       aompls_selected_operator_index, aompls_selected_operator_name, aompls_rmse_curves

end # module
