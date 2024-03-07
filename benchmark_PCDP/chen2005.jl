using DifferentialEquations
using Parameters
using JuMP, OSQP, DAQP
using ModelingToolkit
using Plots
using LinearAlgebra

@with_kw mutable struct Para_tank2
    xmin::Float64 = -0.0
    xmax::Float64 = +20.0
    umin::Float64 = -0.0
    umax::Float64 = +8.0
    horizon::Float64 = 20
    dt::Float64 = 1.0
    Q::Matrix{Float64} = diagm(ones(4))
    R::Matrix{Float64} = Matrix(0.1I,2,2)
end
para = Para_tank2()

function linearization()
    A1 = 28
    A3 = 28
    A2 = 32
    A4 = 32

    k1_negative = 3.33
    k2_negative = 3.35
    k1_positive = 3.14
    k2_positive = 3.29

    g1_negative = 0.7
    g2_negative = 0.6
    g1_positive = 0.43
    g2_positive = 0.34

    T1_negative = 62
    T2_negative = 90
    T3_negative = 23
    T4_negative = 30
    T1_positive = 63
    T2_positive = 91
    T3_positive = 39
    T4_positive = 56


    Ac = [
    -1 / T1_negative 0 A3 / (A1 * T3_negative) 0;
    0 -1 / T2_negative 0 A4 / (A2 * T4_negative);
    0 0 -1 / T3_negative 0;
    0 0 0 -1 / T4_negative]

    Bc = [
    g1_negative * k1_negative / A1 0;
    0 g2_negative * k2_negative / A2;
    0 (1 - g2_negative) * k2_negative / A3;
    (1 - g1_negative) * k1_negative/A4 0]
    return Ac, Bc
end
Ac, Bc = linearization()

#
function mpc_chen2005!(x0, xref)
    time0 = time()
    horizon = 50
    dt = 1
    @show N = Int(round(horizon / dt))
    Ts = [(i-1)*dt for i in 1:N+1] * 1.0
    Q = diagm(ones(4))
    R = Matrix(0.1I,2,2)
    xmin = 0.0
    xmax = 20.0
    umin = 0.0
    umax = 8.0

    ϵ = 0.0005
    k = 0
    Δt = 1e-5

    tk = Ts
    xk = zeros(4,N)
    uk = zeros(N)
    tc = collect(0:Δt:horizon-Δt)
    xc = zeros(4,length(tc))
    uc = zeros(2,length(tc))
    max_k = 15
    while true
        N = length(Ts) - 1
        # @show Ts
        tk = deepcopy(Ts)
        model = Model(DAQP.Optimizer)
        set_optimizer_attribute(model, "verbose", 0)
        @variable(model, u[1:2, 1:N])
        @variable(model, x[1:4, 1:N])
        @constraint(model, x1c2, x[1,:] .<= xmax)

        @constraint(model, xmin .<= x[1,:])
        @constraint(model, xmin .<= x[2,:] .<= xmax)
        @constraint(model, xmin .<= x[3,:] .<= xmax)
        @constraint(model, xmin .<= x[4,:] .<= xmax)
        @constraint(model, umin .<= u[1,:] .<= umax)
        @constraint(model, umin .<= u[2,:] .<= umax)

        @constraint(model, x[:,1] == x0)
        for i in 2:N
            Ad_mpc = exp(Ac*(Ts[i+1]-Ts[i]))
            Bd_mpc = Ac^(-1)*(exp(Ac*(Ts[i+1]-Ts[i]))-I)*Bc
            # Ad_mpc = exp(Ac*1.0)
            # Bd_mpc = Ac^(-1)*(exp(Ac*1.0)-I)*Bc
            @constraint(model, x[:,i] .== Ad_mpc * x[:,i-1] + Bd_mpc * u[:,i-1])
        end

        @objective(model, Min, sum(1*(x[1,:] .- xref[1]).^2 + 1*(x[2,:] .- xref[2]).^2 + 1*(x[3,:] .- xref[3]).^2 + 1*(x[4,:] .- xref[4]).^2) + 0.1*sum((u[1,:]).^2) + 0.1*sum((u[2,:]).^2))

        optimize!(model)
        uk = value.(u)
        xk = value.(x)

        Ad = exp(Ac*Δt)
        Bd = Bc * Δt
        xc[:,1] = xk[:,1]
        uc[:,1] = uk[:,1]

        last_idx = 0
        for i in 1:N
            tlist = collect(Ts[i]:Δt:Ts[i+1]-Δt)
            L = length(tlist)
            for j in eachindex(tlist)
                tc[last_idx+j] = tlist[j]
                if j == 1
                    xc[:,last_idx+j] = xk[:,i]
                    uc[:,last_idx+j] = uk[:,i]
                else
                    xc[:,last_idx+j] = Ad * xc[:,last_idx+j-1] .+ Bd * uk[:,i]
                    uc[:,last_idx+j] = uk[:,i]
                end
            end
            last_idx += L
        end

        # for j in 2:length(tc)
        #     i = 0
        #     for sec in 1:N
        #         if tc[j] >= Ts[sec]
        #             i = sec
        #         end
        #     end
        #     Bd = Bc * Δt
        #     xc[:,j] = Ad * xc[:,j-1] .+ Bd * uk[:,i]
        # end

        # @show k
        violation = false
        for i in 1:N
            Ta = max(Int(round(Ts[i]/Δt)), 1)
            Tb = Int(round(Ts[i+1]/Δt))
            # @show maximum(xc[1,Ta:Tb])
            gx = xc[1,Ta:Tb] .- xmax
            # @show i
            # @show maximum(gx)
            if maximum(gx) > ϵ
                violation = true
                T_insert = (Ts[i] + Ts[i+1]) / 2
                insert_position = searchsortedfirst(Ts, T_insert)
                insert!(Ts, insert_position, T_insert)
            end
        end
        if !violation 
            println("OK")
            break
        end
        k += 1
    end
    time1 = time()
    chen_time = time1 - time0
    return tk, xk, uk, tc, xc, uc, chen_time
end



#
x0 = [1.0, 15.0, 15.0, 1.0]
xref = [19.9, 19.9, 2.4, 2.4]
tk, xk, uk, tc, xc, uc, chen_time = mpc_chen2005!(x0, xref)
@show chen_time

##
plotlyjs()
# plot(tc, xc[1,:])
@show maximum(xk[1,:])
@show maximum(xc[1,:])
@show findmax(xc[1,:])
fig_x = plot(tc[1:10:end], xc[1,1:10:end], label="")
# plot(tc, xc[1,:])
# savefig("xk.png")

##
fig_u = plot(tc[1:10:end], uc[1,1:10:end], label="")

##

fig_x = plot(tc[3900000:4100000], xc[1,3900000:4100000])


##
using JLD2
data = Dict("tk" => tk, "xk" => xk, "uk" => uk, "tc" => tc, "xc" => xc, "uc" => uc)
@save "data_chen2005.jld2" data



