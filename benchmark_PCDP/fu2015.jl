using DifferentialEquations
using Parameters
using JuMP, OSQP, DAQP
using ModelingToolkit
using LinearAlgebra
using Plots

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
function mpc_fu2015!(x0, xref)
    time0 = time()
    horizon = 50
    dt = 1
    N = Int(round(horizon / dt))
    Ts = [(i-1)*dt for i in 1:(N+1)] * 1.0
    Q = diagm(ones(4))
    R = Matrix(0.1I,2,2)
    xmin = 0.0
    xmax = 20.0
    umin = 0.0
    umax = 8.0

    ϵgk = 0.001
    k = 0
    r = 4
    ϵstat = 60
    ϵact = 2.5
    Δt = 1e-5

    tk = Ts
    xk = zeros(4,N)
    uk = zeros(N)
    tc = collect(0:Δt:horizon-Δt)
    xc = zeros(4,length(tc))
    uc = zeros(2,length(tc))
    while true
        N = length(Ts) - 1
        # @show Ts
        tk = Ts
        model = Model(OSQP.Optimizer)
        set_optimizer_attribute(model, "verbose", 0)
        @variable(model, u[1:2, 1:N])
        @variable(model, x[1:4, 1:N])
        
        @constraint(model, x1c2, x[1,:] .<= xmax - ϵgk)
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
            @constraint(model, x[:,i] .== Ad_mpc * x[:,i-1] + Bd_mpc * u[:,i-1])
        end

        @objective(model, Min, sum(1*(x[1,:] .- xref[1]).^2 + 1*(x[2,:] .- xref[2]).^2 + 1*(x[3,:] .- xref[3]).^2 + 1*(x[4,:] .- xref[4]).^2) + 0.1*sum((u[1,:]).^2) + 0.1*sum((u[2,:]).^2))

        optimize!(model)
        tk = Ts
        uk = value.(u)
        xk = value.(x)
        λx1c2 = -dual.(x1c2)
        tolerance = 1e-15
        λx1c2 = [abs(x) < tolerance ? 0 : x for x in λx1c2]
        # @show λx1c2
        
        status = termination_status(model)
        if status != MOI.INFEASIBLE
            xtraj = Array{ODESolution}(undef, N)
            for i in 1:N
                pendulum!(xt, ut, t) = Ac * xt + Bc * ut
                tspan = (Ts[i], Ts[i+1])
                if i == 1
                    xtraj0 = xk[:,1]
                else
                    # xtraj0 = xk[:,i]
                    xtraj0 = xtraj[i-1][end]
                end
                prob = ODEProblem(pendulum!, xtraj0, tspan, uk[:,i])
                sol = solve(prob, Tsit5())
                # @show sol[end]
                # @show xk[:,i+1]
                xtraj[i] = sol
            end

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
                        xc[:,last_idx+j] = xtraj[i](tc[last_idx+j])
                        uc[:,last_idx+j] = uk[:,i]
                    end
                end
                last_idx += L
            end

            # for i in eachindex(tc)
            #     sec_idx = 0
            #     for sec in 1:N
            #         if tc[i] >= Ts[sec] 
            #             sec_idx = sec
            #         end
            #     end
            #     xc[:,i] = xtraj[sec_idx](tc[i])
            #     uc[i] = uk[sec_idx]
            # end

            # @show maximum(xc[1,:])
            gx = xc[1,:] .- xmax
            gmax, gmax_loc = findmax(gx)
            tmax = tc[gmax_loc]
            # @show gmax

            if gmax <= 0
                println("gmax <= 0: true")
                dxdu = Array{ODESolution}(undef, N, 2, N) 
                for i in 1:N
                    for l in 1:2
                        if l == 1
                            un = [1. 0.; 0. 0.]
                        else
                            un = [0. 0.; 0. 1.]
                        end
                        for j in 1:N
                            if i == j
                                bu = 1
                            else
                                bu = 0
                            end
                            sensitivity_tank!(xt, ut, t) = Ac * xt + Bc * ut
                            tspan = (Ts[j], Ts[j+1])
                            if j == 1
                                dxdu0 = zeros(4)
                            else
                                dxdu0 = dxdu[i,l,j-1][end]
                            end
                            prob = ODEProblem(sensitivity_tank!, dxdu0, tspan, bu*un*uk[:,j])
                            sol = solve(prob, Tsit5())
                            dxdu[i,l,j] = sol
                        end
                    end
                end

                dsdu = zeros(2,N)
                for i in 1:N
                    for t in collect(Ts[i]:Δt:Ts[i+1]-Δt)
                        dsdu[:,i] += (2*R*uk[:,i] .+ 2*hcat(dxdu[i,1,i](t), dxdu[i,2,i](t))'*Q*xtraj[i](t)) * Δt
                    end
                end

                expr1 = dsdu
                expr2 = zeros(2,N)
                exprn_bool = Array{Bool}(undef, N)
                for i in 1:N
                    # @show λθc1[i] * (-1) * vec(hcat([dxdu[j,i][end][1] for j in 1:N]))
                    # λθc2[i] * (+1) * vec(hcat([dxdu[j,i][end][2] for j in 1:N]))
                    # sum([λx1c2[j] * vcat(dxdu[j,1,i][end][2], dxdu[j,2,i][end][2]) for j in 1:N])
                    expr2[:,i] += sum([λx1c2[j] * vcat(dxdu[j,1,i][end][2], dxdu[j,2,i][end][2]) for j in 1:N])
                    @show (xk[2,i] - xmax)
                    @show exprn_bool[i] = -λx1c2[i] * ϵact <= λx1c2[i] * (xk[2,i] - xmax) <= 0
                end
                # @show norm(expr1+expr2)
                expr12_bool = norm(expr1+expr2) <= ϵstat
                if expr12_bool == true && all(exprn_bool) 
                    println("OK")
                    break
                else
                    println("expr_bool: false")
                    # @show norm(expr1+expr2)
                    # @show exprn_bool
                    ϵgk = ϵgk / r
                    # break
                end
            else
                println("gmax <= 0: false")
                insert_position = searchsortedfirst(Ts, tmax)
                insert!(Ts, insert_position, tmax)
            end
        else
            println("Infeasible")
            ϵgk = ϵgk / r
        end
        k += 1
    end
    time1 = time()
    fu_time = time1 - time0
    return tk, xk, uk, tc, xc, uc, fu_time
end

#
x0 = [1.0, 15.0, 15.0, 1.0]
xref = [19.9, 19.9, 2.4, 2.4]
tk, xk, uk, tc, xc, uc, fu_time = mpc_fu2015!(x0, xref)

##
@show maximum(xc[1,:])
@show maximum(xk[1,:])
fig_x = plot(tc[1:10:end], xc[1,1:10:end])
# fig = plot(tc[30000:80000], xc[1,30000:80000])
# xlims!(fig, 9, 11)
# ylims!(fig, 19.999,20.001)
##
fig_u = plot(tc[1:10:end], uc[1,1:10:end])

##
fig_x = plot(tk[1:end-2], xk[1,:])
plot!(fig_x, tc, xc[1,:])

##
using JLD2
data = Dict("tk" => tk, "xk" => xk, "uk" => uk, "tc" => tc, "xc" => xc, "uc" => uc)
@save "data_fu2015.jld2" data




