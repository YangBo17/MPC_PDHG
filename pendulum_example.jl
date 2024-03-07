using LinearAlgebra, NumericalIntegration
using DynamicPolynomials, Plots
# using Polynomials
using Profile
# using ProfileView

include("./get_system_matrix_pendulum.jl")

Ac,Bc,Gflat,S,H=get_system_matrix()

global d = 5
n = 4
m = 1
Nineq = (m+n)*2
global d2=Int((d-1)/2+1)

global T = 1 # prediction horizon time
global N = 50 # section number
experiment_T = 2.0
global K = Int(round(experiment_T/(T/N))) # total step
apply_N = 1
apply_T = apply_N*T/N # control application time 
max_k = K+1
T_list = collect(0:1:K) * apply_T

x0 = [0.0, 0.0, 0.0, 0.0]
# reference
xrefs = zeros(d+1,n,N)
for l in 1:N
    xrefs[end,1,l]=0.5
    xrefs[end,2,l]=0.0
    xrefs[end,3,l]=0.0
    xrefs[end,4,l]=0.0
end

# constraints
x_min = zeros(n, d+1)
x_max = zeros(n, d+1)
x_min[1,end] = para.θmin
x_min[2,end] = para.αmin
x_min[3,end] = para.θdmin
x_min[4,end] = para.αdmin
x_max[1,end] = para.θmax
x_max[2,end] = para.αmax
x_max[3,end] = para.θdmax
x_max[4,end] = para.αdmax

u_min = zeros(m, d+1)
u_max = zeros(m, d+1)
u_min[1,end] = para.umin 
u_max[1,end] = para.umax 

p_step_size = 0.4
d_step_size = 0.2

include("./get_PDHG_para_pendulum.jl")
include("./PDHG_GPU_solver.jl")

function apply_poly_u(ut_coef_in,xk,apply_T)
    Δt = 1e-5
    Ad=exp(Ac*Δt)
    apply_steps=Int(round(apply_T/Δt))
    x_test=zeros(n,apply_steps+1)
    x_test[:,1]=xk
    for k in 1:apply_steps
        basis=zeros(d+1)
        for i in 1:d+1
            basis[d+2-i]=(k*Δt)^(i)/(i)-((k-1)*Δt)^(i)/(i)
        end
        delta_Bu=Bc*ut_coef_in*basis
        x_test[:,k+1]=Ad*x_test[:,k]+delta_Bu
    end
    @show ut_coef_in
    x_test[:,end]
    return x_test[:,end]
end

function shift_warm_start(Xf_in,Xg_in,u_in)
    Xf_in=Array{Float32,4}(reshape(Xf_in, d2, d2, 2(m+n), N))
    Xg_in=Array{Float32,4}(reshape(Xg_in, d2, d2, 2(m+n), N))
    u_in=Array{Float32,3}(reshape(u_in, d+1, 2(m+n), N))
    for l in 1:N-apply_N
        Xf_in[:,:,:,l]=Xf_in[:,:,:,l+apply_N]
        Xg_in[:,:,:,l]=Xg_in[:,:,:,l+apply_N]
        u_in[:,:,l]=u_in[:,:,l+1]
    end
    for l in N-apply_N+1:N
        Xf_in[:,:,:,l]=rand(d2, d2, 2(m+n)).-0.5
        Xg_in[:,:,:,l]=rand(d2, d2, 2(m+n)).-0.5
        u_in[:,:,l]=rand(d+1, 2(m+n)).-0.5
    end
    Xf_in=Array{Float32,3}(reshape(Xf_in, d2, d2, 2(m+n)*N))
    Xg_in=Array{Float32,3}(reshape(Xg_in, d2, d2, 2(m+n)*N))
    u_in=Array{Float32,2}(reshape(u_in, d+1, 2(m+n)*N))
    return Xf_in,Xg_in,u_in
end

global k=1

x=zeros(n,max_k)
ut_coef=zeros(m,d+1,N,max_k)
xt_coef=zeros(n,d+1,N,max_k)

x[:,1]=x0   

F,G=calc_FG(T/N,d)

while k <= max_k
    global k, last_Xf, last_Xg, last_dual, inv_left, FF, μμ, obj

    @show k

    if k==1
        @time xt_coef[:,:,:,k],ut_coef[:,:,:,k],last_Xf,last_Xg,last_dual,obj=PDHG_solver(N,d,m,n,p_step_size,d_step_size,FF,μμ, F,G, 5000)

    else
        for j in 1:n
            seg_ind=(N-1)*n + 3*N*(d+1) 
            r[seg_ind+j]=x[j,k]
        end
        μμ=inv_left[n*(d+1)*N+1: n*(d+1)*N+(d+1)*2(m+n)*N,1:size(q,1)]*(-2*q)+inv_left[n*(d+1)*N+1: n*(d+1)*N+(d+1)*2(m+n)*N, size(q,1)+2(m+n)*(d+1)*N+1:end]*[-g; r]

        # last_Xf, last_Xg, last_dual =shift_warm_start(last_Xf, last_Xg, last_dual)

        @time xt_coef[:,:,:,k],ut_coef[:,:,:,k],last_Xf,last_Xg,last_dual,obj=PDHG_solver(N,d,m,n,p_step_size,d_step_size,FF,μμ,F,G, 2000,true,last_Xf,last_Xg,last_dual)
    end

    x[:,k+1]=apply_poly_u(ut_coef[:,:,1,k],x[:,k],apply_T)
    @show obj
    k+=1
    if k>=max_k
        break
    end
end
#
labels = ["θ","θd","α","αd"]
fig = plot()
for i in eachindex(x'[1,:])
    plot!(fig, T_list, x'[:,i], label=labels[i], legend=:outertopright)
end
# xlims!(fig, 0, 300)
# ylims!(fig, -2, 2)
plot(fig)

##


##
@polyvar t
n_points=101 # how many points in a segment
parallel_u=zeros(m,n_points*N,max_k)
parallel_x=zeros(n,n_points*N,max_k)
for k in 1:max_k
    for i in 1:m # inputs
    for j = 1:N # record how many segments
        pu=dot(monomials([t], 0:d),(ut_coef[i,:,j,k]))
        x_list=0:1/(n_points-1):1
        for kk in 1:n_points
            parallel_u[i,(j-1)*n_points+kk,k]=pu(t=>x_list[kk])
        end
    end
    end
    for i in 1:n # states
    for j = 1:N # record how many segments
        px=dot(monomials([t], 0:d),(xt_coef[i,:,j,k]))
        x_list=0:1/(n_points-1):1
        for kk in 1:n_points
            parallel_x[i,(j-1)*n_points+kk,k]=px(t=>x_list[kk])
        end
    end
    end
end
# plot inputs
plot()
for k in 1:max_k
    for i in 1:1 # plot how many segments
        time_axis=(k-1)+(i-1)*1:1/(n_points-1):(k-1)+i*1
        for j = 1:m
            plot!(time_axis,parallel_u[j,(i-1)*n_points+1:i*n_points,k],label="")
        end
    end
end
plot!()
# savefig("u_poly.png")
## plot states
plot()
for k in 1:max_k
    for i in 1:1 # plot how many segments
        time_axis=(k-1)+(i-1)*1:1/(n_points-1):(k-1)+i*1
        for j = 1:n
            plot!(time_axis,parallel_x[j,(i-1)*n_points+1:i*n_points,k],label="")
        end
    end
end
plot!()
# savefig("x_poly.png")
