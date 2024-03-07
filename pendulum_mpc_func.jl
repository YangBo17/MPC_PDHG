using LinearAlgebra, NumericalIntegration
using Crayons
include("./get_system_matrix_pendulum.jl")

red_text = crayon"red"

Ac,Bc,Gflat,S,H=get_system_matrix()

GPU_or_CPU = :CPU 

# global d = 3
n = 4
m = 1
Nineq = (m+n)*2
global d2=Int((d-1)/2+1)

global T = 0.2 # prediction horizon time
# global N = 20 # section number
experiment_T = 10.0
global K = Int(round(experiment_T/(T/N))) # total step
apply_N = 1
apply_T = apply_N*T/N # control application time 
max_k = K+1
T_list = collect(0:1:K) * apply_T 

p_step_size = 0.5
d_step_size = 0.2

x0 = [0.0, 0.0, 0.0, 0.0]
xref0 = [0.0, 0.0, 0.0, 0.0]

@with_kw mutable struct Para
    θmin::Float64 = -1.5
    θmax::Float64 = +1.5
    αmin::Float64 = -0.1
    αmax::Float64 = +0.1
    θdmin::Float64 = -100.0
    θdmax::Float64 = +100.0
    αdmin::Float64 = -100.0
    αdmax::Float64 = +100.0
    umin::Float64 = -15.0
    umax::Float64 = +15.0
    horizon::Float64 = 1.0
    dt::Float64 = 0.1
    Q::Vector{Float64} = [5., 1., 1., 1.]
    R::Float64 = 1.
end
para = Para()

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
    ut_coef_in
    x_test[:,end]
    return x_test[:,end]
end

function shift_warm_start(Xf_in,Xg_in,u_in)
    Xf_in=Array{Float32,4}(reshape(Xf_in, d2, d2, 2(m+n), N))
    Xg_in=Array{Float32,4}(reshape(Xg_in, d2, d2, 2(m+n), N))
    u_in=Array{Float32,3}(reshape(u_in, d+1, 2(m+n), N))
    for l in 1:N-1
        Xf_in[:,:,:,l]=Xf_in[:,:,:,l+1]
        Xg_in[:,:,:,l]=Xg_in[:,:,:,l+1]
        u_in[:,:,l]=u_in[:,:,l+1]
    end
    for l in N-1+1:N
        Xf_in[:,:,:,l]=rand(d2, d2, 2(m+n)).-0.5
        Xg_in[:,:,:,l]=rand(d2, d2, 2(m+n)).-0.5
        u_in[:,:,l]=rand(d+1, 2(m+n)).-0.5
    end
    Xf_in=Array{Float32,3}(reshape(Xf_in, d2, d2, 2(m+n)*N))
    Xg_in=Array{Float32,3}(reshape(Xg_in, d2, d2, 2(m+n)*N))
    u_in=Array{Float32,2}(reshape(u_in, d+1, 2(m+n)*N))
    return Xf_in,Xg_in,u_in
end

xrefs = zeros(d+1,n,N)
for l in 1:N
    xrefs[end,1,l]=xref0[1]
    xrefs[end,2,l]=xref0[2]
    xrefs[end,3,l]=xref0[3]
    xrefs[end,4,l]=xref0[4]
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

x=zeros(n)
ut_coef=zeros(m,d+1,N)
xt_coef=zeros(n,d+1,N)

x[:,1]=x0   

include("./get_PDHG_para_pendulum.jl")
if GPU_or_CPU == :GPU
    include("./PDHG_GPU_solver.jl")
elseif GPU_or_CPU == :CPU
    include("./PDHG_CPU_solver.jl")
end

F,G=calc_FG(T/N,d)

@time xt_coef,ut_coef,last_Xf,last_Xg,last_dual,obj=PDHG_solver(N,d,m,n,p_step_size,d_step_size,FF,μμ, F,G, 1000);
##

for l in 1:N
    xrefs[end,1,l]=xref0[1]
    xrefs[end,2,l]=xref0[2]
    xrefs[end,3,l]=xref0[3]
    xrefs[end,4,l]=xref0[4]
end

# constraints
x_min = zeros(n, d+1)
x_max = zeros(n, d+1)
x_min[1,end] = para.θmin
x_min[2,end] = para.θdmin
x_min[3,end] = para.αmin
x_min[4,end] = para.αdmin
x_max[1,end] = para.θmax
x_max[2,end] = para.θdmax
x_max[3,end] = para.αmax
x_max[4,end] = para.αdmax

u_min = zeros(m, d+1)
u_max = zeros(m, d+1)
u_min[1,end] = para.umin 
u_max[1,end] = para.umax 

include("./get_PDHG_para_pendulum.jl")
if GPU_or_CPU == :GPU
    include("./PDHG_GPU_solver.jl")
elseif GPU_or_CPU == :CPU
    include("./PDHG_CPU_solver.jl")
end

F,G=calc_FG(T/N,d)

# function mpc_pdhg(x0, xref0, last_Xf, last_Xg, last_dual,iter_num)
#     time0 = time()
#     for j in 1:n 
#         seg_ind=(N-1)*n + 3*N*(d+1) 
#         r[seg_ind+j]=x0[j]
#     end
#     xt_coef,ut_coef,last_Xf,last_Xg,last_dual,obj=PDHG_solver(N,d,m,n,p_step_size,d_step_size,FF,μμ,F,G, iter_num,true,last_Xf,last_Xg,last_dual)
#     time1 = time()
#     sdp_time = time1 - time0
#     return xt_coef,ut_coef,last_Xf,last_Xg,last_dual,obj,sdp_time
# end

function mpc_pdhg(x0, xref0, last_Xf, last_Xg, last_dual,iter_num)
    try 
        time0 = time()

        # 更新初值
        for j in 1:n 
            seg_ind=(N-1)*n + 3*N*(d+1) 
            r[seg_ind+j]=x0[j]
        end
        μμ=inv_left[n*(d+1)*N+1: n*(d+1)*N+(d+1)*2(m+n)*N,1:size(q,1)]*(-2*q)+inv_left[n*(d+1)*N+1: n*(d+1)*N+(d+1)*2(m+n)*N, size(q,1)+2(m+n)*(d+1)*N+1:end]*[-g; r]

        # ut_coef=zeros(m,d+1,N)
        # xt_coef=zeros(n,d+1,N)
        # # 更新参考轨迹
        # for l in 1:N
        #     xrefs[end,1,l]=xref0[1]
        #     xrefs[end,2,l]=xref0[2]
        #     xrefs[end,3,l]=xref0[3]
        #     xrefs[end,4,l]=xref0[4]
        # end

        # ## 使用固定reference
        # xref=zeros(d+1,n,N)
        # xref[:,1,:]=xrefs[:,1,:]
        # xref[:,2,:]=xrefs[:,2,:]
        # xref[:,3,:]=xrefs[:,3,:]
        # xref[:,4,:]=xrefs[:,4,:]
        # xref_col=zeros(n*(d+1)*N)
        # for l in 1:N
        #     xref_col[(l-1)*n*(d+1)+1:l*n*(d+1)]=(xref[:,:,l])[:]
        #     #注意这里把三维向量抻长的过程，要按行展开，即先xref[1,:,1]第一行,再xref[2,:,1]
        # end

        # q=-([xref_col; zeros(m*(d+1)*N)]'*P*[Lxcol;Lucol])'

        # last_Xf, last_Xg, last_dual = shift_warm_start(last_Xf, last_Xg, last_dual)

        xt_coef,ut_coef,last_Xf,last_Xg,last_dual,obj=PDHG_solver(N,d,m,n,p_step_size,d_step_size,FF,μμ,F,G, iter_num,false,last_Xf,last_Xg,last_dual)
        poly_time = time()
        # CUDA.reclaim()
        time1 = time()
        sdp_time = (time1 - time0)/iter_num
        return xt_coef,ut_coef,last_Xf,last_Xg,last_dual,obj,poly_time,sdp_time
    catch ex
        println(red_text("MPC thread error = $ex"))
    end
end

##
# ut_coef=zeros(m,d+1,N,2)
# xt_coef=zeros(n,d+1,N,2)

# mpc_time0 = time()
# mpc_pdhg(x0, xref0, last_Xf, last_Xg, last_dual);
# mpc_time1 = time()
# @show mpc_time1 - mpc_time0

# mpc_time0 = time()
# mpc_pdhg(x0, xref0, last_Xf, last_Xg, last_dual);
# mpc_time1 = time()
# @show mpc_time1 - mpc_time0

# mpc_time0 = time()
# xt_coef[:,:,:,1], ut_coef[:,:,:,1], lastxf, lastuf, lastdual = mpc_pdhg(x0, xref0, last_Xf, last_Xg, last_dual);
# mpc_time1 = time()
# @show mpc_time1 - mpc_time0