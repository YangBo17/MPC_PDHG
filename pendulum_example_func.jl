
using Plots
function benchmark(dd, NN)
    global d = dd
    global N = NN
    include("pendulum_mpc_func.jl")

    x0 = [0.0, 0.0, 0.0, 0.0]
    xref0 = [0.0, 0.0, 0.0, 0.0]

    xrefs = zeros(d+1,n,N)

    for l in 1:N
        xrefs[end,1,l]=xref0[1]
        xrefs[end,2,l]=xref0[2]
        xrefs[end,3,l]=xref0[3]
        xrefs[end,4,l]=xref0[4]
    end

    max_k = 100
    x=zeros(n,max_k)
    tk = collect(0:1:max_k-1)
    ut_coef=zeros(m,d+1,N,max_k)
    xt_coef=zeros(n,d+1,N,max_k)
    x[:,1] = x0

    global k = 1
    sdp_times = []
    time0 = time()
    my_times = []
    iter_num = 40
    while true
        global k, last_Xf, last_Xg, last_dual, inv_left, FF, μμ, obj, x0

        # @show k

        if k==1
            xt_coef[:,:,:,k],ut_coef[:,:,:,k],last_Xf,last_Xg,last_dual,obj,poly_time,sdp_time=mpc_pdhg(x0, xref0, last_Xf, last_Xg, last_dual, iter_num)
            my_time = time()-time0
            # @show sdp_time
            push!(my_times, my_time)
            push!(sdp_times, sdp_time)
        else
            xt_coef[:,:,:,k],ut_coef[:,:,:,k],last_Xf,last_Xg,last_dual,obj,poly_time,sdp_time=mpc_pdhg(x0, xref0, last_Xf, last_Xg, last_dual, iter_num)
            my_time = time()-time0
            # @show sdp_time
            push!(my_times, my_time)
            push!(sdp_times, sdp_time)
        end

        x[:,k+1]=apply_poly_u(ut_coef[:,:,1,k],x[:,k],apply_T)
        x0 = x[:,k+1]
        # @show obj
        k+=1
        if k >= max_k
            break
        end
    end
    my_times
    sdp_times
    cpu_times = sum(sdp_times[10:90])/80*1e3
    return cpu_times
end


Ns = [1, 2, 6, 10, 20, 30, 40]
ds = [3, 5, 7]
total_time = zeros(lastindex(Ns), lastindex(ds))
for N_index in 1:lastindex(Ns)
    for d_index in 1:lastindex(ds)
        @show NN = Ns[N_index]
        @show dd = ds[d_index]
        c_time = benchmark(dd, NN)
        @show total_time[N_index, d_index] = c_time
    end
end

##
labels = ["θ","α","θd","αd"]
fig = plot()
for i in eachindex(x'[1,:])
    plot!(fig, tk, x'[:,i], label=labels[i], legend=:outertopright)
end
# xlims!(fig, 0, 300)
# ylims!(fig, -2, 2)
plot(fig)

##
fig_t = plot(my_times, sdp_times, label="sdp time")
xlims!(fig_t, 1, 6)
ylims!(fig_t, 0, 0.04)

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
for k in 1:5
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
for k in 1:5
    for i in 1:1 # plot how many segments
        time_axis=(k-1)+(i-1)*1:1/(n_points-1):(k-1)+i*1
        for j = 1:1
            plot!(time_axis,parallel_x[j,(i-1)*n_points+1:i*n_points,k],label="")
        end
    end
end
plot!()
# savefig("x_poly.png")
