using LinearAlgebra

function eigenval_recover!(A, D, m)
    for i = 1:length(D)
        if D[i] < 0.
	        D[i] = 0.
		end
        for j = 1:m
            A[(i-1) * m + j] *= D[i]
        end
    end
    return
end

function projection(X_d, m::Int64)
    # 这里假设X_d是一个三维数组，需要对每个矩阵进行操作
    for i = 1:size(X_d, 3)
        D, A = eigen((X_d[:, :, i] + X_d[:, :, i]') / 2) # 特征值分解
        # 修正特征值，保留特征向量
        # @show D 
        # @show A
        D = max.(D, 0.0)
        # 重构矩阵
        X_d[:, :, i] = A * Diagonal(D) * A'
    end
    return X_d
end


function interpret_solution(Xf,Xg)
    our_y=zeros(2(m+n)*(d+1)*N)
    xt=zeros(n,d+1,N)
    ut=zeros(m,d+1,N)
    xt_col=zeros(n*(d+1)*N)
    ut_col=zeros(m*(d+1)*N)

    Px=kron(Matrix(1.0I,n*N,n*N),Qx)   
    Pu=kron(Matrix(0.1I,m*N,m*N),Qx)  

    Xf = Array{Float32,4}(reshape(Xf, d2, d2, 2(m+n), N))
    Xg = Array{Float32,4}(reshape(Xg, d2, d2, 2(m+n), N))

    for l in 1:N
        for i in 1:d+1 
            seg=(l-1)*2(m+n)*(d+1)
            for j in 1:2(m+n)
                our_y[seg+(j-1)*(d+1)+i]=tr(F[:,:,i]*Xf[:,:,j,l])+tr(G[:,:,i]*Xg[:,:,j,l])
            end
            for j in 1:n
                xt[j,i,l]=((our_y-g)[seg+(j-1)*(d+1)+i]-(our_y-g)[seg+(j+n-1)*(d+1)+i])/2 #/2 to get average
                xt_col[(l-1)*n*(d+1)+(j-1)*(d+1)+i]=xt[j,i,l]
            end
            for j in 1:m
                ut[j,i,l]=((our_y-g)[seg+(j+2n-1)*(d+1)+i]-(our_y-g)[seg+(j+2n+m-1)*(d+1)+i])/2 #/2 to get average
                ut_col[(l-1)*m*(d+1)+(j-1)*(d+1)+i]=ut[j,i,l]
            end        
        end
    end
    obj=(xt_col-xref_col)'*Px*(xt_col-xref_col)+ut_col'*Pu*ut_col
    # innerp=u'*(our_y)

    return xt,ut,obj#,innerp

end

function PDHG_solver(N,d,m,n,α,β,FF,μμ,F,G, max_iter=20000, if_warm_start=false, last_Xf=nothing, last_Xg=nothing, last_u=nothing)
    d2 = Int((d-1)/2+1)

    if if_warm_start
        Xf = last_Xf
        Xg = last_Xg
        u = last_u
    else
        Xf = rand(d2,d2,2*(m+n)*N) .- 0.5
        Xg = rand(d2,d2,2*(m+n)*N) .- 0.5
        u = zeros(d+1,2*(n+m)*N) .- 0.5
    end

    I_FF = Matrix{Float32}(I, 2*(m+n)*(d+1)*N, 2*(m+n)*(d+1)*N)
    IminusβFF = I_FF - FF*β
    FG = cat(F, G, dims=3)

    X = cat(Xf, Xg, dims=3)
    dX = zeros(size(X))

    u_large = zeros(2*(d+1), 4*(m+n) * N)
    u_large[1:d+1, 1:2*(m+n) * N] = u
    u_large[d+2:end, 2*(m+n) * N + 1:end] = u
    u_large_tmp = deepcopy(u_large)

    for k = 1:max_iter
        oldX = deepcopy(X)
        u_large[1:d+1, 1:2*(m+n) * N] = u
        u_large[d+2:end, 2*(m+n) * N + 1:end] = u
        
        for a = 1:size(X, 1), b = 1:size(X, 2), j = 1:size(X, 3), i = 1:size(FG, 3)
            X[a, b, j] -= α * u_large[i, j] * FG[a, b, i]
        end
        
        X = projection(X, d2)  
        
        for a = 1:size(dX, 1), b = 1:size(dX, 2), j = 1:size(dX, 3), i = 1:size(FG, 3)
            dX = 2 * X - oldX
            u_large_tmp[i, j] = β * FG[a, b, i] * dX[b, a, j]
        end
        
        u += u_large_tmp[1:d+1, 1:2*(m+n) * N] + u_large_tmp[d+2:end, 2*(m+n) * N + 1:end]
        tmp = deepcopy(μμ)
        u = reshape(u, 2*(m+n) * (d+1) * N)
        tmp = IminusβFF * u .- β * tmp
        u = reshape(tmp, d+1, 2*(m+n)*N)
    end

    Xf = X[:,:,1:2*(m+n)*N]
    Xg = X[:,:,(2*(m+n)*N+1):end]
    xt, ut, obj = interpret_solution(Xf, Xg)  
    
    return xt, ut, Xf, Xg, u, obj
end




