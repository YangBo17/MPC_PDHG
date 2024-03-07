using ModelingToolkit
using Parameters

function get_system_matrix()
    # Motor
    Rm = 8.4 # resistence
    kt = 0.042 # current-torque (N-m/A)
    km = 0.042 # Back-emf constant (V-s/rad)
    mr = 0.095 # Mass (kg)
    Lr = 0.085 # Total length (m)
    Jr = mr*Lr^2/3 # Moment of invertia about pivot (kg-m^2)
    br = 1e-3 # damping tuned heuristically to match QUBE-Sero 2 response
    # Pendulum Link
    mp = 0.1 # Mass (kg) 24+71+5.2
    Lp = 0.7 # Total Length (m) 0.129
    l = Lp/2 # Pendulum center of mass (m)
    Jp = mp*Lp^2/3*0.35 # Moment of inertia about pivot
    bp = 5e-5 # damping tuned heuristically to match QUBE-Sero 2 response
    g = 9.81 # gravity

    Jt =Jp*mp*Lr^2+Jr*Jp+1/4*Jr*mp*Lp^2

    Ac = [0 0 1 0;
        0 0 0 1;
        0  1/4*mp^2*Lp^2*Lr*g/Jt    -(Jp+1/4*mp*Lp^2)*br/Jt   -1/2*mp*Lp*Lr*bp/Jt;
        0  1/2*mp*Lp*g*(mp*Lr^2+Jr)/Jt      -1/2*mp*Lp*Lr*br/Jt   -(mp*Lr^2+Jr)*bp/Jt];
    Bc = [0; 
         0; 
         (Jp+1/4*mp*Lp^2)/Jt;
         1/2*mp*Lp*Lr/Jt];
    Cc = [1 0 0 0;
          0 0 1 0]
    Dc = [0;
          0]
    Ac[3,3] = Ac[3,3] - kt*km/Rm*Bc[3]
    Ac[4,3] = Ac[4,3] - kt*km/Rm*Bc[4]
    Bc = kt * Bc / Rm

    P = [Bc  Ac*Bc Ac^2*Bc Ac^3*Bc]
    S = P^(-1)
    S = [S[end,:]'; S[end,:]'*Ac; S[end,:]'*Ac^2; S[end,:]'*Ac^3]

    G = S * Ac * (S)^(-1)
    H = S * Bc
    for i in eachindex(G)
        if abs(G[i])<10^(-10)
            G[i]=0
        end
    end
    for i in eachindex(H)
        if abs(H[i])<10^(-10)
            H[i]=0
        end
        if abs(H[i]-1)<10^(-10)
            H[i]=1
        end
    end
    return Ac,Bc,G,S,H
end
