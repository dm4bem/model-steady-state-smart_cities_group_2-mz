import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
np.set_printoptions(precision=1)

# Data
# ====
# dimensions
L, l, H, wc, wg = 3, 3, 3, 0.20, 0.04  # m

# thermo-physical propertites
λc = 1.4             # W/(m K) wall thermal conductivity
λg = 1.4
λi=0.027
wi=0.08
ρ, c = 1.2, 1000    # kg/m3, J/(kg K) density, specific heat air
hi, ho = 8, 25      # W/(m2 K) convection coefficients in, out
λw, ww = 1, 0.02
Sd = 1.3*2
So = np.array([2*L*H, L*H, Sd, (L*H-Sd) ] )     # outdoor surfaces
# short-wave solar radiation absorbed by each wall
E = 200             # W/m2

# outdoor temperature
To = 15      # °C

# ventilation rate (air-changes per hour)
ACH = 1             # volume/h

V_dot_fen = L * l * H * ACH / 3600  # volumetric air flow rate
m_dot_fen = ρ * V_dot_fen * c               # mass air flow rate
V_dot_door = L * So[2] * ACH / 3600  # volumetric air flow rate
m_dot_door = ρ * V_dot_door *c              # mass air flow rate

nq, nθ = 15, 12  # number of flow-rates branches and of temperaure nodes

# Incidence matrix
# ================
A = np.zeros([nq, nθ])

A[0, 0] = 1                 
A[1, 0], A[1, 1] = -1, 1    
A[2, 1], A[2, 2] = -1, 1   
A[3, 2], A[3, 3] = -1, 1 
A[4, 3], A[4, 4] = -1, 1   
A[5, 4], A[5, 5] = -1, 1    
A[6, 5], A[6, 6] = -1, 1              
A[7, 6], A[7, 7] = -1, 1 
A[8, 7], A[8, 8] = -1, 1
A[9, 8], A[9, 9] = -1, 1
A[10, 9], A[10, 10] = -1, 1
A[11, 10], A[11, 11] = -1, 1
A[12,5] = 1 
A[13,6], A[13,10] = -1, 1   
A[14,11] = 1

# Conductance matrix
# ==================
G = np.zeros(A.shape[0])

# long wave radiation

σ=5.67e-8
ε_wLW = 0.85    # long wave emmisivity: wall surface (concrete)
ε_gLW = 0.90    # long wave emmisivity: glass pyrex
Fwg = 3

Tm = 20 + 273   # K, mean temp for radiative exchange

GLW = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * So[3]



# G0 ... G3 (cyan branches): outdoor convection
"""L4 = 2 * l + 3 * L + 2 * w """     # length outdoor wall room 4

G[0] = ho * So[0]
G[1] = wc*λc/So[0]
G[2] = wc*λc/So[0]
G[3] = wi*λi/So[0]
G[4] = wi*λi/So[0]
G[5] = hi*So[0]  # convection mur
G[6] = hi*So[3] + GLW # convection mur interieur 
G[7] = wc*λc/So[3] # conduction mur interieux 
G[8] = wc*λc/So[3]
G[9] = wi*λi/So[3]
G[10] = wi*λi/So[3]
G[11] = hi * So[3]

G[12] = m_dot_fen # flux fenetre
G[13] = m_dot_door # flux porte
G[14] = 0
# Vector of temperature sources
# =============================
b = np.zeros(A.shape[0])

b[0] = To        # cyan branches: outdoor temperature for walls
b[12] = To 

# Vector of flow-rate sources
# =============================
f = np.zeros(A.shape[1])

# Solar radiation
exterior_wall = 0
f[exterior_wall] = E * So[0]


# Indexes of outputs
# ==================
indoor_air = [5,11]   # indoor air temperature nodes
controller = 14  # controller branches

print(f"Maximum value of conductance: {max(G):.0f} W/K")

b[controller] = 20  # °C setpoint temperature of the rooms
G[controller] = 1e5            # P-controller gain

print("G : ",G)
"""θ = np.linalg.inv(A.T @ np.diag(G) @ A)"""

θ = np.linalg.inv(A.T @ np.diag(G) @ A) @ (A.T @ np.diag(G) @ b + f)
q = np.diag(G) @ (-A @ θ + b)
print("Clim à 20°C")
print("θ:", θ[indoor_air], "°C")
print("q:", q[controller], "W")

## Step response

#Capacities 

Dw = 2300
Cw = 880
Dg = 2500
Cg = 1210
Da = 1.2
Ca = 1000
Va = 9
Di = 55
Ci = 1210
Cwall = Dw*Cw*ww*So[0]
Ciso = Di*Ci*wi*So[0]
Cglass = Dg*Cg*wg*So[1]
Cwall_mid = Dw*Cw*ww*So[3]
Ciso_mid = Di*Ci*wi*So[3]
Ca = Da*Ca*Va

C = np.array([0, Cwall, 0, Ciso, 0, Ca,0,Cwall_mid,0,Ciso_mid,0,Ca])

dt = 300
duration = 60*60*24*2
time_steps = int(duration / dt)
time = np.linspace(0, duration, time_steps)
# Initialize temperature DataFrames
θ_exp = pd.DataFrame(index=time, columns=range(nθ)).fillna(0.0)
θ_imp = pd.DataFrame(index=time, columns=range(nθ)).fillna(0.0)

# Identity matrix
I = np.eye(nθ)

# Simulation loop
for k in range(time_steps - 1):
    θ_exp.iloc[k + 1] = (I + dt * (A.T @ np.diag(G) @ A)) @ θ_exp.iloc[k] + dt * (A.T @ np.diag(G) @ b + f)
    θ_imp.iloc[k + 1] = np.linalg.inv(I - dt * (A.T @ np.diag(G) @ A)) @ (θ_imp.iloc[k] + dt * (A.T @ np.diag(G) @ b + f))

# Extract the indoor air temperature responses
y_exp = θ_exp[indoor_air]
y_imp = θ_imp[indoor_air]

# Plotting the results
plt.figure(figsize=(10, 6))
for col in y_exp.columns:
    plt.plot(time / 3600, y_exp[col], label=f'Explicit Euler Node {col}')
    plt.plot(time / 3600, y_imp[col], linestyle='--', label=f'Implicit Euler Node {col}')

plt.xlabel('Time (hours)')
plt.ylabel('Temperature (°C)')
plt.title('Step Response of Indoor Air Temperature')
plt.legend()
plt.grid()
plt.show()

