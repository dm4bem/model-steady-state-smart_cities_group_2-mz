import numpy as np

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
To = 15         # °C

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
A[13,5], A[13,11] = -1, 1    # branch 9: node 8 -> node 9
A[14,11] = 1

# Conductance matrix
# ==================
G = np.zeros(A.shape[0])

#Radiation longue distance 

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
G[5] = hi*So[0]
G[6] = hi*So[3]+GLW  # convection + rayonnement du mur interieur
G[7] = wc*λc/So[3]
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

"""# Zone 2 & 4 free-running; solar rad; without ventilation
G[[17, 19]] = 0     # controller gains for room 2 & 4

# Solar radiation
exterior_wall = [0, 1, 5, 7]
f[exterior_wall] = E * So

θ = np.linalg.inv(A.T @ np.diag(G) @ A) @ (A.T @ np.diag(G) @ b + f)
q = np.diag(G) @ (-A @ θ + b)
print("2. 2 & 4 free-run w/o ventilation")
print("θ:", θ[indoor_air], "°C")
print("q:", q[controller], "W")

# Zone 2 & 4 free-running; solar rad;
# Ventilation outdoor -> room 2 -> room 4 -> outdoor
ventilation = range(13, 16)
G[ventilation] = m_dot * c, m_dot * c, 0

θ = np.linalg.inv(A.T @ np.diag(G) @ A) @ (A.T @ np.diag(G) @ b + f)
q = np.diag(G) @ (-A @ θ + b)
print("3. 2 & 4 free-run, ventilation out -> 2 -> 4 -> out")
print("θ:", θ[indoor_air], "°C")
print("q:", q[controller], "W")

# Zone 2 & 4 free-running; solar rad;
# Ventilation outdoor -> room 4 -> room 2 -> outdoor
G[ventilation] = 0, m_dot * c, m_dot * c

θ = np.linalg.inv(A.T @ np.diag(G) @ A) @ (A.T @ np.diag(G) @ b + f)
q = np.diag(G) @ (-A @ θ + b)
print("4. 2 & 4 free-run, ventilation out -> 4 -> 2 -> out")
print("θ:", θ[indoor_air], "°C")
print("q:", q[controller], "W")"""