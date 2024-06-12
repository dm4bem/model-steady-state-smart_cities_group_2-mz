#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:07:44 2024

@author: gustavewendling
"""

import numpy as np
import pandas as pd
from dm4bem import read_epw

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
To = 0      # °C

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


G[0] = ho * So[0]
G[1] = wc*λc/So[0]
G[2] = wc*λc/So[0]
G[3] = wi*λi/So[0]
G[4] = wi*λi/So[0]
G[5] = hi*So[0]  # convection mur
G[6] = hi*So[3]  +GLW # convection mur interieur 
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


# Inputs
# ======
filename ='FRA_Lyon.074810_IWEC.epw'

θ = 4.3          # °C, indoor temperature all time
θday =   θ      # °C, indoor temperature during day,, e.g.: 06:00 - 22:00
θnight = 4.2    # °C, indoor temperature during night 23:00 - 06:00

period_start = '2000-01-01'
period_end = '2000-12-31'

daytime_start = '08:00:00+01:00'
daytime_end = '18:00:00+01:00'


# Computation
# ===========
# read Energy Plus Weather data (file .EPW)
[data, meta] = read_epw(filename, coerce_year=2000)

# select outdoor air temperature; call it θout
df = data[["temp_air"]]
del data
df = df.rename(columns={'temp_air': 'θout'})

# Select the data for a period of the year
df = df.loc[period_start:period_end]

# Compute degree-hours for fixed set-point
# ----------------------------------------
df['Δθfix'] = θ - df['θout'].where(
    df['θout'] < θ,
    θ)


# Define start time for day and night
day_start = pd.to_datetime(daytime_start).time()
day_end = pd.to_datetime(daytime_end).time()

# Daytime should be between 00:00 and 24:00
# Daytime including midnight is not allowed, e.g., 22:00 till 06:00
day = (df.index.time >= day_start) & (df.index.time <= day_end)
night = ~day


# Degree-hours for daytime
df['Δθday'] = θday - df['θout'].where(
    (df['θout'] < θday) & day,
    θday)

# Degree-hours for nighttime
df['Δθnight'] = θnight - df['θout'].where(
    (df['θout'] < θnight) & night,
    θnight)

# Sum of degree-hours for fixed indoor temperature
DHH_fix = df['Δθfix'].sum()

# Sum of degree-hours for intermittent heating
DHH_interm = df['Δθday'].sum() + df['Δθnight'].sum()

# Results
# =======
print(f"degree-hours fixed set-point: {DHH_fix:.1f} h·K")
print(f"degree-hours variable set-point: {DHH_interm:.1f} h·K")
print(f"Estimated savings: {(DHH_fix - DHH_interm) / DHH_fix * 100:.0f} %")