#Antes de correr el programa se ejecuto este codigo"

""" ```python
#Codigo de precarga de distancias

import pandas as pd
import requests
import time as time_module

# Cargar nodos
clientes = pd.read_csv("clients.csv")
depots = pd.read_csv("depots.csv")

nodes = pd.concat([
    clientes[["StandardizedID","Longitude","Latitude"]].rename(columns={"StandardizedID":"NodeID"}),
    depots[["StandardizedID","Longitude","Latitude"]].rename(columns={"StandardizedID":"NodeID"})
]).reset_index(drop=True)

N = len(nodes)
print(f"Total nodos: {N}")

# --- CHUNK SIZE ---
CHUNK = 25   # puedes subir a 30 si quieres

dist_rows = []
time_rows = []


def osrm_table_chunk(rows_idx, cols_idx):
    Hace una llamada OSRM/table pero solo con subset de nodos.
    
    subset = nodes.loc[rows_idx + cols_idx]
    coords = [f"{row['Longitude']},{row['Latitude']}" for _, row in subset.iterrows()]
    coords_str = ";".join(coords)

    url = f"http://router.project-osrm.org/table/v1/car/{coords_str}?annotations=distance,duration"
    
    r = requests.get(url).json()

    if "distances" not in r:
        print("Error en una llamada OSRM/table:", r)
        return None, None

    return r["distances"], r["durations"]



print("Iniciando cálculo OSRM/table por bloques...\n")

for r0 in range(0, N, CHUNK):
    for c0 in range(0, N, CHUNK):

        rows_idx = list(range(r0, min(r0 + CHUNK, N)))
        cols_idx = list(range(c0, min(c0 + CHUNK, N)))

        print(f"Chunk: filas {r0}-{r0+CHUNK}, columnas {c0}-{c0+CHUNK}")

        dist_block, time_block = osrm_table_chunk(rows_idx, cols_idx)

        # Si falló, intentemos una vez más
        if dist_block is None:
            print("Reintentando chunk en 3 segundos...")
            time_module.sleep(3)
            dist_block, time_block = osrm_table_chunk(rows_idx, cols_idx)

        # Si todavía fallo → abortar
        if dist_block is None:
            raise Exception("OSRM no respondió correctamente, abortando.")

        # Guardamos cada entrada no diagonal
        for ii, i_idx in enumerate(rows_idx):
            id_i = nodes.loc[i_idx, "NodeID"]
            for jj, j_idx in enumerate(cols_idx):
                id_j = nodes.loc[j_idx, "NodeID"]
                if id_i == id_j:
                    continue

                d_km = dist_block[ii][jj] / 1000.0
                t_hr = time_block[ii][jj] / 3600.0

                dist_rows.append([id_i, id_j, d_km])
                time_rows.append([id_i, id_j, t_hr])

        time_module.sleep(0.15)  # pequeña pausa para no saturar el servidor


# Guardar resultados
pd.DataFrame(dist_rows, columns=["i","j","dist_km"]).to_csv("distances.csv", index=False)
pd.DataFrame(time_rows, columns=["i","j","time_hr"]).to_csv("times.csv", index=False)

print("\nListo: matrices guardadas en distances.csv y times.csv")
```"""

import pandas as pd
import numpy as np
import pyomo.environ as pyo
import folium


clientes = pd.read_csv('Caso3/clients.csv')
depots = pd.read_csv('Caso3/depots.csv')
parametros = pd.read_csv('Caso3/parameters_urban.csv')
vehiculos = pd.read_csv('Caso3/vehicles.csv')

distances = pd.read_csv("Caso3/distances.csv")
times = pd.read_csv("Caso3/times.csv")


dist_dict = {(row["i"], row["j"]): row["dist_km"] for _, row in distances.iterrows()}
time_dict = {(row["i"], row["j"]): row["time_hr"] for _, row in times.iterrows()}



Problem3 = pyo.ConcreteModel()
Problem3.C = pyo.Set(initialize=clientes["StandardizedID"].tolist())     # Clientes
Problem3.D = pyo.Set(initialize=depots["StandardizedID"].tolist())       # Depots
Problem3.V = pyo.Set(initialize=vehiculos["StandardizedID"].tolist())    # Vehículos

N_list = clientes["StandardizedID"].tolist() + depots["StandardizedID"].tolist()
Problem3.N = pyo.Set(initialize=N_list)                                  # Todos los nodos


# Demanda por cliente
demand_dict = clientes.set_index("StandardizedID")["Demand"].to_dict()
Problem3.demand = pyo.Param(Problem3.C, initialize=demand_dict)

# Capacidad de cada vehículo
capacity_dict = vehiculos.set_index("StandardizedID")["Capacity"].to_dict()
Problem3.capacity = pyo.Param(Problem3.V, initialize=capacity_dict)

# Rango / autonomía por vehículo
range_dict = vehiculos.set_index("StandardizedID")["Range"].to_dict()
Problem3.range = pyo.Param(Problem3.V, initialize=range_dict)

# Restricción de tipo de vehículo permitido en cada cliente
Problem3.clientRestriction = clientes.set_index("StandardizedID")["VehicleSizeRestriction"].to_dict()

# Tipo de vehículo (solo como dict para usar en las reglas)
vehicleType_dict = vehiculos.set_index("StandardizedID")["VehicleType"].to_dict()
Problem3.vehicleType = vehicleType_dict

# Costos
param_dict = parametros.set_index("Parameter")["Value"].to_dict()
Problem3.C_fixed = float(param_dict["C_fixed"])
Problem3.C_dist  = float(param_dict["C_dist"])
Problem3.C_time  = float(param_dict["C_time"])
Problem3.fuel_price = float(param_dict["fuel_price"])

# Eficiencias de combustible (dict plano)
fuel_eff_dict = {k: v for k, v in param_dict.items() if "fuel_efficiency" in k}
Problem3.fuel_eff = fuel_eff_dict

clientes_ids = set(clientes["StandardizedID"].tolist())
depots_ids   = set(depots["StandardizedID"].tolist())

K = 5  # número de vecinos más cercanos entre clientes

from collections import defaultdict

# 1) Distancias cliente-cliente
dist_cc = defaultdict(list)
for (i, j), d in dist_dict.items():
    if i in clientes_ids and j in clientes_ids and i != j:
        dist_cc[i].append((j, d))

A_reducida = set()
dist_reduc = {}
time_reduc = {}

# 2) Mantener SIEMPRE todos los arcos depot <-> cliente
for (i, j), d in dist_dict.items():
    if (i in depots_ids and j in clientes_ids) or (i in clientes_ids and j in depots_ids):
        A_reducida.add((i, j))
        dist_reduc[(i, j)] = d
        time_reduc[(i, j)] = time_dict[(i, j)]

# 3) Entre clientes: solo los K vecinos más cercanos
for i, lst in dist_cc.items():
    lst_sorted = sorted(lst, key=lambda x: x[1])[:K]
    for j, d in lst_sorted:
        if (i, j) not in A_reducida:
            A_reducida.add((i, j))
            dist_reduc[(i, j)] = d
            time_reduc[(i, j)] = time_dict[(i, j)]

# Crear conjunto de arcos y parámetros
Problem3.A = pyo.Set(initialize=list(A_reducida), dimen=2)
Problem3.dist = pyo.Param(Problem3.A, initialize=dist_reduc)
Problem3.time = pyo.Param(Problem3.A, initialize=time_reduc)

#VARIABLES

# x[v,(i,j)] = 1 si el vehículo v recorre el arco i->j
Problem3.x = pyo.Var(Problem3.V, Problem3.A, domain=pyo.Binary)

# y[v] = 1 si el vehículo v es utilizado
Problem3.y = pyo.Var(Problem3.V, domain=pyo.Binary)

# u[v,n]: carga acumulada en vehículo v al salir del nodo n
max_demand = clientes["Demand"].sum()
Problem3.u = pyo.Var(Problem3.V, Problem3.N, bounds=(0, max_demand))


def dv_rule(model, v):
    return sum(
        model.dist[i_j] * model.x[v, i_j]
        for i_j in model.A
    )

Problem3.d = pyo.Expression(Problem3.V, rule=dv_rule)


def eficiencia_de(v):
    # Busca en fuel_efficiency_* según el tipo de vehículo
    tipo = Problem3.vehicleType[v].replace(" ", "_").lower()
    for k, val in Problem3.fuel_eff.items():
        if tipo in k.lower():
            return float(val)
    # Valor por defecto si no lo encuentra
    return 30.0

#FUNC OBJ

def total_cost_rule(model):

    fixed_cost = sum(
        model.C_fixed * model.y[v]
        for v in model.V
    )

    distance_cost = sum(
        model.C_dist * model.dist[i_j] * model.x[v, i_j]
        for v in model.V
        for i_j in model.A
    )

    time_cost = sum(
        model.C_time * model.time[i_j] * model.x[v, i_j]
        for v in model.V
        for i_j in model.A
    )

    fuel_cost = sum(
        model.fuel_price * (model.d[v] / eficiencia_de(v))
        for v in model.V
    )

    return fixed_cost + distance_cost + time_cost + fuel_cost

Problem3.obj = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

#RESTRICCIONES

# 1. Cada cliente exactamente una vez
def one_visit_rule(model, c):
    return sum(
        model.x[v, (i, c)]
        for v in model.V
        for (i, j) in model.A if j == c
    ) == 1

Problem3.one_visit = pyo.Constraint(Problem3.C, rule=one_visit_rule)

# 2. Balance de flujo para cada vehículo y nodo
def flow_balance_rule(model, v, n):
    return sum(
        model.x[v, (i, n)]
        for (i, j) in model.A if j == n
    ) == sum(
        model.x[v, (n, j)]
        for (i, j) in model.A if i == n
    )

Problem3.flow = pyo.Constraint(Problem3.V, Problem3.N, rule=flow_balance_rule)

# 3. Asignación de depot: todos los vehículos salen del mismo depot (el primero)
default_depot = depots["StandardizedID"].iloc[0]
vehicle_depot = {v: default_depot for v in Problem3.V}

def enforce_start(model, v):
    depot_v = vehicle_depot[v]
    return sum(
        model.x[v, (depot_v, j)]
        for (i, j) in model.A if i == depot_v
    ) == model.y[v]

Problem3.start = pyo.Constraint(Problem3.V, rule=enforce_start)

def enforce_end(model, v):
    depot_v = vehicle_depot[v]
    return sum(
        model.x[v, (i, depot_v)]
        for (i, j) in model.A if j == depot_v
    ) == model.y[v]

Problem3.end = pyo.Constraint(Problem3.V, rule=enforce_end)

# 4. Compatibilidad vehículo–cliente (robusta, usando textos del CSV)
def sanitize(s):
    s = str(s).lower()
    # quitar espacios, tabs, saltos, NBSP
    s = s.replace(" ", "").replace("\t","").replace("\n","").replace("\r","").replace("\xa0","")
    return s

def compatibility_rule(model, v, i, j):
    if j not in model.C:
        return pyo.Constraint.Skip

    veh_type_norm = sanitize(model.vehicleType[v])
    cli_req_norm  = sanitize(model.clientRestriction[j])

    permitido = 1 if veh_type_norm == cli_req_norm else 0

    return model.x[v, (i, j)] <= permitido


Problem3.compat = pyo.Constraint(Problem3.V, Problem3.A, rule=compatibility_rule)

# 5. Carga en el depot = 0
def load_origin_rule(model, v, d):
    return model.u[v, d] == 0

Problem3.load_origin = pyo.Constraint(Problem3.V, Problem3.D, rule=load_origin_rule)

# 6. Capacidad máxima de vehículo
def capacity_limit_rule(model, v, n):
    return model.u[v, n] <= model.capacity[v]

Problem3.cap = pyo.Constraint(Problem3.V, Problem3.N, rule=capacity_limit_rule)

# 7. MTZ anti-subtour entre clientes
def mtz_rule(model, v, i, j):
    if i in model.C and j in model.C and (i, j) in model.A:
        M = 9999
        return model.u[v, j] >= model.u[v, i] + model.demand[i] - (1 - model.x[v, (i, j)]) * M
    return pyo.Constraint.Skip

Problem3.mtz = pyo.Constraint(Problem3.V, Problem3.C, Problem3.C, rule=mtz_rule)

# 8. Rango / autonomía
def range_rule(model, v):
    return model.d[v] <= model.range[v]

Problem3.range_cons = pyo.Constraint(Problem3.V, rule=range_rule)

print("Iniciando solver...")

solver = pyo.SolverFactory("highs")

solver.options["time_limit"] = 600        # límite duro de tiempo (segundos)
solver.options["presolve"] = "on"
solver.options["parallel"] = "on"
solver.options["threads"] = 0               # 0 = todos los cores
solver.options["random_seed"] = 42

solver.options["mip_heuristic_effort"] = "high"
solver.options["mip_heuristic_strategy"] = "on"

solver.options["mip_rel_gap"] = 0.10        # 10% gap relativo

solver.options["allow_unbounded_or_infeasible"] = "on"
solver.options["mip_detect_symmetry"] = "on"
solver.options["mip_improving_solution"] = "on"

resultado = solver.solve(Problem3, tee=True)

print("\n===== RESULTADO DEL SOLVER =====")
print("Estado del solver:", resultado.solver.status)
print("Condición de terminación:", resultado.solver.termination_condition)

try:
    if hasattr(resultado, "solution") and resultado.solution.status != "none":
        print("El solver devolvió solución MIP (aunque no óptima).")
    else:
        print("El solver NO devolvió solución MIP; posiblemente infeasible.")
except:
    pass


print("\n===== RUTAS OBTENIDAS =====")

rutas = []
for v in Problem3.V:
    for (i, j) in Problem3.A:
        val = Problem3.x[v, (i, j)].value
        if val is not None and val > 0.5:
            rutas.append((v, i, j))

if len(rutas) == 0:
    print("No se encontraron rutas (modelo infeasible o sin seleccionar vehículos).")
else:
    for r in rutas:
        print(r)

print("\n===== DISTANCIA TOTAL POR VEHICULO =====")
for v in Problem3.V:
    total = 0
    for (i, j) in Problem3.A:
        val_x = Problem3.x[v, (i, j)].value
        if val_x is not None and val_x > 0.5:
            total += Problem3.dist[i, j]
    print(f"{v}: {total:.2f} km")

rows = []

for v in Problem3.V:
    # Solo consideramos vehículos usados
    if Problem3.y[v].value is None or Problem3.y[v].value < 0.5:
        continue

    depot = vehicle_depot[v]

    # Reconstruir ruta a partir de x[v,(i,j)]
    ruta = [depot]
    current = depot
    visitados = []

    while True:
        # Buscar el siguiente nodo j tal que x[v,(current,j)] = 1
        siguientes = [
            j for (i, j) in Problem3.A
            if i == current and (Problem3.x[v, (i, j)].value is not None) and Problem3.x[v, (i, j)].value > 0.5
        ]

        if not siguientes:
            # No hay más arcos saliendo
            break

        j = siguientes[0]
        ruta.append(j)

        if j in Problem3.C:
            visitados.append(j)

        if j == depot:
            # Cerramos el ciclo
            break

        current = j

    # Si la ruta no cerró bien, la ignoramos
    if len(ruta) < 2:
        continue

    # InitialLoad = suma de demandas de los clientes atendidos
    initial_load = sum(pyo.value(Problem3.demand[c]) for c in visitados)

    # ClientsServed
    clients_served = len(visitados)

    # DemandsSatisfied como lista separada por guiones
    demands_list = [pyo.value(Problem3.demand[c]) for c in visitados]
    demands_str = "-".join(str(int(d)) for d in demands_list)

    # RouteSequence: "CD01 - C005 - C023 - CD01"
    route_seq_str = " - ".join(ruta)

    # Distancia y tiempo totales
    total_dist = 0.0
    total_time_min = 0.0

    for i, j in zip(ruta[:-1], ruta[1:]):
        if (i, j) in Problem3.A:
            total_dist += float(pyo.value(Problem3.dist[i, j]))
            total_time_min += float(pyo.value(Problem3.time[i, j]) * 60.0)  # horas -> minutos

    # FuelCost por vehículo usando la misma lógica de la FO
    dist_v = float(pyo.value(Problem3.d[v]))
    fuel_cost = float(pyo.value(Problem3.fuel_price)) * (dist_v / eficiencia_de(v))

    rows.append({
        "VehicleId": v,
        "DepotId": depot,
        "InitialLoad": initial_load,
        "RouteSequence": route_seq_str,
        "ClientsServed": clients_served,
        "DemandsSatisfied": demands_str,
        "TotalDistance": total_dist,
        "TotalTime": total_time_min,
        "FuelCost": fuel_cost
    })

df_verif = pd.DataFrame(rows)

# Escribir header EXACTO y luego las filas
with open("verificacion_caso3.csv", "w", encoding="utf-8") as f:
    f.write("VehicleId , DepotId , InitialLoad , RouteSequence , ClientsServed , DemandsSatisfied , TotalDistance , TotalTime , FuelCost\n")
    df_verif.to_csv(f, index=False, header=False)




# Construimos tabla de nodos con coordenadas
nodes = pd.concat([
    clientes[["StandardizedID", "Longitude", "Latitude"]].rename(columns={"StandardizedID": "NodeID"}),
    depots[["StandardizedID", "Longitude", "Latitude"]].rename(columns={"StandardizedID": "NodeID"})
]).reset_index(drop=True)

coord_map = {
    row["NodeID"]: (row["Latitude"], row["Longitude"])
    for _, row in nodes.iterrows()
}

# Centro del mapa: promedio de clientes
center_lat = clientes["Latitude"].mean()
center_lon = clientes["Longitude"].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

# Marcar depots
for _, row in depots.iterrows():
    folium.Marker(
        [row["Latitude"], row["Longitude"]],
        popup=row["StandardizedID"],
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)

# Colores para rutas
colors = [
    "blue", "green", "purple", "orange", "darkred", "cadetblue",
    "darkblue", "darkgreen", "lightred", "lightblue", "lightgreen"
]

color_idx = 0

for v in Problem3.V:
    if Problem3.y[v].value is None or Problem3.y[v].value < 0.5:
        continue

    depot = vehicle_depot[v]
    ruta = [depot]
    current = depot

    while True:
        siguientes = [
            j for (i, j) in Problem3.A
            if i == current and (Problem3.x[v, (i, j)].value is not None) and Problem3.x[v, (i, j)].value > 0.5
        ]
        if not siguientes:
            break
        j = siguientes[0]
        ruta.append(j)
        if j == depot:
            break
        current = j

    if len(ruta) < 2:
        continue

    coords = [coord_map[n] for n in ruta if n in coord_map]

    folium.PolyLine(
        coords,
        weight=4,
        opacity=0.8,
        color=colors[color_idx % len(colors)],
        tooltip=f"Vehículo {v}"
    ).add_to(m)

    color_idx += 1

# Guardar mapa
m.save("mapa_caso3.html")
print("Mapa guardado como mapa_caso3.html")
