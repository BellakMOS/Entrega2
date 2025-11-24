import pandas as pd
import numpy as np
import pyomo.environ as pyo

clients_df = pd.read_csv("clients.csv")
depots_df = pd.read_csv("depots.csv")
vehicles_df = pd.read_csv("vehicles.csv")

clients = list(clients_df["StandardizedID"])
depots = ["CD01", "CD02", "CD03"]
vehicles = list(vehicles_df["VehicleType"])

demand_dict = dict(zip(clients_df["StandardizedID"], clients_df["Demand"]))
depot_cap_dict = dict(zip(depots_df["StandardizedID"], depots_df["Capacity"]))
vehicle_cap_dict = dict(
    zip(vehicles_df["VehicleType"], vehicles_df["Capacity"]))

coord = {}
for _, r in clients_df.iterrows():
    coord[r["StandardizedID"]] = (r["Latitude"], r["Longitude"])
for _, r in depots_df.iterrows():
    coord[r["StandardizedID"]] = (r["Latitude"], r["Longitude"])


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1) * \
        np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


nodes = clients + depots
dist = {}
for i in nodes:
    for j in nodes:
        if i == j:
            dist[(i, j)] = 0
        else:
            lat1, lon1 = coord[i]
            lat2, lon2 = coord[j]
            dist[(i, j)] = haversine(lat1, lon1, lat2, lon2)


model = pyo.ConcreteModel()

model.D = pyo.Set(initialize=depots)
model.C = pyo.Set(initialize=clients)
model.N = model.D | model.C
model.V = pyo.Set(initialize=vehicles)

model.demand = pyo.Param(model.C, initialize=demand_dict)
model.cap_vehicle = pyo.Param(model.V, initialize=vehicle_cap_dict)
filtered_cap_depot = {d: depot_cap_dict[d] for d in depots}
model.cap_depot = pyo.Param(model.D, initialize=filtered_cap_depot)

model.dist = pyo.Param(model.N, model.N, initialize=dist)

model.x = pyo.Var(model.V, model.N, model.N, within=pyo.Binary)
model.y = pyo.Var(model.V, model.D, within=pyo.Binary)
model.load_var = pyo.Var(model.V, model.N, within=pyo.NonNegativeReals)


def obj_rule(m):
    return sum(
        m.dist[i, j] * m.x[v, i, j]
        for v in m.V for i in m.N for j in m.N if i != j
    )


model.Obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)


def assign_one_depot(m, v):
    return sum(m.y[v, d] for d in m.D) == 1


model.AssignDepot = pyo.Constraint(model.V, rule=assign_one_depot)


def start_depot(m, v, d):
    return sum(m.x[v, d, j] for j in m.N if j != d) <= m.y[v, d] * len(m.N)


model.StartDepot = pyo.Constraint(model.V, model.D, rule=start_depot)


def end_depot(m, v, d):
    return sum(m.x[v, j, d] for j in m.N if j != d) <= m.y[v, d] * len(m.N)


model.EndDepot = pyo.Constraint(model.V, model.D, rule=end_depot)


def visit_once(m, c):
    return sum(m.x[v, i, c] for v in m.V for i in m.N if i != c) == 1


model.VisitOnce = pyo.Constraint(model.C, rule=visit_once)


def flow_balance(m, v, n):
    return sum(m.x[v, i, n] for i in m.N if i != n) == sum(m.x[v, n, j] for j in m.N if j != n)


model.Flow = pyo.Constraint(model.V, model.N, rule=flow_balance)


def vehicle_capacity(m, v):
    return sum(m.load_var[v, c] for c in m.C) <= m.cap_vehicle[v]


model.VehicleCap = pyo.Constraint(model.V, rule=vehicle_capacity)


def depot_capacity(m, d):
    return sum(
        m.demand[c] * sum(m.x[v, d, c] for v in m.V)
        for c in m.C
    ) <= m.cap_depot[d]


model.DepotCap = pyo.Constraint(model.D, rule=depot_capacity)


def load_balance(m, v, i, j):
    if i in m.C and j in m.C:
        return m.load_var[v, j] >= m.load_var[v, i] + m.demand[j] - m.cap_vehicle[v] * (1 - m.x[v, i, j])
    return pyo.Constraint.Skip


model.LoadBalance = pyo.Constraint(
    model.V, model.N, model.N, rule=load_balance)

print("Solving")
solver = pyo.SolverFactory("highs")
result = solver.solve(model, tee=True)
print(result.solver.status)
print(result.solver.termination_condition)


def get_vehicle_depot(v):
    """Return assigned depot or None."""
    for d in depots:
        if pyo.value(model.y[v, d]) > 0.5:
            return d
    return None


def get_route(v):
    depot = get_vehicle_depot(v)
    if depot is None:
        return [], []

    arcs = [(i, j) for i in nodes for j in nodes
            if i != j and pyo.value(model.x[v, i, j]) > 0.5]

    if len(arcs) == 0:
        return [], []

    next_dict = {}
    for i, j in arcs:
        next_dict[i] = j

    route = [depot]
    visited = set()
    current = depot

    for _ in range(len(nodes)):
        if current not in next_dict:
            break
        nxt = next_dict[current]
        route.append(nxt)
        if nxt in clients:
            visited.add(nxt)
        current = nxt

    if route[-1] != depot:
        route.append(depot)

    return route, list(visited)


rows = []
for v in vehicles:
    route, served = get_route(v)
    if len(served) == 0:
        continue

    depot = get_vehicle_depot(v)
    demands = [demand_dict[c] for c in served]

    total_dist = sum(dist[(route[i], route[i+1])] for i in range(len(route)-1))

    rows.append({
        "VehicleID": v,
        "Depot": depot,
        "Route": "-".join(route),
        "ClientsServed": len(served),
        "Demands": "-".join(str(x) for x in demands),
        "TotalDistance": round(total_dist, 3),
        "TotalTime": round(total_dist / 25, 3)
    })

df = pd.DataFrame(rows)
df.to_csv("verificacion_caso2.csv", index=False)

print(">>> verificacion_caso2.csv generado exitosamente.")
