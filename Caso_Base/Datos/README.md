# Base Case Project - Standard CVRP

## Overview

This is the **foundational case** that MUST be completed before attempting any of the project-specific cases (A, B, or C). It implements a standard Capacitated Vehicle Routing Problem (CVRP) with:
- Single depot
- Homogeneous vehicle fleet
- Basic capacity and range constraints
- No special features (time windows, multiple depots, refueling, etc.)

**Purpose**: Validate that your core CVRP implementation works correctly before adding project-specific complexity.

## Data Files

### 1. `clients.csv`
Contains information about delivery locations.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| ClientID | Integer | - | Numeric client identifier (1, 2, 3, ...) |
| StandardizedID | String | - | Standardized ID for verification files (C001, C002, C003, ...) |
| LocationID | Integer | - | Unified location identifier across all entities |
| Latitude | Float | degrees | Geographic latitude |
| Longitude | Float | degrees | Geographic longitude |
| Demand | Float | kg | Quantity of goods required by client |

### 2. `vehicles.csv`
Details about available vehicles for deliveries.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| VehicleID | Integer | - | Numeric vehicle identifier (1, 2, 3, ...) |
| StandardizedID | String | - | Standardized ID for verification files (V001, V002, V003, ...) |
| Capacity | Float | kg | Maximum load the vehicle can carry |
| Range | Float | km | Maximum distance vehicle can travel (fuel/battery limit) |

### 3. `depots.csv`
Information about the distribution center.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| DepotID | Integer | - | Numeric depot identifier (1) |
| StandardizedID | String | - | Standardized ID for verification files (CD01) |
| LocationID | Integer | - | Unified location identifier |
| Latitude | Float | degrees | Geographic latitude |
| Longitude | Float | degrees | Geographic longitude |

## ID Standardization

**IMPORTANT**: Your verification files MUST use StandardizedID format:
- **Clients**: Use `StandardizedID` column (C001, C002, ...) in verification files
- **Vehicles**: Use `StandardizedID` column (V001, V002, ...) in verification files
- **Depot**: Use `StandardizedID` column (CD01) in verification files

The numeric IDs (ClientID, VehicleID, DepotID) are provided for backward compatibility and internal use.

## Cost Parameters

See `parameters_base.csv` for reference values. For this base case, you should implement the basic cost structure:

```
Total Cost = Σ(distance traveled) + Σ(time spent) + Σ(fuel consumed)
```

Students should research and incorporate reasonable values for:
- Fuel price (reference: 16,300 COP/gallon)
- Fuel efficiency (reference: ~30 km/gallon)
- Maintenance costs per kilometer
- Driver wages per hour

## Constraints

1. **Capacity**: Each vehicle cannot exceed its capacity limit
2. **Range**: Each vehicle cannot travel more than its range on a single trip
3. **Coverage**: All clients must be visited exactly once
4. **Route Structure**: Each route must start and end at the depot

## Verification File Format

Create `verificacion_caso1.csv` with this structure:

```csv
VehicleId,DepotId,InitialLoad,RouteSequence,ClientsServed,DemandsSatisfied,TotalDistance,TotalTime,FuelCost
V001,CD01,750,CD01-C005-C023-C017-CD01,3,215-320-215,28.4,67.2,98500
```

### Column Explanations:
- **VehicleId**: Standardized vehicle ID (V001, V002, ...)
- **DepotId**: Standardized depot ID (CD01)
- **InitialLoad**: Total load when leaving depot (kg)
- **RouteSequence**: Complete route with depot at start/end, clients in visit order
- **ClientsServed**: Number of clients visited
- **DemandsSatisfied**: Demand fulfilled at each client (in visit order), separated by hyphens
- **TotalDistance**: Total km traveled
- **TotalTime**: Total time in minutes
- **FuelCost**: Total fuel cost in COP

## Validation

Use the provided validation script:

```bash
cd cvrp_content/utils/base_case
python base_case_verification.py --method haversine --verbose
```

The validator will check:
- ✓ Routes start and end at depot
- ✓ Vehicle capacities not exceeded
- ✓ Vehicle ranges not exceeded
- ✓ All clients visited exactly once
- ✓ Demands correctly satisfied

## Important Notes

- **Missing verification file = significant grade penalty**
- Use periods for decimal separators (28.4 not 28,4)
- Use hyphens without spaces for lists (215-320-215 not 215 - 320 - 215)
- Routes must form closed loops (depot → clients → depot)
- This case has NO time windows, NO multiple depots, NO special vehicle types

## Next Steps

Once this base case is validated, proceed to your assigned project:
- **Project A**: Multiple depots with inventory constraints
- **Project B**: Hybrid fleet (drones + trucks) with time windows
- **Project C**: National logistics with refueling stations and tolls
