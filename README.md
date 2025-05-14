# Willow Springs A-CAES Arbitrage Modeling

This project models energy arbitrage strategies for an adiabatic compressed air energy storage (A-CAES) facility located in Willow Springs, California. The model uses locational marginal pricing (LMP) data from the CAISO node TWILGHTL_7_N001 to optimize charge/discharge cycles for revenue maximization.

## Repo Structure

```text
├── arbitrage.py            # Core arbitrage evaluation functions
├── arbitrage_types.py      # Type definitions for arbitrage modeling
├── data/
│   ├── node/               # Monthly LMP data from TWILGHTL_7_N001 (2024)
│   └── region_benchmark/   # LMP data across ERCOT/CAISO/MISO
├── data.py                 # Data processing utilities
├── dijkstra_benchmark.py   # Dijkstra-based arbitrage optimization algorithm
├── naive_benchmark.py      # Daily-window benchmark w/ foresight
├── output/                 # Output files from regional benchmarks
├── region_selection.py     # Analysis for initial region selection
├── generate_trade_data.py  # Generate trade data across Dijkstra/Naive
├── plot_trade_data.py      # Visualization tools for arbitrage analysis
└── trade_data/             # Generated trade data and visualizations
```

## Facility Specifications

- **Location:** Willow Springs, California (CAISO node: TWILGHTL_7_N001)
- **Charging:** 30 MW for 8 hours (240 MWh input)
- **Discharging:** 30 MW for 4.8 hours (142.8 MWh output)
- **Storage Capacity:** 168 MWh
- **Charge Efficiency:** 70%
- **Discharge Efficiency:** 85%
- **Round-trip Efficiency:** 59.5%
- **Pressure Range:** 8-12 MPa
- **TES Volume:** 4,000 m³
- **Compression/Expansion:** 3 stages each

## Arbitrage Algorithms

### Naive Benchmark Algorithm

Implements a simple fixed-H approach where each day is divided into 24-hour blocks starting at hour H, with one full charge/discharge cycle per day. The algorithm evaluates all possible starting hours and selects the most profitable configuration over the course of an entire year.

### Dijkstra Benchmark Algorithm

Implements a sliding-window Dijkstra-based arbitrage strategy with discretized state-of-charge levels. This approach is inspired by the GridWorks SCADA implementation (https://github.com/thegridelectric/gridworks-scada/blob/1b907584f66830d25f4210e89d4b0c0f2fe935d0/gw_spaceheat/actors/flo.py).

## How to Run

### Data Processing

```bash
# Process raw LMP data
python data.py
```

### Running Arbitrage Algorithms

```bash
# Run the naive benchmark algorithm
python naive_benchmark.py

# Run the Dijkstra-based algorithm
python dijkstra_benchmark.py
```

### Generating Trade Analysis

```bash
# Generate trade data for analysis
python generate_trade_data.py

# Create visualizations
python plot_trade_data.py
```

## Configuring Parameters

Key parameters can be modified in the respective algorithm files:

### System Parameters (both algorithms)

- `max_charge_mw`: Maximum charging power in MW (default: 30.0)
- `max_discharge_mw`: Maximum discharging power in MW (default: 30.0)
- `storage_capacity_mwh`: Storage capacity in MWh (default: 168.0)
- `charge_efficiency`: Charging efficiency (default: 0.70)
- `discharge_efficiency`: Discharging efficiency (default: 0.85)

### Naive Benchmark Parameters

- `t_charge_hrs`: Number of hours to charge (default: 8.0)
- `t_discharge_hrs`: Number of hours to discharge (default: 4.8)
- `num_workers`: Number of parallel processes for faster execution (default: 8)

### Dijkstra Algorithm Parameters

- `window_hours`: Optimization window size in hours (default: 48)
- `soc_bins`: Number of state-of-charge discretization bins (default: 50)
- `time_step_hours`: Time step for optimization (default: 1.0)
- `allow_partial_actions`: Whether to allow charges/discharges for less than 1 hour (default: True)
