import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta
import os

# ---------- Input Parameters ----------
file_paths = [
    "data/caiso_2023/caiso_lmp_rt_15min_zones_2023Q1.csv",
    "data/caiso_2023/caiso_lmp_rt_15min_zones_2023Q2.csv",
    "data/caiso_2023/caiso_lmp_rt_15min_zones_2023Q3.csv",
    "data/caiso_2023/caiso_lmp_rt_15min_zones_2023Q4.csv"
]

output_path = "output/caiso_benchmark_2023_hourly_fixedH.xlsx"
combined_8760_path = "data/caiso_2023/caiso_2023_hourly_interp.xlsx"

MW_charge = 30
MW_discharge = 30
t_charge_hrs = 8
t_discharge_hrs = 5

# ---------- Step 1: Load & Concatenate Quarterly CSVs ----------
def load_caiso_q_files(file_paths):
    all_quarters = []
    for path in file_paths:
        df = pd.read_csv(
            path,
            skiprows=3,
            names=[
                "utc_end", "local_start", "local_end", "local_date", "hour",
                "NP-15 LMP", "SP-15 LMP", "ZP-26 LMP",
                "NP-15 Congestion", "SP-15 Congestion", "ZP-26 Congestion",
                "NP-15 Energy", "SP-15 Energy", "ZP-26 Energy",
                "NP-15 Loss", "SP-15 Loss", "ZP-26 Loss"
            ],
            usecols=["local_start", "NP-15 LMP", "SP-15 LMP", "ZP-26 LMP"]
        )
        # Remove the extra header row embedded in the data
        df = df[df["NP-15 LMP"] != "NP-15 LMP"]
        df["local_start"] = pd.to_datetime(df["local_start"])
        for col in ["NP-15 LMP", "SP-15 LMP", "ZP-26 LMP"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.set_index("local_start", inplace=True)
        all_quarters.append(df)
    return pd.concat(all_quarters)

# ---------- Step 2: Resample to Hourly and Interpolate ----------
def interpolate_and_pad_to_8760(df_15min, freq="H", start_date=None, end_date=None):
    # Resample to hourly averages.
    hourly_df = df_15min.resample(freq).mean()
    # If start/end dates not provided, derive them from the data.
    if start_date is None:
        start_date = hourly_df.index.min().normalize()
    if end_date is None:
        end_date = hourly_df.index.max().normalize()
    # Create a full date range for the period.
    full_range = pd.date_range(start_date, end_date + pd.Timedelta(hours=23), freq=freq)
    hourly_df = hourly_df.reindex(full_range)
    interpolated_flags = hourly_df.isna().astype(int)
    hourly_df_interp = hourly_df.interpolate(method="time", limit_direction="both")
    for col in hourly_df.columns:
        hourly_df_interp[f"{col}_is_interpolated"] = interpolated_flags[col]
    return hourly_df_interp

# ---------- Step 3: Fixed-H Benchmark ----------
def run_fixed_H_arbitrage_benchmark(df_hourly, MW_charge, MW_discharge, t_charge_hrs, t_discharge_hrs, start_date=None, end_date=None):
    # If not provided, derive start/end dates from data.
    if start_date is None:
        start_date = df_hourly.index.min().normalize()
    if end_date is None:
        end_date = df_hourly.index.max().normalize()
    # Build the set of days to evaluate.
    all_days = pd.date_range(start_date, end_date, freq="D")
    final_results = {}
    # Use the first three columns (the three zones).
    zones = df_hourly.columns[:3]

    for zone in zones:
        zone_prices = df_hourly[zone]
        best_H = None
        best_total_revenue = -np.inf
        best_H_results = None

        # Evaluate each fixed offset H ∈ [0, 23] consistently across all days.
        for H in tqdm(range(24), desc=f"Evaluating H for {zone}"):
            results = []
            for day in all_days:
                block_start = day + timedelta(hours=H)
                block_end = block_start + timedelta(hours=24)
                block = zone_prices[block_start:block_end]
                if len(block) < 24:
                    continue

                best_revenue = -np.inf
                best_block = None
                # Evaluate charge start times from hour 0 to 10 within the block.
                for t_c in range(0, 11):
                    t_charge_end = t_c + t_charge_hrs
                    # Evaluate discharge intervals starting after charging ends,
                    # ensuring the discharge window is within the 24-hour block.
                    for t_d in range(t_charge_end, 24 - t_discharge_hrs + 1):
                        t_discharge_end = t_d + t_discharge_hrs
                        charge_prices = block.iloc[t_c:t_charge_end].values
                        discharge_prices = block.iloc[t_d:t_discharge_end].values
                        revenue = -MW_charge * np.sum(charge_prices) + MW_discharge * np.sum(discharge_prices)

                        if revenue > best_revenue:
                            best_block = {
                                "date": day.date(),
                                "block_start": block.index[0],
                                "block_offset_H": H,
                                "charge_start": block.index[t_c],
                                "charge_end": block.index[t_charge_end - 1],
                                "discharge_start": block.index[t_d],
                                "discharge_end": block.index[t_discharge_end - 1],
                                "revenue": revenue
                            }
                            best_revenue = revenue

                if best_block:
                    results.append(best_block)

            df_H = pd.DataFrame(results)
            # If no valid blocks were found for this H, skip it.
            if df_H.empty or "revenue" not in df_H.columns:
                continue
            total_revenue = df_H["revenue"].sum()
            if total_revenue > best_total_revenue:
                best_total_revenue = total_revenue
                best_H = H
                best_H_results = df_H

        if best_H_results is not None:
            best_H_results["annual_revenue"] = best_total_revenue
            best_H_results["best_block_offset_H"] = best_H
            final_results[zone] = best_H_results

    return final_results

# ---------- Step 4: Run ----------
if __name__ == "__main__":
    # Load the 15-min data.
    df_15min = load_caiso_q_files(file_paths)
    # Determine the date range dynamically from the data.
    data_start = df_15min.index.min().normalize()
    data_end = df_15min.index.max().normalize()
    
    # Interpolate and pad to get the full hourly series.
    df_hourly = interpolate_and_pad_to_8760(df_15min, freq="H", start_date=data_start, end_date=data_end)
    
    # Save the combined hourly (8760 or 8784) data.
    df_hourly.to_excel(combined_8760_path)
    print(f"✅ Combined hourly data saved to: {combined_8760_path}")
    
    # Run the fixed-H arbitrage benchmark.
    results = run_fixed_H_arbitrage_benchmark(
        df_hourly=df_hourly,
        MW_charge=MW_charge,
        MW_discharge=MW_discharge,
        t_charge_hrs=t_charge_hrs,
        t_discharge_hrs=t_discharge_hrs,
        start_date=data_start,
        end_date=data_end
    )

    # Write benchmark results to Excel.
    with pd.ExcelWriter(output_path) as writer:
        for zone, df in results.items():
            df.to_excel(writer, sheet_name=zone, index=False)

    print(f"\n✅ Benchmark complete. Output saved to:\n{output_path}")
