import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from datetime import timedelta
import os

# ---------- Input Data ----------
# iso_name = "caiso"

# file_paths = [
#     "data/caiso_2023/caiso_lmp_rt_15min_zones_2023Q1.csv",
#     "data/caiso_2023/caiso_lmp_rt_15min_zones_2023Q2.csv",
#     "data/caiso_2023/caiso_lmp_rt_15min_zones_2023Q3.csv",
#     "data/caiso_2023/caiso_lmp_rt_15min_zones_2023Q4.csv"
# ]

# output_path = "output/caiso_benchmark_2023_hourly_fixedH.xlsx"
# combined_8760_path = "data/caiso_2023/caiso_2023_hourly_interp.xlsx"

# iso_name = "ercot"

# file_paths = [
#     "data/ercot_2024/ercot_lmp_rt_15min_hubs_2024Q1.csv",
#     "data/ercot_2024/ercot_lmp_rt_15min_hubs_2024Q2.csv",
#     "data/ercot_2024/ercot_lmp_rt_15min_hubs_2024Q3.csv",
#     "data/ercot_2024/ercot_lmp_rt_15min_hubs_2024Q4.csv"
# ]

# output_path = "output/ercot_benchmark_2024_hourly_fixedH.xlsx"
# combined_8760_path = "data/ercot_2024/ercot_2024_hourly_interp.xlsx"

iso_name = "miso"

file_paths = [
    "data/miso_2024/miso_lmp_rt_5min_hubs_2024Q1.csv",
    "data/miso_2024/miso_lmp_rt_5min_hubs_2024Q2.csv",
    "data/miso_2024/miso_lmp_rt_5min_hubs_2024Q3.csv",
    "data/miso_2024/miso_lmp_rt_5min_hubs_2024Q4.csv"
]

output_path = "output/miso_benchmark_2024_hourly_fixedH.xlsx"
combined_8760_path = "data/miso_2024/miso_2024_hourly_interp.xlsx"

# ---------- Input System Params ----------
MW_charge = 30
MW_discharge = 30
t_charge_hrs = 8
t_discharge_hrs = 5

# ---------- Step 1: Load & Concatenate Quarterly CSVs ----------
def load_generic_lmp_files(file_paths, iso_name):
    all_quarters = []

    for path in file_paths:
        if iso_name.lower() == "caiso":
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
            df = df[df["NP-15 LMP"] != "NP-15 LMP"]

        elif iso_name.lower() == "ercot":
            df = pd.read_csv(
                path,
                skiprows=3,
                names=[
                    "utc_end", "local_start", "local_end", "local_date", "hour",
                    "Bus average LMP", "Houston LMP", "Hub average LMP",
                    "North LMP", "Panhandle LMP", "South LMP", "West LMP"
                ],
                usecols=["local_start", "Houston LMP", "North LMP", "Panhandle LMP", "South LMP", "West LMP"]
            )
            df = df[df["Houston LMP"] != "Houston LMP"]
		
        elif iso_name.lower() == "miso":
            df = pd.read_csv(
                path,
                skiprows=3,
                names=[
                    "utc_end", "local_start", "local_end", "local_date", "hour",
                    "Arkansas Hub LMP", "Illinois Hub LMP", "Indiana Hub LMP",
                    "Louisiana Hub LMP", "Michigan Hub LMP", "Minnesota Hub LMP",
                    "Mississippi Hub LMP", "Texas Hub LMP",
                    "Arkansas Hub Congestion", "Illinois Hub Congestion", "Indiana Hub Congestion",
                    "Louisiana Hub Congestion", "Michigan Hub Congestion", "Minnesota Hub Congestion",
                    "Mississippi Hub Congestion", "Texas Hub Congestion",
                    "Arkansas Hub Loss", "Illinois Hub Loss", "Indiana Hub Loss",
                    "Louisiana Hub Loss", "Michigan Hub Loss", "Minnesota Hub Loss",
                    "Mississippi Hub Loss", "Texas Hub Loss"
                ],
                usecols=[
                    "local_start", "Arkansas Hub LMP", "Illinois Hub LMP", "Indiana Hub LMP",
                    "Louisiana Hub LMP", "Michigan Hub LMP", "Minnesota Hub LMP",
                    "Mississippi Hub LMP", "Texas Hub LMP"
                ]
            )
            df = df[df["Arkansas Hub LMP"] != "Arkansas Hub LMP"]
            df.rename(columns={
                "Arkansas Hub LMP": "Arkansas",
                "Illinois Hub LMP": "Illinois",
                "Indiana Hub LMP": "Indiana",
                "Louisiana Hub LMP": "Louisiana",
                "Michigan Hub LMP": "Michigan",
                "Minnesota Hub LMP": "Minnesota",
                "Mississippi Hub LMP": "Mississippi",
                "Texas Hub LMP": "Texas"
            }, inplace=True)
    
        else:
            raise ValueError(f"Unsupported iso_name: {iso_name}")

        # Parse timestamps (no warning because format is consistent)
        df["local_start"] = pd.to_datetime(df["local_start"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        df = df.dropna(subset=["local_start"])

        # Convert all price columns to numeric
        for col in df.columns:
            if col != "local_start":
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
    if start_date is None:
        start_date = df_hourly.index.min().normalize()
    if end_date is None:
        end_date = df_hourly.index.max().normalize()
    all_days = pd.date_range(start_date, end_date, freq="D")

    final_results = {}
    zones = [col for col in df_hourly.columns if not col.endswith("_is_interpolated")]

    for zone in zones:
        zone_prices = df_hourly[zone]
        best_H = None
        best_total_revenue = -np.inf
        best_H_results = None

        for H in tqdm(range(24), desc=f"Evaluating benchmark for {zone}"):
            results = []
            for day in all_days:
                block_start = day + timedelta(hours=H)
                block_end = block_start + timedelta(hours=24)
                block = zone_prices[block_start:block_end]
                if len(block) < 24:
                    continue

                best_revenue = -np.inf
                best_block = None
                for t_c in range(0, 11):
                    t_charge_end = t_c + t_charge_hrs
                    for t_d in range(t_charge_end, 24 - t_discharge_hrs + 1):
                        t_discharge_end = t_d + t_discharge_hrs
                        charge_prices = block.iloc[t_c:t_charge_end].values
                        discharge_prices = block.iloc[t_d:t_discharge_end].values

                        cost = MW_charge * np.sum(charge_prices)
                        revenue = MW_discharge * np.sum(discharge_prices)
                        net = revenue - cost

                        if net > best_revenue:
                            best_block = {
                                "date": day.date(),
                                "block_start": block.index[0],
                                "charge_start": block.index[t_c],
                                "charge_end": block.index[t_charge_end - 1],
                                "discharge_start": block.index[t_d],
                                "discharge_end": block.index[t_discharge_end - 1],
                                "avg_charge_price": np.mean(charge_prices),
                                "charge_cost": cost,
                                "avg_discharge_price": np.mean(discharge_prices),
                                "discharge_revenue": revenue,
                                "revenue": net
                            }
                            best_revenue = net

                if best_block:
                    results.append(best_block)

            df_H = pd.DataFrame(results)
            if df_H.empty:
                continue
            total_revenue = df_H["revenue"].sum()
            if total_revenue > best_total_revenue:
                best_total_revenue = total_revenue
                best_H = H
                best_H_results = df_H

        if best_H_results is not None:
            summary_df = pd.DataFrame({
                "Metric": ["Best block offset (H)", "Annual revenue"],
                "Value": [best_H, best_total_revenue]
            })
            final_results[zone] = (best_H_results, summary_df)

    return final_results


# ---------- Step 4: Run ----------
if __name__ == "__main__":
    # Load the 15-min data.
    df_15min = load_generic_lmp_files(file_paths, iso_name=iso_name)
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
        for zone, (df, summary_df) in results.items():
            df.to_excel(writer, sheet_name=zone, index=False)
            # for cell in ["A{}".format(len(df) + 3), "B{}".format(len(df) + 3), "A{}".format(len(df) + 4), "B{}".format(len(df) + 4)]:
            #     writer.sheets[zone].cell(cell).style = 'Input'
            summary_df.to_excel(writer, sheet_name=zone, index=False, startrow=len(df) + 2)

    print(f"\n✅ Benchmark complete. Output saved to:\n{output_path}")
