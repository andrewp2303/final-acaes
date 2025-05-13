"""
Generate standardized CSV output from arbitrage benchmark algorithms.

This script runs both the naive benchmark and Dijkstra benchmark algorithms,
extracts trade information, and saves it to standardized CSV files for plotting.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import os

from data import load_pricing_dataset
from naive_benchmark import NaiveBenchmarkAlgorithm
from dijkstra_benchmark import DijkstraArbitrageAlgorithm
from arbitrage import evaluate_arbitrage

def generate_trade_csv(algorithm_name, evaluation_result, output_file):
    """
    Generate a standardized CSV file from algorithm evaluation results.
    
    Args:
        algorithm_name: Name of the algorithm
        evaluation_result: Result from evaluate_arbitrage function
        output_file: Path to save the CSV file
    """
    # Extract action details
    actions = evaluation_result["actions"]
    
    # Create dataframe with standardized format
    records = []
    cumulative_revenue = 0.0
    
    for action in actions:
        # Get the date in Pacific time to handle DST correctly
        start_time_pacific = action["start_time"].astimezone(pytz.timezone('US/Pacific'))
        date = start_time_pacific.date()
        
        # Calculate trade amount and revenue
        if action["type"] == "charge":
            trade_amount = -action["energy_mwh"]  # Negative for charging (buying)
            trade_cost = action["cost"]
            cumulative_revenue -= trade_cost
        else:  # discharge
            trade_amount = action["energy_mwh"]  # Positive for discharging (selling)
            trade_revenue = action["revenue"]
            cumulative_revenue += trade_revenue
        
        records.append({
            "algorithm": algorithm_name,
            "date": date,
            "datetime": action["start_time"],
            "trade_type": action["type"],
            "power_mw": action["power_mw"],
            "energy_mwh": abs(trade_amount),  # Store as absolute value
            "price_per_mwh": action["avg_price"],
            "trade_amount": trade_amount,  # Negative for charge, positive for discharge
            "soc_after": action["soc_after"],
            "cumulative_revenue": cumulative_revenue
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"Saved trade data to {output_file}")
    
    return df

def main():
    # Load pricing dataset
    print("Loading pricing dataset...")
    prices = load_pricing_dataset()
    
    # System parameters for CAES facility
    max_charge_mw = 30.0        # 30 MW charging power
    max_discharge_mw = 30.0     # 30 MW discharging power
    storage_capacity_mwh = 168.0  # Energy stored in air pressure
    
    # Efficiency parameters
    charge_efficiency = 0.70    # 70% charging efficiency (30% losses)
    discharge_efficiency = 0.85  # 85% discharging efficiency (15% losses)
    rte = charge_efficiency * discharge_efficiency  # 59.5% round-trip efficiency
    
    print(f"System Specifications:")
    print(f"  Charging: {max_charge_mw} MW for {storage_capacity_mwh/(max_charge_mw*charge_efficiency):.1f} hours")
    print(f"  Discharging: {max_discharge_mw} MW for {(storage_capacity_mwh*discharge_efficiency)/max_discharge_mw:.1f} hours")
    print(f"  Storage capacity: {storage_capacity_mwh} MWh")
    print(f"  Round-trip efficiency: {rte:.1%}")
    
    # Create output directory if it doesn't exist
    output_dir = "trade_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Naive Benchmark algorithm
    print("\nRunning Naive Benchmark algorithm...")
    naive_algo = NaiveBenchmarkAlgorithm()
    naive_strategy = naive_algo.generate_strategy(
        prices=prices,
        max_charge_mw=max_charge_mw,
        max_discharge_mw=max_discharge_mw,
        storage_capacity_mwh=storage_capacity_mwh,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
        num_workers=8
    )
    
    # Evaluate Naive Benchmark
    print("Evaluating Naive Benchmark strategy...")
    naive_evaluation = evaluate_arbitrage(
        prices=prices,
        strategy=naive_strategy,
        max_charge_mw=max_charge_mw,
        max_discharge_mw=max_discharge_mw,
        storage_capacity_mwh=storage_capacity_mwh,
        rte=rte,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency
    )
    
    # Generate CSV for Naive Benchmark
    naive_csv = os.path.join(output_dir, "naive_benchmark_trades.csv")
    naive_df = generate_trade_csv("Naive Benchmark", naive_evaluation, naive_csv)
    
    # Run Dijkstra algorithm
    print("\nRunning Dijkstra algorithm...")
    dijkstra_algo = DijkstraArbitrageAlgorithm(window_hours=48, soc_bins=50)
    dijkstra_strategy = dijkstra_algo.generate_strategy(
        prices=prices,
        max_charge_mw=max_charge_mw,
        max_discharge_mw=max_discharge_mw,
        storage_capacity_mwh=storage_capacity_mwh,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency
    )
    
    # Evaluate Dijkstra
    print("Evaluating Dijkstra strategy...")
    dijkstra_evaluation = evaluate_arbitrage(
        prices=prices,
        strategy=dijkstra_strategy,
        max_charge_mw=max_charge_mw,
        max_discharge_mw=max_discharge_mw,
        storage_capacity_mwh=storage_capacity_mwh,
        rte=rte,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency
    )
    
    # Generate CSV for Dijkstra
    dijkstra_csv = os.path.join(output_dir, "dijkstra_benchmark_trades.csv")
    dijkstra_df = generate_trade_csv("Dijkstra", dijkstra_evaluation, dijkstra_csv)
    
    # Generate combined CSV for easy comparison
    combined_df = pd.concat([naive_df, dijkstra_df])
    combined_csv = os.path.join(output_dir, "combined_trades.csv")
    combined_df.to_csv(combined_csv, index=False)
    print(f"Saved combined trade data to {combined_csv}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Naive Benchmark: {len(naive_df)} trades, final revenue: ${naive_df['cumulative_revenue'].iloc[-1]:.2f}")
    print(f"Dijkstra: {len(dijkstra_df)} trades, final revenue: ${dijkstra_df['cumulative_revenue'].iloc[-1]:.2f}")
    
    # Create a daily revenue summary
    print("\nGenerating daily revenue summary...")
    
    # Function to calculate daily revenue
    def get_daily_revenue(df):
        # Group by date and calculate daily revenue
        daily = df.groupby('date').agg({
            'algorithm': 'first',
            'trade_type': 'count',
            'energy_mwh': 'sum',
            'cumulative_revenue': 'last'
        }).rename(columns={
            'trade_type': 'num_trades',
            'energy_mwh': 'total_energy_mwh',
            'cumulative_revenue': 'revenue_to_date'
        })
        
        # Calculate daily revenue (difference from previous day)
        daily['daily_revenue'] = daily['revenue_to_date'].diff()
        daily.loc[daily.index[0], 'daily_revenue'] = daily.loc[daily.index[0], 'revenue_to_date']
        
        return daily
    
    # Get daily revenue for each algorithm
    naive_daily = get_daily_revenue(naive_df)
    dijkstra_daily = get_daily_revenue(dijkstra_df)
    
    # Save daily revenue to CSV
    naive_daily_csv = os.path.join(output_dir, "naive_daily_revenue.csv")
    dijkstra_daily_csv = os.path.join(output_dir, "dijkstra_daily_revenue.csv")
    
    naive_daily.to_csv(naive_daily_csv)
    dijkstra_daily.to_csv(dijkstra_daily_csv)
    
    # Combine daily revenue
    combined_daily = pd.concat([naive_daily, dijkstra_daily])
    combined_daily_csv = os.path.join(output_dir, "combined_daily_revenue.csv")
    combined_daily.to_csv(combined_daily_csv)
    
    print(f"Saved daily revenue data to {output_dir}/")
    print("Done!")

if __name__ == "__main__":
    main()
