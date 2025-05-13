"""
Naive Benchmark Algorithm for Energy Arbitrage

This algorithm implements a simple fixed-H approach where, for each H in [0, 23]:
1. Each day is divided into 24-hour blocks starting at hour H
2. For each day, evaluate all possible charge start offsets (0, 1, ..., floor(24 - t_charge_hrs - t_discharge_hrs) )
3. For each of these charge starts, evaluate all possible discharge start offsets (ceil(charge_end), ..., floor(24 - t_discharge_hrs))
4. One full cycle per day (charge and discharge must complete before the next day)
5. For each day, select the most profitable cycle
6. Combine the best cycles from each day to form a strategy for a given H
7. Select the H that results in the highest total profit, and return the strategy for that H

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from tqdm import tqdm
import pytz
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

from arbitrage_types import PricingDataset, ArbitrageAlgorithm, ArbitrageStrategy, ChargeDischargeAction


def process_H_parallel(H, price_data, max_charge_mw, max_discharge_mw, t_charge_hrs, t_discharge_hrs, 
                      charge_efficiency, discharge_efficiency, storage_capacity_mwh):
    """
    Process a single starting hour H in parallel.
    
    Args:
        H: Starting hour (0-23)
        price_data: DataFrame with LMP prices
        max_charge_mw: Maximum charging power in MW
        max_discharge_mw: Maximum discharging power in MW
        t_charge_hrs: Number of hours to charge
        t_discharge_hrs: Number of hours to discharge
        charge_efficiency: Charging efficiency (0-1)
        discharge_efficiency: Discharging efficiency (0-1)
        storage_capacity_mwh: Storage capacity in MWh
        
    Returns:
        Tuple of (H, actions, total_profit)
    """
    # Extract the timezone from the price data
    pacific_tz = pytz.timezone('US/Pacific')
    
    # Get all unique dates from the price data
    # Convert to Pacific time first to handle DST correctly
    dates = set()
    for dt in price_data.index:
        # Get date part only, in Pacific time
        dt_pacific = dt.astimezone(pacific_tz)
        dates.add(dt_pacific.date())
    
    # Sort the dates and create datetime objects with proper timezone
    all_days = sorted(list(dates))
    all_days = [pacific_tz.localize(datetime.combine(day, datetime.min.time())) for day in all_days]
    
    actions = []
    total_profit = 0.0
    
    # Process each day with progress bar
    for day in tqdm(all_days, desc=f"Processing days for H={H}", leave=False):
        # Define the 24-hour block starting at hour H
        # The day already has the correct timezone from our preprocessing
        block_start = day.replace(hour=H, minute=0, second=0, microsecond=0)
        block_end = block_start + timedelta(hours=24)
        
        # Handle DST transitions by ensuring we're using wall clock time
        # This ensures we always get 24 wall-clock hours regardless of DST changes
        if block_end.fold != block_start.fold:  # Indicates a DST transition
            # Adjust for DST transition by using wall clock time
            block_end = pacific_tz.normalize(block_end)
        
        # Get prices for this block
        block_mask = (price_data.index >= block_start) & (price_data.index < block_end)
        block = price_data.loc[block_mask]
        
        if len(block) < 24:
            continue  # Skip incomplete blocks, e.g. DST transitions
        
        # Find optimal charge/discharge periods
        best_profit = -np.inf
        best_charge_start = None
        best_charge_end = None
        best_discharge_start = None
        best_discharge_end = None
        
        # Try all possible combinations
        for t_c in range(0, 24 - math.floor(t_charge_hrs) + 1):
            t_charge_end = t_c + t_charge_hrs
            
            for t_d in range(math.ceil(t_charge_end), 24 - math.floor(t_discharge_hrs) + 1):
                t_discharge_end = t_d + t_discharge_hrs
                
                # Get prices for charge and discharge periods
                charge_times = block.index[t_c:math.ceil(t_charge_end)]
                discharge_times = block.index[t_d:math.ceil(t_discharge_end)]
                
                if len(charge_times) == 0 or len(discharge_times) == 0:
                    continue
                
                charge_prices = block.loc[charge_times, 'LMP'].values
                discharge_prices = block.loc[discharge_times, 'LMP'].values
                
                # Calculate exact charge duration in hours (may be fractional)
                charge_duration_hrs = min(t_charge_hrs, len(charge_times))
                
                # Calculate exact discharge duration in hours (may be fractional)
                discharge_duration_hrs = min(t_discharge_hrs, len(discharge_times))
                
                # Calculate energy stored after charging (accounting for efficiency)
                energy_charged = max_charge_mw * charge_duration_hrs * charge_efficiency
                energy_charged = min(energy_charged, storage_capacity_mwh)  # Cap at storage capacity
                
                # Calculate energy available for discharge (limited by stored energy)
                energy_discharged = min(energy_charged, max_discharge_mw * discharge_duration_hrs / discharge_efficiency)
                
                # Calculate actual energy delivered to grid
                energy_delivered = energy_discharged * discharge_efficiency
                
                # Calculate cost and revenue using average prices and exact energy amounts
                avg_charge_price = sum(charge_prices) / len(charge_prices) if len(charge_prices) > 0 else 0
                avg_discharge_price = sum(discharge_prices) / len(discharge_prices) if len(discharge_prices) > 0 else 0
                
                cost = max_charge_mw * charge_duration_hrs * avg_charge_price
                revenue = energy_delivered * avg_discharge_price
                
                profit = revenue - cost
                
                if profit > best_profit:
                    best_profit = profit
                    best_charge_start = charge_times[0]
                    best_charge_end = charge_times[-1] + timedelta(hours=1)  # End time is exclusive
                    best_discharge_start = discharge_times[0]
                    best_discharge_end = discharge_times[-1] + timedelta(hours=1)  # End time is exclusive
        
        # Add actions if profitable
        if best_profit > 0 and best_charge_start is not None:
            # Calculate energy stored after charging (accounting for efficiency)
            energy_charged = max_charge_mw * (best_charge_end - best_charge_start).total_seconds() / 3600 * charge_efficiency
            
            # Calculate energy available for discharge (limited by storage capacity)
            energy_to_discharge = energy_charged
            
            # Calculate actual discharge duration (limited by energy available)
            # We need to divide by discharge power and then multiply by discharge efficiency
            # (or equivalently divide by (discharge power / discharge efficiency))
            discharge_duration_hrs = energy_to_discharge / (max_discharge_mw / discharge_efficiency)
            
            # Calculate the exact discharge end time using the precise duration
            # This avoids rounding errors from converting to hours/minutes
            discharge_seconds = int(discharge_duration_hrs * 3600)  # Convert hours to seconds exactly
            target_discharge_end = best_discharge_start + timedelta(seconds=discharge_seconds)
            
            # Ensure discharge doesn't extend beyond the current block's end time
            # This prevents overlap with the next day's actions
            if target_discharge_end > block_end:
                # If discharge would extend beyond block_end, truncate it
                target_discharge_end = block_end
            
            actual_discharge_end = target_discharge_end
            
            # Add charge action
            actions.append(ChargeDischargeAction(
                start_time=best_charge_start,
                end_time=best_charge_end,
                is_charge=True,
                power_mw=max_charge_mw
            ))
            
            # Add discharge action
            actions.append(ChargeDischargeAction(
                start_time=best_discharge_start,
                end_time=actual_discharge_end,
                is_charge=False,
                power_mw=max_discharge_mw
            ))
            
            total_profit += best_profit
    
    return H, actions, total_profit


class NaiveBenchmarkAlgorithm(ArbitrageAlgorithm):
    """
    Naive benchmark algorithm that uses a fixed-H approach.
    
    For each possible starting hour H (0-23), it divides the year into 24-hour blocks
    and finds the optimal charge/discharge periods within each block.
    """
    
    def generate_strategy(self, prices: PricingDataset, 
                         max_charge_mw: float, max_discharge_mw: float,
                         storage_capacity_mwh: float, 
                         charge_efficiency: float, discharge_efficiency: float,
                         num_workers: int = 8) -> ArbitrageStrategy:
        """
        Generate an arbitrage strategy using the naive benchmark approach.
        
        Args:
            prices: PricingDataset with price predictions
            max_charge_mw: Maximum charging power in MW
            max_discharge_mw: Maximum discharging power in MW
            storage_capacity_mwh: Storage capacity in MWh
            charge_efficiency: Charging efficiency (0-1)
            discharge_efficiency: Discharging efficiency (0-1)
            num_workers: Number of parallel processes to use
            
        Returns:
            ArbitrageStrategy object
        """
        # Calculate charge and discharge durations based on efficiencies
        # For charging: we need to account for losses during charging
        # To charge up to storage_capacity_mwh, we need to input more energy due to losses
        # So we need to divide by charge efficiency
        t_charge_hrs = storage_capacity_mwh / (max_charge_mw * charge_efficiency)
        
        # For discharging: we need to account for the discharge efficiency
        # The stored energy divided by discharge efficiency gives us the output energy
        # Then divide by power to get hours
        t_discharge_hrs = (storage_capacity_mwh * discharge_efficiency) / max_discharge_mw
        
        print(f"Calculated charge duration: {t_charge_hrs:.2f} hours")
        print(f"Calculated discharge duration: {t_discharge_hrs:.2f} hours")
        
        # Get prices as a DataFrame with datetime index
        price_data = prices.data
        
        # Parallelize over all H values (0-23)
        all_results = []
        
        # Use ProcessPoolExecutor to parallelize the evaluation of different H values
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = []
            
            # Create a progress bar for the H values
            for H in range(24):
                future = executor.submit(
                    process_H_parallel,
                    H, 
                    price_data, 
                    max_charge_mw, 
                    max_discharge_mw, 
                    t_charge_hrs, 
                    t_discharge_hrs, 
                    charge_efficiency, 
                    discharge_efficiency, 
                    storage_capacity_mwh
                )
                futures.append(future)
            
            # Process results as they complete with a progress bar
            for future in tqdm(as_completed(futures), total=24, desc="Evaluating starting hours"):
                H, actions, total_profit = future.result()
                all_results.append((H, actions, total_profit))
        
        # Find the best H
        best_H, best_actions, best_total_profit = max(all_results, key=lambda x: x[2])
        
        print(f"Best starting hour H: {best_H}")
        print(f"Total arbitrage profit: ${best_total_profit:.2f}")
        
        return ArbitrageStrategy(actions=best_actions)


if __name__ == "__main__":
    from data import load_pricing_dataset
    from arbitrage import evaluate_arbitrage, summarize_evaluation
    
    # Load the dataset
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
    print(f"  Charge efficiency: {charge_efficiency:.1%}")
    print(f"  Discharge efficiency: {discharge_efficiency:.1%}")
    print(f"  Round-trip efficiency: {rte:.1%}\n")
    
    # Create and run the algorithm
    algorithm = NaiveBenchmarkAlgorithm()
    strategy = algorithm.generate_strategy(
        prices=prices,
        max_charge_mw=max_charge_mw,
        max_discharge_mw=max_discharge_mw,
        storage_capacity_mwh=storage_capacity_mwh,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
        num_workers=8  # Use 8 parallel processes, adjust as needed
    )
    
    # Evaluate the strategy
    evaluation = evaluate_arbitrage(
        prices=prices,
        strategy=strategy,
        max_charge_mw=max_charge_mw,
        max_discharge_mw=max_discharge_mw,
        storage_capacity_mwh=storage_capacity_mwh,
        rte=rte,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency
    )
    
    # Print summary
    print(summarize_evaluation(evaluation))
