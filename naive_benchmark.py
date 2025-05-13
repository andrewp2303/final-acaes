"""
Naive Benchmark Algorithm for Energy Arbitrage

This algorithm implements a simple fixed-H approach where:
1. Each day is divided into 24-hour blocks starting at hour H
2. For each block, find the optimal charge/discharge periods
3. Charging must complete before discharging begins
4. One full cycle per day
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from types import PricingDataset, ArbitrageAlgorithm, ArbitrageStrategy, ChargeDischargeAction


class NaiveBenchmarkAlgorithm(ArbitrageAlgorithm):
    """
    Naive benchmark algorithm that uses a fixed-H approach.
    
    For each possible starting hour H (0-23), it divides the year into 24-hour blocks
    and finds the optimal charge/discharge periods within each block.
    """
    
    def generate_strategy(self, prices: PricingDataset, 
                         max_charge_mw: float, max_discharge_mw: float,
                         storage_capacity_mwh: float, rte: float) -> ArbitrageStrategy:
        """
        Generate an arbitrage strategy using the naive benchmark approach.
        
        Args:
            prices: PricingDataset with price predictions
            max_charge_mw: Maximum charging power in MW
            max_discharge_mw: Maximum discharging power in MW
            storage_capacity_mwh: Storage capacity in MWh
            rte: Round-trip efficiency (0-1)
            
        Returns:
            ArbitrageStrategy object
        """
        # Calculate charge and discharge durations
        t_charge_hrs = int(storage_capacity_mwh / max_charge_mw)
        t_discharge_hrs = int(storage_capacity_mwh / max_discharge_mw)
        
        # Get prices as a DataFrame with datetime index
        price_data = prices.data
        
        # Find the best starting hour H
        best_H = None
        best_total_revenue = -np.inf
        best_actions = []
        
        # Try each possible starting hour H
        for H in range(24):
            actions, total_revenue = self._evaluate_fixed_H(
                price_data=price_data,
                H=H,
                max_charge_mw=max_charge_mw,
                max_discharge_mw=max_discharge_mw,
                t_charge_hrs=t_charge_hrs,
                t_discharge_hrs=t_discharge_hrs,
                rte=rte
            )
            
            if total_revenue > best_total_revenue:
                best_H = H
                best_total_revenue = total_revenue
                best_actions = actions
        
        print(f"Best starting hour H: {best_H}")
        print(f"Total revenue: ${best_total_revenue:.2f}")
        
        return ArbitrageStrategy(actions=best_actions)
    
    def _evaluate_fixed_H(self, price_data: pd.DataFrame, H: int,
                         max_charge_mw: float, max_discharge_mw: float,
                         t_charge_hrs: int, t_discharge_hrs: int,
                         rte: float) -> Tuple[List[ChargeDischargeAction], float]:
        """
        Evaluate a specific starting hour H.
        
        Args:
            price_data: DataFrame with LMP prices
            H: Starting hour (0-23)
            max_charge_mw: Maximum charging power in MW
            max_discharge_mw: Maximum discharging power in MW
            t_charge_hrs: Number of hours to charge
            t_discharge_hrs: Number of hours to discharge
            rte: Round-trip efficiency (0-1)
            
        Returns:
            Tuple of (list of ChargeDischargeAction, total revenue)
        """
        # Get all unique days in the dataset
        all_days = pd.Series(price_data.index.date).unique()
        all_days = [datetime.combine(day, datetime.min.time()) for day in all_days]
        
        actions = []
        total_revenue = 0.0
        charge_efficiency = np.sqrt(rte)
        discharge_efficiency = np.sqrt(rte)
        
        # Process each day
        for day in all_days:
            # Define the 24-hour block starting at hour H
            block_start = day + timedelta(hours=H)
            block_end = block_start + timedelta(hours=24)
            
            # Get prices for this block
            block_mask = (price_data.index >= block_start) & (price_data.index < block_end)
            block = price_data.loc[block_mask]
            
            if len(block) < 24:
                continue  # Skip incomplete blocks
            
            # Find optimal charge/discharge periods
            best_revenue = -np.inf
            best_charge_start = None
            best_charge_end = None
            best_discharge_start = None
            best_discharge_end = None
            
            # Try all possible combinations
            for t_c in range(0, 24 - t_charge_hrs + 1):
                t_charge_end = t_c + t_charge_hrs
                
                for t_d in range(t_charge_end, 24 - t_discharge_hrs + 1):
                    t_discharge_end = t_d + t_discharge_hrs
                    
                    # Get prices for charge and discharge periods
                    charge_times = block.index[t_c:t_charge_end]
                    discharge_times = block.index[t_d:t_discharge_end]
                    
                    charge_prices = block.loc[charge_times, 'LMP'].values
                    discharge_prices = block.loc[discharge_times, 'LMP'].values
                    
                    # Calculate revenue
                    cost = max_charge_mw * sum(charge_prices)
                    revenue = max_discharge_mw * sum(discharge_prices)
                    
                    # Apply efficiency
                    net = revenue * discharge_efficiency - cost / charge_efficiency
                    
                    if net > best_revenue:
                        best_revenue = net
                        best_charge_start = charge_times[0]
                        best_charge_end = charge_times[-1] + timedelta(hours=1)  # End time is exclusive
                        best_discharge_start = discharge_times[0]
                        best_discharge_end = discharge_times[-1] + timedelta(hours=1)  # End time is exclusive
            
            # Add actions if profitable
            if best_revenue > 0 and best_charge_start is not None:
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
                    end_time=best_discharge_end,
                    is_charge=False,
                    power_mw=max_discharge_mw
                ))
                
                total_revenue += best_revenue
        
        return actions, total_revenue


if __name__ == "__main__":
    from data import load_pricing_dataset
    from arbitrage import evaluate_arbitrage, summarize_evaluation
    
    # Load the dataset
    prices = load_pricing_dataset()
    
    # System parameters
    max_charge_mw = 30.0
    max_discharge_mw = 30.0
    storage_capacity_mwh = 240.0  # 30 MW * 8 hours
    rte = 0.85  # Round-trip efficiency
    
    # Create and run the algorithm
    algorithm = NaiveBenchmarkAlgorithm()
    strategy = algorithm.generate_strategy(
        prices=prices,
        max_charge_mw=max_charge_mw,
        max_discharge_mw=max_discharge_mw,
        storage_capacity_mwh=storage_capacity_mwh,
        rte=rte
    )
    
    # Evaluate the strategy
    evaluation = evaluate_arbitrage(
        prices=prices,
        strategy=strategy,
        max_charge_mw=max_charge_mw,
        max_discharge_mw=max_discharge_mw,
        storage_capacity_mwh=storage_capacity_mwh,
        rte=rte
    )
    
    # Print summary
    print(summarize_evaluation(evaluation))
