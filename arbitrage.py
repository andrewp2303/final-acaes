import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List

from arbitrage_types import PricingDataset, ArbitrageStrategy, ChargeDischargeAction


def evaluate_arbitrage(prices: PricingDataset, strategy: ArbitrageStrategy, 
                     max_charge_mw: float, max_discharge_mw: float,
                     storage_capacity_mwh: float, rte: float, 
                     charge_efficiency: float = None, discharge_efficiency: float = None) -> Dict:
    """
    Evaluate an arbitrage strategy against actual prices and system constraints.
    
    Args:
        prices: PricingDataset with actual prices
        strategy: ArbitrageStrategy to evaluate
        max_charge_mw: Maximum charging power in MW
        max_discharge_mw: Maximum discharging power in MW
        storage_capacity_mwh: Storage capacity in MWh
        rte: Round-trip efficiency (0-1)
        charge_efficiency: Optional charging efficiency (0-1)
        discharge_efficiency: Optional discharging efficiency (0-1)
        
    Returns:
        Dictionary with evaluation results including:
        - total_profit: Net profit from the strategy
        - total_cost: Total cost of charging
        - total_revenue: Total revenue from discharging
        - is_feasible: Whether the strategy is feasible given constraints
        - soc_profile: State of charge profile over time
        - actions: List of actions with financial details
        - validation_error: Details about validation failure if not feasible
    """
    # If separate efficiencies aren't provided, derive them from RTE
    if charge_efficiency is None or discharge_efficiency is None:
        # Default to equal distribution of losses if not specified
        charge_efficiency = np.sqrt(rte)
        discharge_efficiency = np.sqrt(rte)
    
    # First validate the strategy against system constraints
    validation_result = strategy.validate(
        max_charge_mw=max_charge_mw,
        max_discharge_mw=max_discharge_mw,
        storage_capacity_mwh=storage_capacity_mwh,
        rte=charge_efficiency * discharge_efficiency,  # Compute effective RTE
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency
    )
    
    # Check if validation failed and return error details
    if validation_result is not True:
        print(f"Strategy validation failed: {validation_result['error']}")
        
        # For overlapping actions, show both actions and their overlap period
        if 'action1' in validation_result and 'action2' in validation_result:
            action1 = validation_result['action1']
            action2 = validation_result['action2']
            print(f"Action 1: {'Charge' if action1.is_charge else 'Discharge'} from {action1.start_time} to {action1.end_time} at {action1.power_mw} MW")
            print(f"Action 2: {'Charge' if action2.is_charge else 'Discharge'} from {action2.start_time} to {action2.end_time} at {action2.power_mw} MW")
            overlap_start = max(action1.start_time, action2.start_time)
            overlap_end = min(action1.end_time, action2.end_time)
            overlap_duration = (overlap_end - overlap_start).total_seconds() / 3600
            print(f"Overlap period: {overlap_start} to {overlap_end} ({overlap_duration:.2f} hours)")
        
        # For other types of errors
        elif 'action' in validation_result:
            action = validation_result['action']
            print(f"Problematic action: {'Charge' if action.is_charge else 'Discharge'} from {action.start_time} to {action.end_time} at {action.power_mw} MW")
        
        if 'deficit' in validation_result:
            print(f"Energy deficit: {validation_result['deficit']:.2f} MWh")
        elif 'overflow' in validation_result:
            print(f"Energy overflow: {validation_result['overflow']:.2f} MWh")
        
        return {
            "total_profit": 0.0,
            "total_cost": 0.0,
            "total_revenue": 0.0,
            "is_feasible": False,
            "soc_profile": pd.Series(),
            "actions": [],
            "validation_error": validation_result
        }
    
    # Sort actions by start time
    sorted_actions = sorted(strategy.actions, key=lambda x: x.start_time)
    
    # Initialize tracking variables
    total_cost = 0.0
    total_revenue = 0.0
    soc_mwh = 0.0
    
    # Track SOC over time
    soc_profile = []
    action_details = []
    
    # Process each action
    for action in sorted_actions:
        # Get prices for this action's time period
        action_prices = prices.get_prices(
            start_time=action.start_time,
            end_time=action.end_time
        )
        
        # Calculate duration in hours
        duration_hours = (action.end_time - action.start_time).total_seconds() / 3600
        
        # Calculate financial impact
        if action.is_charge:
            # Charging: we pay for energy
            energy_in = action.power_mw * duration_hours * charge_efficiency
            
            # Calculate cost based on exact time spent in each hour
            cost = 0
            start_time = action.start_time
            end_time = action.end_time
            
            # Iterate through each hour in the price series
            for i, price in enumerate(action_prices):
                hour_start = action_prices.index[i]
                hour_end = hour_start + timedelta(hours=1)
                
                # Calculate overlap with this hour
                overlap_start = max(start_time, hour_start)
                overlap_end = min(end_time, hour_end)
                
                if overlap_end > overlap_start:
                    # Calculate fraction of hour used
                    fraction = (overlap_end - overlap_start).total_seconds() / 3600
                    # Calculate cost for this hour
                    hour_cost = action.power_mw * fraction * price
                    cost += hour_cost
            
            total_cost += cost
            soc_mwh += energy_in
            
            action_detail = {
                "type": "charge",
                "start_time": action.start_time,
                "end_time": action.end_time,
                "power_mw": action.power_mw,
                "energy_mwh": energy_in,
                "avg_price": action_prices.mean() if len(action_prices) > 0 else 0,
                "cost": cost,
                "soc_after": soc_mwh
            }
        else:
            # Discharging: we get paid for energy
            # For discharge: The energy taken from storage (energy_out) is more than the energy delivered to the grid
            # due to efficiency losses. So we need to divide by discharge_efficiency.
            energy_out = action.power_mw * duration_hours / discharge_efficiency
            
            # Calculate revenue based on exact time spent in each hour
            revenue = 0
            start_time = action.start_time
            end_time = action.end_time
            
            # Iterate through each hour in the price series
            for i, price in enumerate(action_prices):
                hour_start = action_prices.index[i]
                hour_end = hour_start + timedelta(hours=1)
                
                # Calculate overlap with this hour
                overlap_start = max(start_time, hour_start)
                overlap_end = min(end_time, hour_end)
                
                if overlap_end > overlap_start:
                    # Calculate fraction of hour used
                    fraction = (overlap_end - overlap_start).total_seconds() / 3600
                    # Calculate revenue for this hour
                    hour_revenue = action.power_mw * fraction * price
                    revenue += hour_revenue
            
            total_revenue += revenue
            soc_mwh -= energy_out
            
            action_detail = {
                "type": "discharge",
                "start_time": action.start_time,
                "end_time": action.end_time,
                "power_mw": action.power_mw,
                "energy_mwh": energy_out,
                "avg_price": action_prices.mean() if len(action_prices) > 0 else 0,
                "revenue": revenue,
                "soc_after": soc_mwh
            }
        
        # Record SOC after this action
        soc_profile.append((action.end_time, soc_mwh))
        action_details.append(action_detail)
    
    # Calculate net revenue
    total_profit = total_revenue - total_cost
    
    # If we've made it this far, the strategy is feasible
    is_feasible = True
    
    # Convert SOC profile to Series
    if soc_profile:
        soc_df = pd.DataFrame(soc_profile, columns=['time', 'soc'])
        soc_series = pd.Series(soc_df['soc'].values, index=soc_df['time'])
    else:
        soc_series = pd.Series()
    
    return {
        "total_profit": total_profit,
        "total_cost": total_cost,
        "total_revenue": total_revenue,
        "is_feasible": is_feasible,
        "soc_profile": soc_series,
        "actions": action_details
    }


def summarize_evaluation(evaluation_result: Dict, verbose: bool = False) -> str:
    """
    Generate a human-readable summary of an arbitrage evaluation.
    
    Args:
        evaluation_result: Result from evaluate_arbitrage function
        verbose: If True, include detailed information about each action
        
    Returns:
        Formatted string with evaluation summary
    """
    if not evaluation_result["is_feasible"]:
        summary = ["Strategy is not feasible with the given system constraints."]
        
        # Include validation error details if available
        if "validation_error" in evaluation_result and evaluation_result["validation_error"]:
            error_info = evaluation_result["validation_error"]
            summary.append(f"\nValidation Error: {error_info['error']}")
            
            if 'action' in error_info:
                action = error_info['action']
                summary.append(f"Problematic action: {'Charge' if action.is_charge else 'Discharge'} ")
                summary.append(f"  From: {action.start_time}")
                summary.append(f"  To: {action.end_time}")
                summary.append(f"  Power: {action.power_mw} MW")
                
                duration_hours = (action.end_time - action.start_time).total_seconds() / 3600
                summary.append(f"  Duration: {duration_hours:.2f} hours")
            
            if 'deficit' in error_info:
                summary.append(f"Energy deficit: {error_info['deficit']:.2f} MWh")
                summary.append(f"Current SOC: {error_info['current_soc']:.2f} MWh")
                summary.append(f"Energy needed: {error_info['energy_needed']:.2f} MWh")
            elif 'overflow' in error_info:
                summary.append(f"Energy overflow: {error_info['overflow']:.2f} MWh")
                summary.append(f"Current SOC: {error_info['current_soc']:.2f} MWh")
                summary.append(f"Capacity: {error_info['capacity']:.2f} MWh")
                summary.append(f"Energy to add: {error_info['energy_to_add']:.2f} MWh")
        
        return "\n".join(summary)
    
    summary = ["\nArbitrage Strategy Evaluation:"]
    summary.append(f"Total Discharging Revenue: ${evaluation_result['total_revenue']:.2f}")
    summary.append(f"- Total Charging Cost: ${evaluation_result['total_cost']:.2f}")
    summary.append("-" * 30)
    summary.append(f"Total Arbitrage Profit: ${evaluation_result['total_profit']:.2f}")
    
    # Count charge and discharge actions
    if evaluation_result["actions"]:
        charge_actions = [a for a in evaluation_result["actions"] if a["type"] == "charge"]
        discharge_actions = [a for a in evaluation_result["actions"] if a["type"] == "discharge"]
        
        total_charge_energy = sum(a["energy_mwh"] for a in charge_actions)
        total_discharge_energy = sum(a["energy_mwh"] for a in discharge_actions)
        
        summary.append(f"\nTotal Actions: {len(evaluation_result['actions'])}")
        summary.append(f"Charge Actions: {len(charge_actions)}")
        summary.append(f"Discharge Actions: {len(discharge_actions)}")
        summary.append(f"Total Energy Charged: {total_charge_energy:.2f} MWh")
        summary.append(f"Total Energy Discharged: {total_discharge_energy:.2f} MWh")
        
        # Calculate average prices
        avg_charge_price = sum(a["avg_price"] * a["energy_mwh"] for a in charge_actions) / total_charge_energy if total_charge_energy > 0 else 0
        avg_discharge_price = sum(a["avg_price"] * a["energy_mwh"] for a in discharge_actions) / total_discharge_energy if total_discharge_energy > 0 else 0
        price_spread = avg_discharge_price - avg_charge_price
        
        summary.append(f"Average Charge LMP: ${avg_charge_price:.2f}/MWh")
        summary.append(f"Average Discharge LMP: ${avg_discharge_price:.2f}/MWh")
        summary.append(f"Price Spread: ${price_spread:.2f}/MWh")
        
        # If verbose, show detailed actions
        if verbose:
            summary.append("\nDetailed Actions:")
            for i, action in enumerate(evaluation_result["actions"]):
                if action["type"] == "charge":
                    summary.append(f"  {i+1}. CHARGE: {action['start_time']} to {action['end_time']}")
                    summary.append(f"     Power: {action['power_mw']:.2f} MW, Energy: {action['energy_mwh']:.2f} MWh")
                    summary.append(f"     Avg Price: ${action['avg_price']:.2f}, Cost: ${action['cost']:.2f}")
                else:
                    summary.append(f"  {i+1}. DISCHARGE: {action['start_time']} to {action['end_time']}")
                    summary.append(f"     Power: {action['power_mw']:.2f} MW, Energy: {action['energy_mwh']:.2f} MWh")
                    summary.append(f"     Avg Price: ${action['avg_price']:.2f}, Revenue: ${action['revenue']:.2f}")
    
    return "\n".join(summary)


def compare_strategies(prices: PricingDataset, strategies: Dict[str, ArbitrageStrategy],
                      max_charge_mw: float, max_discharge_mw: float,
                      storage_capacity_mwh: float, rte: float) -> pd.DataFrame:
    """
    Compare multiple arbitrage strategies.
    
    Args:
        prices: PricingDataset with actual prices
        strategies: Dictionary mapping strategy names to ArbitrageStrategy objects
        max_charge_mw: Maximum charging power in MW
        max_discharge_mw: Maximum discharging power in MW
        storage_capacity_mwh: Storage capacity in MWh
        rte: Round-trip efficiency (0-1)
        charge_efficiency: Charging efficiency (0-1)
        discharge_efficiency: Discharging efficiency (0-1)
        
    Returns:
        DataFrame comparing strategy performance
    """
    results = []
    
    for name, strategy in strategies.items():
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
        
        results.append({
            "strategy": name,
            "profit": evaluation["total_profit"],
            "cost": evaluation["total_cost"],
            "revenue": evaluation["total_revenue"],
            "is_feasible": evaluation["is_feasible"],
            "num_actions": len(evaluation["actions"])
        })
    
    return pd.DataFrame(results).sort_values(by="profit", ascending=False)
