import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List

from arbitrage_types import PricingDataset, ArbitrageStrategy, ChargeDischargeAction


def evaluate_arbitrage(prices: PricingDataset, strategy: ArbitrageStrategy, 
                     max_charge_mw: float, max_discharge_mw: float,
                     storage_capacity_mwh: float, rte: float) -> Dict:
    """
    Evaluate an arbitrage strategy against actual prices and system constraints.
    
    Args:
        prices: PricingDataset with actual prices
        strategy: ArbitrageStrategy to evaluate
        max_charge_mw: Maximum charging power in MW
        max_discharge_mw: Maximum discharging power in MW
        storage_capacity_mwh: Storage capacity in MWh
        rte: Round-trip efficiency (0-1)
        
    Returns:
        Dictionary with evaluation results including:
        - total_revenue: Net revenue from the strategy
        - total_cost: Total cost of charging
        - total_income: Total income from discharging
        - is_feasible: Whether the strategy is feasible given constraints
        - soc_profile: State of charge profile over time
        - actions: List of actions with financial details
    """
    # First validate the strategy against system constraints
    is_feasible = strategy.validate(
        max_charge_mw=max_charge_mw,
        max_discharge_mw=max_discharge_mw,
        storage_capacity_mwh=storage_capacity_mwh,
        rte=rte
    )
    
    if not is_feasible:
        return {
            "total_revenue": 0.0,
            "total_cost": 0.0,
            "total_income": 0.0,
            "is_feasible": False,
            "soc_profile": pd.Series(),
            "actions": []
        }
    
    # Sort actions by start time
    sorted_actions = sorted(strategy.actions, key=lambda x: x.start_time)
    
    # Initialize tracking variables
    total_cost = 0.0
    total_income = 0.0
    soc_mwh = 0.0
    charge_efficiency = np.sqrt(rte)
    discharge_efficiency = np.sqrt(rte)
    
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
            cost = action.power_mw * sum(action_prices)
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
            energy_out = action.power_mw * duration_hours / discharge_efficiency
            income = action.power_mw * sum(action_prices)
            total_income += income
            soc_mwh -= energy_out
            
            action_detail = {
                "type": "discharge",
                "start_time": action.start_time,
                "end_time": action.end_time,
                "power_mw": action.power_mw,
                "energy_mwh": energy_out,
                "avg_price": action_prices.mean() if len(action_prices) > 0 else 0,
                "income": income,
                "soc_after": soc_mwh
            }
        
        # Record SOC after this action
        soc_profile.append((action.end_time, soc_mwh))
        action_details.append(action_detail)
    
    # Calculate net revenue
    total_revenue = total_income - total_cost
    
    # Convert SOC profile to Series
    if soc_profile:
        soc_df = pd.DataFrame(soc_profile, columns=['time', 'soc'])
        soc_series = pd.Series(soc_df['soc'].values, index=soc_df['time'])
    else:
        soc_series = pd.Series()
    
    return {
        "total_revenue": total_revenue,
        "total_cost": total_cost,
        "total_income": total_income,
        "is_feasible": is_feasible,
        "soc_profile": soc_series,
        "actions": action_details
    }


def summarize_evaluation(evaluation_result: Dict) -> str:
    """
    Generate a human-readable summary of an arbitrage evaluation.
    
    Args:
        evaluation_result: Result from evaluate_arbitrage function
        
    Returns:
        Formatted string with evaluation summary
    """
    if not evaluation_result["is_feasible"]:
        return "Strategy is not feasible with the given system constraints."
    
    summary = ["Arbitrage Strategy Evaluation"]
    summary.append("-" * 30)
    summary.append(f"Total Revenue: ${evaluation_result['total_revenue']:.2f}")
    summary.append(f"Total Cost: ${evaluation_result['total_cost']:.2f}")
    summary.append(f"Total Income: ${evaluation_result['total_income']:.2f}")
    
    # Summarize actions
    if evaluation_result["actions"]:
        summary.append("\nActions:")
        for i, action in enumerate(evaluation_result["actions"]):
            if action["type"] == "charge":
                summary.append(f"  {i+1}. CHARGE: {action['start_time']} to {action['end_time']}")
                summary.append(f"     Power: {action['power_mw']:.2f} MW, Energy: {action['energy_mwh']:.2f} MWh")
                summary.append(f"     Avg Price: ${action['avg_price']:.2f}, Cost: ${action['cost']:.2f}")
            else:
                summary.append(f"  {i+1}. DISCHARGE: {action['start_time']} to {action['end_time']}")
                summary.append(f"     Power: {action['power_mw']:.2f} MW, Energy: {action['energy_mwh']:.2f} MWh")
                summary.append(f"     Avg Price: ${action['avg_price']:.2f}, Income: ${action['income']:.2f}")
    
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
        
    Returns:
        DataFrame comparing strategy performance
    """
    results = []
    
    for name, strategy in strategies.items():
        evaluation = evaluate_arbitrage(
            prices=prices,
            strategy=strategy,
            max_charge_mw=max_charge_mw,
            max_discharge_mw=max_discharge_mw,
            storage_capacity_mwh=storage_capacity_mwh,
            rte=rte
        )
        
        results.append({
            "strategy": name,
            "revenue": evaluation["total_revenue"],
            "cost": evaluation["total_cost"],
            "income": evaluation["total_income"],
            "is_feasible": evaluation["is_feasible"],
            "num_actions": len(evaluation["actions"])
        })
    
    return pd.DataFrame(results).sort_values(by="revenue", ascending=False)
