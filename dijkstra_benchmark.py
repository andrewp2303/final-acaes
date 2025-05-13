import numpy as np
from datetime import timedelta
from tqdm import tqdm
import logging

from arbitrage_types import PricingDataset, ArbitrageStrategy, ChargeDischargeAction, ArbitrageAlgorithm
from data import process_lmp_data
from arbitrage import evaluate_arbitrage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DijkstraArbitrageAlgorithm(ArbitrageAlgorithm):
    """
    Implements a receding-horizon (sliding-window) Dijkstra-based arbitrage strategy
    for CAES energy storage.
    """
    def __init__(self,
                 window_hours: int = 48,
                 soc_bins: int = 50,
                 time_step_hours: float = 1.0):
        self.window = window_hours
        self.bins = soc_bins
        self.dt = time_step_hours  # in hours
        self.safety_margin = 0.05  # 5% safety margin to avoid floating point issues

    def generate_strategy(self,
                          prices: PricingDataset,
                          max_charge_mw: float,
                          max_discharge_mw: float,
                          storage_capacity_mwh: float,
                          charge_efficiency: float,
                          discharge_efficiency: float) -> ArbitrageStrategy:
        """
        Generates an arbitrage strategy using Dijkstra's shortest path algorithm with
        discretized SOC levels and receding horizon optimization.
        
        This implementation enforces strict SOC constraints:
        - Charging: Ensure SOC + energy_in*charge_efficiency <= capacity
        - Discharging: Ensure SOC >= energy_out/discharge_efficiency
        """
        # Extract time-indexed price series
        price_series = prices.get_prices()
        times = list(price_series.index)
        prices_arr = price_series.values
        total_steps = len(prices_arr)

        # Discretize SOC levels
        soc_levels = np.linspace(0.0, storage_capacity_mwh, self.bins)

        current_soc = 0.0
        actions = []
        action_times = set()

        # Define a function to find the closest SOC bin index
        def find_soc_bin(soc_value):
            return int(np.abs(soc_levels - soc_value).argmin())
        
        # Set a strict effective capacity limit with safety margin to avoid floating point issues
        # and account for discretization errors in the SOC bins
        effective_capacity = storage_capacity_mwh * (1 - self.safety_margin)
        logging.info(f"Using effective storage capacity of {effective_capacity:.2f} MWh ({self.safety_margin*100:.1f}% safety margin)")
        
        for start in tqdm(range(0, total_steps - self.window), desc="Optimizing windows"):
            window_prices = prices_arr[start : start + self.window + 1]
            T = self.window

            dp = np.full((T+1, self.bins), np.inf)
            back_ptr = {}

            i0 = find_soc_bin(current_soc) 
            dp[0, i0] = 0.0

            for t in range(T):
                price = window_prices[t]
                for i in range(self.bins):
                    cost_so_far = dp[t, i]
                    if not np.isfinite(cost_so_far):
                        continue
                    soc = soc_levels[i]

                    for action_power in (-max_discharge_mw, 0.0, max_charge_mw):
                        # Calculate energy delta based on action type
                        if action_power > 0:  # Charging
                            energy_in = action_power * self.dt
                            delta = energy_in * charge_efficiency
                            # Check if charging would exceed capacity (with safety margin)
                            if soc + delta > effective_capacity:
                                continue
                        elif action_power < 0:  # Discharging
                            # Energy removed from storage is more than delivered
                            energy_removed = -action_power * self.dt / discharge_efficiency
                            # Check if discharging would go below 0
                            if soc < energy_removed:
                                continue
                            delta = -energy_removed
                        else:
                            delta = 0.0

                        new_soc = soc + delta

                        j = int(np.abs(soc_levels - new_soc).argmin())

                        step_cost = price * action_power * self.dt
                        new_cost = cost_so_far + step_cost

                        if new_cost < dp[t+1, j]:
                            dp[t+1, j] = new_cost
                            back_ptr[(t+1, j)] = (t, i, action_power)

            # Find minimum cost state at final timestep (after filtering NaN/Inf)
            final_costs = dp[T]
            valid_indices = np.isfinite(final_costs)
            
            if not np.any(valid_indices):
                # No valid path was found for this window
                continue
                
            j_final = np.nanargmin(final_costs)
            
            # Trace back the path
            path = []
            t, j = T, j_final
            while t > 0 and (t, j) in back_ptr:
                t_prev, i_prev, power = back_ptr[(t, j)]
                path.append((t_prev, i_prev, power))
                t, j = t_prev, i_prev
            path.reverse()

            # Reset SOC for this window's simulation
            window_soc = current_soc
            
            # Apply actions from the path and track SOC
            window_actions = []
            for t_offset, i_prev, power in path:
                if abs(power) < 1e-6:
                    continue
                    
                start_time = times[start + t_offset]
                if start_time in action_times:
                    continue  # skip duplicates
                    
                end_time = start_time + timedelta(hours=self.dt)
                
                # Check if action is viable with current SOC
                is_charge = power > 0
                power_mw = abs(power)
                
                # Double-check against our current window state, not just the DP state
                # This is critical because we have a discretized SOC in the DP table
                if is_charge:
                    energy_in = power_mw * self.dt
                    energy_stored = energy_in * charge_efficiency
                    
                    # Strictly enforce capacity constraint (with safety margin)
                    if window_soc + energy_stored > effective_capacity:
                        logging.info(f"Skipping charge action at {start_time}: SOC={window_soc:.2f}, would add {energy_stored:.2f} MWh")
                        continue
                        
                    # Super-extra-additional safety check (belt-and-suspenders approach)
                    next_soc = window_soc + energy_stored
                    if next_soc > effective_capacity:
                        logging.warning(f"Secondary safety check caught capacity violation: {next_soc:.2f} > {effective_capacity:.2f}")
                        continue
                        
                    window_soc = next_soc
                else:  # discharge
                    energy_out = power_mw * self.dt  # energy delivered to grid
                    energy_removed = energy_out / discharge_efficiency  # energy from storage
                    
                    # Verify we have enough energy
                    if window_soc < energy_removed:
                        logging.info(f"Skipping discharge action at {start_time}: SOC={window_soc:.2f}, would need {energy_removed:.2f} MWh")
                        continue
                    
                    # Additional safety check for discharge
                    next_soc = window_soc - energy_removed
                    if next_soc < 0:
                        logging.warning(f"Secondary safety check caught negative SOC: {next_soc:.2f}")
                        continue
                    
                    window_soc = next_soc
                
                # Add action to our window actions list
                new_action = ChargeDischargeAction(
                    start_time=start_time,
                    end_time=end_time,
                    is_charge=is_charge,
                    power_mw=power_mw
                )
                window_actions.append(new_action)
                action_times.add(start_time)
                
            # Add all window actions to our main actions list
            actions.extend(window_actions)
            
            # Update the real SOC based on what actually happened
            current_soc = window_soc

        # Pre-validate the entire strategy before returning
        # This performs a full SOC simulation as a final check
        final_actions = []
        soc = 0.0  # Start from empty storage
        
        # Sort actions by start time
        sorted_actions = sorted(actions, key=lambda x: x.start_time)
        
        for action in sorted_actions:
            duration_hours = (action.end_time - action.start_time).total_seconds() / 3600
            
            if action.is_charge:
                energy_in = action.power_mw * duration_hours
                energy_stored = energy_in * charge_efficiency
                
                # Final check that this action won't exceed capacity
                if soc + energy_stored > storage_capacity_mwh * 0.999:  # Leave 0.1% buffer
                    logging.warning(f"Pre-validation: Skipping charge at {action.start_time}: SOC={soc:.2f}, would add {energy_stored:.2f}")
                    continue
                    
                soc += energy_stored
            else:
                energy_out = action.power_mw * duration_hours  # energy delivered
                energy_removed = energy_out / discharge_efficiency  # energy from storage
                
                # Final check that we have enough energy
                if soc < energy_removed:
                    logging.warning(f"Pre-validation: Skipping discharge at {action.start_time}: SOC={soc:.2f}, would need {energy_removed:.2f}")
                    continue
                    
                soc -= energy_removed
                
            final_actions.append(action)
            
        logging.info(f"Strategy generation complete. Created {len(final_actions)} actions from original {len(actions)}")
        return ArbitrageStrategy(actions=final_actions)


if __name__ == "__main__":
    from data import load_pricing_dataset
    from arbitrage import summarize_evaluation

    pricing = load_pricing_dataset()

    max_charge_mw = 30.0
    max_discharge_mw = 30.0
    storage_capacity_mwh = 168.0

    charge_efficiency = 0.70
    discharge_efficiency = 0.85
    rte = charge_efficiency * discharge_efficiency

    print(f"System Specifications:")
    print(f"  Charging: {max_charge_mw} MW for {storage_capacity_mwh/(max_charge_mw*charge_efficiency):.1f} hours")
    print(f"  Discharging: {max_discharge_mw} MW for {(storage_capacity_mwh*discharge_efficiency)/max_discharge_mw:.1f} hours")
    print(f"  Storage capacity: {storage_capacity_mwh} MWh")
    print(f"  Charge efficiency: {charge_efficiency:.1%}")
    print(f"  Discharge efficiency: {discharge_efficiency:.1%}")
    print(f"  Round-trip efficiency: {rte:.1%}")
    print(f"  Maximum theoretical cycle value: ${storage_capacity_mwh * discharge_efficiency * 10:.2f} (at $10/MWh spread)\n")

    algo = DijkstraArbitrageAlgorithm(window_hours=48, soc_bins=50)
    strategy = algo.generate_strategy(
        pricing,
        max_charge_mw=max_charge_mw,
        max_discharge_mw=max_discharge_mw,
        storage_capacity_mwh=storage_capacity_mwh,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency
    )

    result_df = evaluate_arbitrage(
        pricing,
        strategy,
        max_charge_mw=max_charge_mw,
        max_discharge_mw=max_discharge_mw,
        storage_capacity_mwh=storage_capacity_mwh,
        rte=rte,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency
    )
    print(summarize_evaluation(result_df))