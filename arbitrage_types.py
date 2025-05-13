from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Protocol, Union
from datetime import datetime
import pandas as pd
import numpy as np


class PricingDataset:
    """
    Class representing a pricing dataset with hourly LMP values.
    """
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with a DataFrame containing HOUR and LMP columns.
        
        Args:
            data: DataFrame with HOUR (datetime with timezone) and LMP (float) columns
        """
        # Ensure the required columns exist
        required_cols = ['HOUR', 'LMP']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Dataset must contain columns: {required_cols}")
        
        self.data = data.copy()
        # Set HOUR as index if it's not already
        if 'HOUR' in self.data.columns:
            self.data.set_index('HOUR', inplace=True)
        
        # Ensure the data is sorted by time
        self.data.sort_index(inplace=True)
    
    @classmethod
    def from_csv(cls, file_path: str) -> 'PricingDataset':
        """
        Load pricing data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            PricingDataset object
        """
        data = pd.read_csv(file_path)
        # Convert HOUR to datetime if it's not already
        if 'HOUR' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['HOUR']):
            # Use utc=True to avoid FutureWarning about mixed timezones
            data['HOUR'] = pd.to_datetime(data['HOUR'], utc=True).dt.tz_convert('US/Pacific')
        return cls(data)
    
    def get_prices(self, start_time: Optional[datetime] = None, 
                  end_time: Optional[datetime] = None) -> pd.Series:
        """
        Get LMP prices for a specified time range.
        
        Args:
            start_time: Start time (inclusive)
            end_time: End time (inclusive)
            
        Returns:
            Series of LMP values indexed by datetime
        """
        if start_time is None and end_time is None:
            return self.data['LMP']
        
        mask = True
        if start_time is not None:
            mask = mask & (self.data.index >= start_time)
        if end_time is not None:
            mask = mask & (self.data.index <= end_time)
            
        return self.data.loc[mask, 'LMP']
    
    def get_hours_with_interpolated_data(self) -> pd.Series:
        """
        Get hours where the LMP data was interpolated.
        
        Returns:
            Boolean Series where True indicates interpolated data
        """
        if 'is_interpolated' in self.data.columns:
            return self.data['is_interpolated']
        return pd.Series(False, index=self.data.index)


@dataclass
class ChargeDischargeAction:
    """
    Represents a single charge or discharge action.
    """
    start_time: datetime  # Start time of the action
    end_time: datetime    # End time of the action
    is_charge: bool       # True for charging, False for discharging
    power_mw: float       # Power in MW (positive for both charge and discharge)


@dataclass
class ArbitrageStrategy:
    """
    Represents a complete arbitrage strategy with a series of charge/discharge actions.
    """
    actions: List[ChargeDischargeAction]
    
    def validate(self, max_charge_mw: float, max_discharge_mw: float, 
                storage_capacity_mwh: float, rte: float,
                charge_efficiency: float = None, discharge_efficiency: float = None) -> Union[bool, Dict]:
        """
        Validate that the strategy is feasible given system constraints.
        
        Args:
            max_charge_mw: Maximum charging power in MW
            max_discharge_mw: Maximum discharging power in MW
            storage_capacity_mwh: Storage capacity in MWh
            rte: Round-trip efficiency (0-1)
            charge_efficiency: Optional charging efficiency (0-1)
            discharge_efficiency: Optional discharging efficiency (0-1)
            
        Returns:
            True if strategy is feasible, or a dictionary with error details if not feasible
        """
        # Sort actions by start time
        sorted_actions = sorted(self.actions, key=lambda x: x.start_time)
        
        # Check for overlapping actions
        for i in range(len(sorted_actions) - 1):
            if sorted_actions[i].end_time > sorted_actions[i + 1].start_time:
                return {
                    'valid': False,
                    'error': 'Overlapping actions detected',
                    'action1': sorted_actions[i],
                    'action2': sorted_actions[i + 1]
                }
        
        # If separate efficiencies aren't provided, derive them from RTE
        if charge_efficiency is None or discharge_efficiency is None:
            # Default to equal distribution of losses
            charge_efficiency = np.sqrt(rte)
            discharge_efficiency = np.sqrt(rte)
        
        # Check power constraints and simulate SOC
        soc_mwh = 0.0
        
        for action in sorted_actions:
            # Calculate duration in hours
            duration_hours = (action.end_time - action.start_time).total_seconds() / 3600
            
            if action.is_charge:
                # Check charging power constraint
                if action.power_mw > max_charge_mw:
                    return {
                        'valid': False,
                        'error': 'Charging power exceeds maximum',
                        'action': action,
                        'max_allowed': max_charge_mw,
                        'attempted': action.power_mw
                    }
                
                # Calculate energy added to storage
                energy_in = action.power_mw * duration_hours * charge_efficiency
                
                # Check storage capacity constraint with a small tolerance (0.5%)
                # This allows for small floating-point precision issues
                tolerance = 0.005 * storage_capacity_mwh  # 0.5% tolerance
                if soc_mwh + energy_in > storage_capacity_mwh + tolerance:
                    return {
                        'valid': False,
                        'error': 'Storage capacity exceeded',
                        'action': action,
                        'current_soc': soc_mwh,
                        'energy_to_add': energy_in,
                        'capacity': storage_capacity_mwh,
                        'overflow': soc_mwh + energy_in - storage_capacity_mwh
                    }
                
                soc_mwh += energy_in
            else:
                # Check discharging power constraint
                if action.power_mw > max_discharge_mw:
                    return {
                        'valid': False,
                        'error': 'Discharging power exceeds maximum',
                        'action': action,
                        'max_allowed': max_discharge_mw,
                        'attempted': action.power_mw
                    }
                
                # Calculate energy removed from storage
                # For discharge: divide by power first, then divide by efficiency
                # (or equivalently, multiply energy out by efficiency)
                energy_out = action.power_mw * duration_hours / discharge_efficiency
                
                # Check if enough energy in storage
                if soc_mwh < energy_out:
                    return {
                        'valid': False,
                        'error': 'Insufficient energy in storage for discharge',
                        'action': action,
                        'current_soc': soc_mwh,
                        'energy_needed': energy_out,
                        'deficit': energy_out - soc_mwh
                    }
                
                soc_mwh -= energy_out
        
        return True


class ArbitrageAlgorithm(ABC):
    """
    Abstract base class for arbitrage algorithms.
    """
    @abstractmethod
    def generate_strategy(self, prices: PricingDataset, 
                         max_charge_mw: float, max_discharge_mw: float,
                         storage_capacity_mwh: float, 
                         charge_efficiency: float, discharge_efficiency: float) -> ArbitrageStrategy:
        """
        Generate an arbitrage strategy based on price predictions.
        
        Args:
            prices: PricingDataset with price predictions
            max_charge_mw: Maximum charging power in MW
            max_discharge_mw: Maximum discharging power in MW
            storage_capacity_mwh: Storage capacity in MWh
            rte: Round-trip efficiency (0-1)
            
        Returns:
            ArbitrageStrategy object
        """
        pass
