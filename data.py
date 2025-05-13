import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
import pytz
from arbitrage_types import PricingDataset


def process_lmp_data():
    """
    Process 5-minute LMP data from CAISO node TWILGHTL_7_N001 to hourly averages.
    - Filter for LMP_PRC only
    - Average duplicate datetimes
    - Compress 5-minute LMPs into hourly averages
    - Combine all monthly CSVs
    - Ensure all hours in 2024 exist, interpolate if needed
    - Output to data/2024_all_TWILGHTL_7_N001.csv
    """
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Get all CSV files in the data/node directory
    csv_files = glob.glob('data/node/*.csv')
    
    if not csv_files:
        print("No CSV files found in data/node directory")
        return
    
    # Process each CSV file
    all_hourly_data = []
    
    for file_path in csv_files:
        print(f"Processing {os.path.basename(file_path)}...")
        
        # Read the CSV file
        try:
            # Read in chunks to handle large files
            chunk_size = 100000
            chunks = pd.read_csv(file_path, chunksize=chunk_size)
            df_chunks = []
            
            for chunk in chunks:
                # Filter for LMP_PRC only
                chunk_filtered = chunk[chunk['XML_DATA_ITEM'] == 'LMP_PRC']
                df_chunks.append(chunk_filtered)
            
            df = pd.concat(df_chunks)
            
            if df.empty:
                print(f"No LMP_PRC data found in {os.path.basename(file_path)}")
                continue
                
        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")
            continue
        
        # Convert timestamps to datetime
        df['DATETIME'] = pd.to_datetime(df['INTERVALSTARTTIME_GMT'])
        
        # Convert to Pacific time (CAISO's local time)
        df['DATETIME'] = df['DATETIME'].dt.tz_convert('US/Pacific')
        
        # Average duplicate datetimes
        df = df.groupby('DATETIME').agg({'VALUE': 'mean'}).reset_index()
        
        # Extract hour from datetime - handle DST ambiguity by always using the first occurrence
        # First, extract the date and hour components separately
        df['DATE'] = df['DATETIME'].dt.date
        df['HOUR_OF_DAY'] = df['DATETIME'].dt.hour
        
        # Create a new datetime from the components (this avoids DST issues)
        df['HOUR'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + 
                                   df['HOUR_OF_DAY'].astype(str) + ':00:00').dt.tz_localize('US/Pacific', ambiguous='NaT')
        
        # Drop rows with NaT (ambiguous times that couldn't be resolved)
        df = df.dropna(subset=['HOUR'])
        
        # Drop temporary columns
        df = df.drop(['DATE', 'HOUR_OF_DAY'], axis=1)
        
        # Group by hour and calculate average LMP
        hourly_df = df.groupby('HOUR').agg(
            avg_lmp=('VALUE', 'mean'),
            count=('VALUE', 'count')
        ).reset_index()
        
        all_hourly_data.append(hourly_df)
    
    # Combine all monthly data
    if not all_hourly_data:
        print("No data processed from any CSV files")
        return
        
    combined_df = pd.concat(all_hourly_data)
    
    # Average duplicate hours
    combined_df = combined_df.groupby('HOUR').agg(
        avg_lmp=('avg_lmp', 'mean'),
        count=('count', 'sum')
    ).reset_index()
    
    # Create a complete datetime range for 2024
    pacific_tz = pytz.timezone('US/Pacific')
    start_date = pacific_tz.localize(datetime(2024, 1, 1))
    end_date = pacific_tz.localize(datetime(2024, 12, 31, 23))
    
    # Generate all hours in 2024 (leap year, so 8784 hours)
    all_hours = pd.date_range(start=start_date, end=end_date, freq='h')
    complete_df = pd.DataFrame({'HOUR': all_hours})
    
    # Merge with the processed data
    final_df = pd.merge(complete_df, combined_df, on='HOUR', how='left')
    
    # Check if we have the correct start and end dates
    if final_df['HOUR'].min() != start_date:
        print(f"ERROR: Start date is not January 1, 2024. Found: {final_df['HOUR'].min()}")
    
    if final_df['HOUR'].max() != end_date:
        print(f"ERROR: End date is not December 31, 2024 11pm. Found: {final_df['HOUR'].max()}")
    
    # Add a column to track if data was interpolated
    final_df['is_interpolated'] = final_df['avg_lmp'].isna()
    
    # Interpolate missing values
    final_df['avg_lmp'] = final_df['avg_lmp'].interpolate(method='linear')
    
    # Fill any remaining NaN values (at the beginning or end) with the nearest valid value
    # Using ffill() and bfill() instead of fillna(method='...') to avoid deprecation warning
    final_df['avg_lmp'] = final_df['avg_lmp'].bfill().ffill()
    
    # Rename for clarity and keep the interpolation flag
    final_df = final_df[['HOUR', 'avg_lmp', 'is_interpolated']].rename(columns={'avg_lmp': 'LMP'})
    
    # Save to CSV
    output_path = 'data/2024_all_TWILGHTL_7_N001.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(f"Total hours: {len(final_df)}")
    
    # Report on missing data that was interpolated
    missing_count = combined_df['HOUR'].nunique()
    total_hours = len(all_hours)
    print(f"Hours with data: {missing_count} out of {total_hours} ({missing_count/total_hours*100:.2f}%)")
    print(f"Hours interpolated: {total_hours - missing_count}")
    
    # Check if the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    return final_df


def load_pricing_dataset(file_path=None):
    """
    Load the processed LMP data as a PricingDataset object.
    
    Args:
        file_path: Path to the CSV file with processed data. If None, uses default path.
        
    Returns:
        PricingDataset object
    """
    if file_path is None:
        file_path = 'data/2024_all_TWILGHTL_7_N001.csv'
    
    if not os.path.exists(file_path):
        print(f"Data file {file_path} not found. Processing raw data...")
        process_lmp_data()
    
    return PricingDataset.from_csv(file_path)


if __name__ == "__main__":
    process_lmp_data()
