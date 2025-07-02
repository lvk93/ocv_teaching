# data_retrieval.py
"""
Retrieval scripts for fetching realistic OCV data from LiionDB.

This module provides functions to connect to the LiionDB PostgreSQL database,
query half-cell open-circuit voltage (OCV) curves for specified materials
or by material properties (e.g., nickel content), and parse them into
usable numpy arrays.

Requirements:
  - liiondb (clone and install per https://github.com/ndrewwang/liiondb)
  - pandas
  - sqlalchemy
  - psycopg2
"""
import pandas as pd
import numpy as np


def fetch_half_cell_ocv_by_material(material_name: str) -> dict:
    """
    Fetch all half-cell OCV curves for a given material name.

    Args:
        material_name: Exact name of the material in LiionDB (e.g., 'NMC811').

    Returns:
        Dict mapping data_id to a numpy array of shape (N, 2) with SOC and voltage pairs.
    """
    # Connect to LiionDB
    db_obj, _ = fn_db.liiondb()

    # Query raw_data, class, and binary function for half-cell OCV
    query = """
    SELECT data_id, raw_data, raw_data_class, function
    FROM data
    JOIN parameter ON parameter.parameter_id = data.parameter_id
    JOIN material ON material.material_id = data.material_id
    WHERE parameter.name = 'half cell ocv'
      AND material.name = %(material)s;
    """
    df = pd.read_sql(query, db_obj, params={'material': material_name})

    # Parse each raw_data entry
    ocv_curves = {}
    for _, row in df.iterrows():
        # read_data expects a DataFrame with one row
        single_row_df = row.to_frame().T
        arr = read_data(single_row_df)  # numpy array of [SOC, V]
        ocv_curves[int(row['data_id'])] = arr

    return ocv_curves


def fetch_half_cell_ocv_by_nickel(ni_min: float = 0.5) -> dict:
    """
    Fetch half-cell OCV curves for cathode materials with nickel content >= ni_min.

    Args:
        ni_min: Minimum Ni fraction in the cathode (0 to 1).

    Returns:
        Dict mapping (material_name, data_id) to numpy arrays of SOC-voltage.
    """
    db_obj, _ = liiondb()
    query = """
    SELECT data_id, material.name AS material, raw_data, raw_data_class, function
    FROM data
    JOIN parameter ON parameter.parameter_id = data.parameter_id
    JOIN material ON material.material_id = data.material_id
    WHERE parameter.name = 'half cell ocv'
      AND material.ni >= %(ni)s;
    """
    df = pd.read_sql(query, db_obj, params={'ni': ni_min})

    ocv_data = {}
    for _, row in df.iterrows():
        single_row_df = row.to_frame().T
        arr = read_data(single_row_df)
        key = (row['material'], int(row['data_id']))
        ocv_data[key] = arr

    return ocv_data


if __name__ == '__main__':
    # Example usage
    print("Fetching OCV curves for NMC811...")
    curves = fetch_half_cell_ocv_by_material('NMC811')
    for data_id, arr in curves.items():
        print(f"Data ID: {data_id}, shape: {arr.shape}")

    print("\nFetching OCV for Ni >= 0.6 cathodes...")
    ni_curves = fetch_half_cell_ocv_by_nickel(ni_min=0.6)
    for (mat, data_id), arr in ni_curves.items():
        print(f"Material: {mat}, Data ID: {data_id}, shape: {arr.shape}")
