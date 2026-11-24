from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = "2019-01"
TRAIN_END = "2026-10"
TEST_START = "2026-11"
TEST_END = "2028-05"
FORECAST_HORIZON = 19

# Three 19-month rolling-origin backtest cutoffs (last valid month before 19-mo validation)
BACKTEST_CUTOFFS = ["2024-05", "2024-11", "2025-03"]

CANONICAL_REGIONS = [
    "North America",
    "Europe",
    "Asia Pacific",
    "India",
    "Latin America",
    "Middle East & Africa",
]

REGION_MAP = {
    "india": "India",
    "asia pacific": "Asia Pacific",
    "apac": "Asia Pacific",
    "asia-pacific": "Asia Pacific",
    "europe": "Europe",
    "eu": "Europe",
    "latin america": "Latin America",
    "latam": "Latin America",
    "latin-america": "Latin America",
    "north america": "North America",
    "n. america": "North America",
    "middle east & africa": "Middle East & Africa",
    "middle east and africa": "Middle East & Africa",
    "me&a": "Middle East & Africa",
    "mea": "Middle East & Africa",
}

CHANNEL_MAP = {
    "direct oem": "Direct OEM",
    "direct-oem": "Direct OEM",
    "oem direct": "Direct OEM",
    "tier-1 distributor": "Tier-1 Distributor",
    "tier 1 distributor": "Tier-1 Distributor",
    "t1 distributor": "Tier-1 Distributor",
    "aftermarket": "Aftermarket",
    "after-market": "Aftermarket",
    "export": "Export",
}

CANONICAL_CUSTOMERS = [
    "Vantara Auto",
    "Helios Auto Parts",
    "Orion Driveline",
    "Quantum Drive Systems",
    "Meridian Assembly",
    "Apex Tier One",
    "Solaris Automotive",
    "Nexon Motor Group",
    "Crestline Manufacturing",
    "Atlas Mideast Components",
    "Falcon Gulf Motors",
    "Kestrel Mobility",
]

FAMILY_TO_BU = {
    "Brake & Safety Systems": "Safety & Compliance",
    "Chassis & Structural": "Advanced Manufacturing",
    "Electrical Assemblies": "Electronics & Controls",
    "Hydraulic Systems": "Fluid Control Group",
    "Powertrain Components": "Power & Drive Systems",
    "Precision Machined Parts": "Advanced Manufacturing",
    "Thermal Management": "Fluid Control Group",
}

MACRO_COLS = [
    "ISM_Manufacturing_PMI",
    "Auto_Production_YoY_Pct",
    "Vehicle_Inventory_Days",
    "Raw_Material_Cost_Index",
    "New_Vehicle_Orders_Index",
    "Industrial_Util_Rate_Pct",
    "Fuel_Price_Index",
    "EV_Adoption_Index",
]