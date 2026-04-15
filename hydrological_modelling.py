

#pip install whitebox

"""# Reading the DTM"""

import matplotlib.pyplot as plt
import rasterio
import numpy as np

# Load raster
with rasterio.open("raster_5m.tif") as src:
    dem = src.read(1).astype("float32")
    nodata = src.nodata

# Handle NoData
if nodata is not None:
    dem[dem == nodata] = np.nan

# ==========================================================
# 1️⃣ PLOT: ORIGINAL DEM (NO HILLSHADE)
# ==========================================================
plt.figure(figsize=(10, 8))

plt.imshow(
    dem,
    cmap="terrain",
    vmin=np.nanpercentile(dem, 2),
    vmax=np.nanpercentile(dem, 98)
)

plt.colorbar(label="Elevation (m)")
plt.title("Original DTM (Elevation)")
plt.axis("off")
plt.tight_layout()
plt.show()

#low_threshold = 200  # meters
#low_lying = dem < low_threshold

#low_threshold = np.nanpercentile(dem, 25)  # bottom 15% of elevations
#low_lying = dem < low_threshold

#plt.figure(figsize=(10, 8))

#plt.imshow(dem, cmap="terrain", alpha=0.6)        # DEM base
#plt.imshow(low_lying, cmap="Blues", alpha=0.5)    # Low-lying zones

#plt.title("Low-Lying Areas")
#plt.axis("off")
#plt.colorbar(label="Elevation (m)")
#plt.show()

"""# Install Whitebox Tools"""

import os

from whitebox.whitebox_tools import WhiteboxTools


current_dir = os.getcwd()

wbt = WhiteboxTools()
wbt.set_working_dir(current_dir)
wbt.verbose = True

import rasterio
from rasterio.crs import CRS

with rasterio.open("raster_5m.tif") as src:
    meta = src.meta.copy()
    data = src.read(1)

meta.update({"crs": CRS.from_epsg(32644)})

with rasterio.open("raster_georef.tif", "w", **meta) as dst:
    dst.write(data, 1)

"""# Fill Depressions"""

wbt.fill_depressions(
    dem="raster_georef.tif",
    output="dtm_filled.tif"
)

# Load raster
with rasterio.open("dtm_filled.tif") as src:
    dem = src.read(1).astype("float32")
    nodata = src.nodata

# Handle NoData
if nodata is not None:
    dem[dem == nodata] = np.nan

# ==========================================================
# 1️⃣ PLOT: ORIGINAL DEM (NO HILLSHADE)
# ==========================================================
plt.figure(figsize=(10, 8))

plt.imshow(
    dem,
    cmap="terrain",
    vmin=np.nanpercentile(dem, 2),
    vmax=np.nanpercentile(dem, 98)
)

plt.colorbar(label="Elevation (m)")
plt.title("Filled DTM (Elevation)")
plt.axis("off")
plt.tight_layout()
plt.show()

"""# Flow Direction"""

# ==========================================================
# FLOW DIRECTION
# ==========================================================
wbt.d8_pointer(
    dem="dtm_filled.tif",
    output="flow_dir.tif"
)

# ==========================================================
# LOAD FLOW DIRECTION
# ==========================================================
import rasterio
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open("flow_dir.tif") as src:
    flow_dir = src.read(1).astype("float32")
    nodata = src.nodata

if nodata is not None:
    flow_dir[flow_dir == nodata] = np.nan

# ==========================================================
# PLOT FLOW DIRECTION
# ==========================================================
plt.figure(figsize=(10, 8))

plt.imshow(flow_dir, cmap="viridis")

plt.colorbar(label="Flow Direction (D8)")
plt.title("Flow Direction (D8 Pointer)")
plt.axis("off")
plt.tight_layout()
plt.show()

"""# Flow accumulation"""

# ==========================================================
# FLOW ACCUMULATION
# ==========================================================
wbt.d8_flow_accumulation(
    i="dtm_filled.tif",
    output="flow_acc.tif",
    out_type="cells"
)

# ==========================================================
# LOAD FLOW ACCUMULATION
# ==========================================================
with rasterio.open("flow_acc.tif") as src:
    flow = src.read(1).astype("float32")
    nodata = src.nodata

if nodata is not None:
    flow[flow == nodata] = np.nan

flow[flow <= 0] = np.nan

# ==========================================================
# PLOT FLOW ACCUMULATION
# ==========================================================
flow_log = np.log1p(flow)   # VERY IMPORTANT

plt.figure(figsize=(10, 8))

plt.imshow(flow_log, cmap="Blues")

plt.colorbar(label="Log Flow Accumulation")
plt.title("Flow Accumulation")
plt.axis("off")
plt.tight_layout()
plt.show()

"""# Slope"""

wbt.slope(
    dem="dtm_filled.tif",
    output="slope.tif",
    units="degrees"
)

with rasterio.open("slope.tif") as src:
    slope = src.read(1).astype("float32")
    nodata = src.nodata

if nodata is not None:
    slope[slope == nodata] = np.nan

plt.figure(figsize=(10,8))
plt.imshow(slope, cmap="inferno")
plt.colorbar(label="Slope (degrees)")
plt.title("Slope Map")
plt.axis("off")
plt.show()

"""# TWI (Topographic Wetness Index)"""

# ==========================================================
# TOPOGRAPHIC WETNESS INDEX (TWI)
# ==========================================================
wbt.wetness_index(
    sca="flow_acc.tif",   # specific catchment area
    slope="slope.tif",
    output="twi.tif"
)


with rasterio.open("twi.tif") as src:
    twi = src.read(1).astype("float32")
    nodata = src.nodata

if nodata is not None:
    twi[twi == nodata] = np.nan

plt.figure(figsize=(10,8))

plt.imshow(twi, cmap="YlGnBu")

plt.colorbar(label="TWI")
plt.title("Topographic Wetness Index (TWI)")
plt.axis("off")
plt.show()

"""# HAND (Height Above Nearest Drainage)"""

wbt.extract_streams(
    flow_accum="flow_acc.tif",
    output="streams.tif",
    threshold=1000
)

wbt.elevation_above_stream(
    dem="dtm_filled.tif",
    streams="streams.tif",
    output="hand.tif"
)

import rasterio
import numpy as np

# ==========================================================
# LOAD HAND
# ==========================================================
with rasterio.open("hand.tif") as src:
    hand = src.read(1).astype("float32")
    nodata = src.nodata

# Handle NoData
if nodata is not None:
    hand[hand == nodata] = np.nan

# ==========================================================
# PLOT: HAND (RAW)
# ==========================================================
plt.figure(figsize=(10, 8))

plt.imshow(
    hand,
    cmap="viridis",
    vmin=np.nanpercentile(hand, 2),
    vmax=np.nanpercentile(hand, 98)
)

plt.colorbar(label="Height Above Drainage (m)")
plt.title("HAND (Elevation Above Nearest Drainage)")
plt.axis("off")

plt.tight_layout()
plt.show()

"""# Load all the layers"""

import rasterio
import numpy as np

def load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr

dem   = load_raster("dtm_filled.tif")
slope = load_raster("slope.tif")
twi   = load_raster("twi.tif")
hand  = load_raster("hand.tif")

"""# Normalise them"""

def normalize(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

twi_n   = normalize(twi)
slope_n = normalize(slope)
elev_n  = normalize(dem)
hand_n  = normalize(hand)

# Invert because low are bad

slope_n = 1 - slope_n
elev_n  = 1 - elev_n
hand_n  = 1 - hand_n

"""# Weighted Model"""

#index = (
    #0.2 * twi_n +
    #0.2 * slope_n +
    #0.2 * elev_n +
    #0.2 * hand_n
#)

index = (twi_n + slope_n + elev_n + hand_n) / 4
hotspots = index > np.nanpercentile(index, 70)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))

plt.imshow(index, cmap="RdYlBu_r")
plt.colorbar(label="Waterlogging Index")

plt.title("Waterlogging Susceptibility Map")
plt.axis("off")
plt.show()

plt.figure(figsize=(10,8))

plt.imshow(index, cmap="terrain", alpha=0.7)
plt.imshow(hotspots, cmap="Reds", alpha=0.6)

plt.title("Waterlogging Hotspots")
plt.axis("off")
plt.show()

"""# Natural drainage network"""

wbt.extract_streams(
    flow_accum="flow_acc.tif",
    output="streams.tif",
    threshold=10
)

wbt.raster_streams_to_vector(
    streams="streams.tif",
    d8_pntr="flow_dir.tif",
    output="streams.shp"
)

import rasterio
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# LOAD STREAMS
# ==========================================================
with rasterio.open("streams.tif") as src:
    streams = src.read(1)
    nodata = src.nodata

# Handle NoData
if nodata is not None:
    streams = np.where(streams == nodata, np.nan, streams)

# Convert to binary (important)
streams_bin = streams > 0

# ==========================================================
# PLOT STREAMS
# ==========================================================
plt.figure(figsize=(10, 8))

plt.imshow(streams_bin, cmap="Blues")

plt.title("Extracted Stream Network")
plt.axis("off")

plt.tight_layout()
plt.show()

"""# Overlay streams on hotspots"""

# ==========================================================
# OVERLAY: STREAMS + HOTSPOTS
# ==========================================================
plt.figure(figsize=(10, 8))

# Base: DEM (optional but useful)
plt.imshow(dem, cmap="terrain", alpha=0.6)

# Streams
plt.imshow(streams_bin, cmap="Blues", alpha=0.4)

# Hotspots
plt.imshow(hotspots, cmap="Reds", alpha=0.6)

plt.title("Streams vs Waterlogging Hotspots")
plt.axis("off")

plt.tight_layout()
plt.show()

"""# Find Unconnected hotspots"""

# ==========================================================
# FIND HOTSPOTS NOT CONNECTED TO STREAMS
# ==========================================================
unconnected = hotspots & (~streams_bin)

plt.figure(figsize=(10,8))

plt.imshow(dem, cmap="terrain", alpha=0.6)
plt.imshow(unconnected, cmap="Reds", alpha=0.8)

plt.title("Unconnected Waterlogging Hotspots")
plt.axis("off")

plt.show()

"""# Workflow for alternate drainage network"""

# ==========================================================
# NORMALIZE FUNCTION
# ==========================================================
def normalize(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

# Normalize inputs
slope_n = normalize(slope)
hand_n  = normalize(hand)
elev_n  = normalize(dem)

# Invert where needed (low = better)
slope_n = 1 - slope_n
hand_n  = 1 - hand_n
elev_n  = 1 - elev_n

# ==========================================================
# COST FUNCTION
# ==========================================================
cost = (
    0.5 * slope_n +
    0.3 * hand_n +
    0.2 * elev_n
)

# Save cost raster
with rasterio.open(
    "cost.tif",
    "w",
    driver="GTiff",
    height=cost.shape[0],
    width=cost.shape[1],
    count=1,
    dtype="float32",
    crs=src.crs,
    transform=src.transform,
) as dst:
    dst.write(cost, 1)

# ==========================================================
# SAVE UNCONNECTED HOTSPOTS
# ==========================================================
unconnected = (hotspots & (~streams_bin)).astype("int32")

with rasterio.open(
    "targets.tif",
    "w",
    driver="GTiff",
    height=unconnected.shape[0],
    width=unconnected.shape[1],
    count=1,
    dtype="int32",
    crs=src.crs,
    transform=src.transform,
) as dst:
    dst.write(unconnected, 1)

wbt.cost_distance(
    source="streams.tif",
    cost="cost.tif",
    out_accum="cost_dist.tif",
    out_backlink="backlink.tif"
)

# ==========================================================
# LEAST-COST DRAINAGE PATHS (FIXED)
# ==========================================================
wbt.cost_pathway(
    destination="targets.tif",
    backlink="backlink.tif",
    output="drain_paths.tif"
)

import rasterio
import numpy as np

# ==========================================================
# LOAD DRAIN PATHS
# ==========================================================
with rasterio.open("drain_paths.tif") as src:
    drains = src.read(1)
    nodata = src.nodata

if nodata is not None:
    drains = np.where(drains == nodata, np.nan, drains)

# Convert to binary
drains_bin = drains > 0

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

plt.imshow(dem, cmap="terrain", alpha=0.5)        # DEM
plt.imshow(streams_bin, cmap="Blues", alpha=0.6)  # Existing streams
plt.imshow(drains_bin, cmap="Greens", alpha=0.8)  # Proposed drains
plt.imshow(unconnected, cmap="Reds", alpha=0.4)   # Unconnected hotspots

plt.title("Proposed Alternate Drainage Network")
plt.axis("off")
plt.tight_layout()
plt.show()

