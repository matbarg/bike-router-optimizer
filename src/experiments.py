# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1. Imports

# %%
import random
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import ipywidgets as widgets

from IPython.display import HTML, display
from geopy.distance import geodesic
from shapely.geometry import Point

# %% [markdown]
# # 2. Configuration

# %%
RANDOM_SEED = 42

OD_PAIRS_COUNT = 10
ROUTES_PER_OD = 50

THETA_MIN = 0.2
THETA_MAX = 2.0

BASE_SAMPLE_ID = -1

API_URL = "http://localhost:8080/api/route"

EXAMPLE_OD_ID = 10
EXAMPLE_SAMPLE_ID = BASE_SAMPLE_ID

OD_DISTANCE_GROUPS = [
    {
        "label": "1-3km",
        "min_km": 1,
        "max_km": 3,
        "share": 0.15,
    },
    {
        "label": "3-5km",
        "min_km": 3,
        "max_km": 5,
        "share": 0.20,
    },
    {
        "label": "5-10km",
        "min_km": 5,
        "max_km": 10,
        "share": 0.40,
    },
    {
        "label": ">10km",
        "min_km": 10,
        "max_km": 15,
        "share": 0.25,
    },
]

# %% [markdown]
# ## 2.1 Export Configuration

# %%
EXPORT_DIR = Path("thesis_exports")
EXPORT_DIR.mkdir(exist_ok=True)

PLOTS_DIR = EXPORT_DIR / "plots"
MAPS_DIR = EXPORT_DIR / "maps"
TABLES_DIR = EXPORT_DIR / "tables"

PLOTS_DIR.mkdir(exist_ok=True)
MAPS_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

ENABLE_EXPORTS = True

def export_plot(fig, filename, pdf=True):
    """Export a matplotlib figure to PDF."""
    if not ENABLE_EXPORTS:
        return
    
    plot_path = PLOTS_DIR / f"{filename}.pdf"
    fig.savefig(plot_path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Exported plot: {plot_path}")

def export_map(folium_map, filename):
    """Export a folium map to PNG using Selenium."""
    if not ENABLE_EXPORTS:
        return
    
    import tempfile
    import time
    
    html_path = MAPS_DIR / f"{filename}_temp.html"
    png_path = MAPS_DIR / f"{filename}.png"
    
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        
        folium_map.save(str(html_path))
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        driver.get(f"file://{html_path.absolute()}")
        time.sleep(3)
        
        driver.save_screenshot(str(png_path))
        driver.quit()
        
        html_path.unlink()
        print(f"Exported map (PNG): {png_path}")
        
    except ImportError as e:
        print(f"Selenium/webdriver-manager not installed: {e}")
        print(f"Install with: pip install selenium webdriver-manager")
        html_path.unlink()
    except Exception as e:
        print(f"Error exporting map to PNG: {e}")
        html_path.unlink()

def export_table(df, table_name, include_csv=True, include_latex=True):
    """Export a DataFrame to CSV and LaTeX formats."""
    if not ENABLE_EXPORTS:
        return
    
    if include_csv:
        csv_path = TABLES_DIR / f"{table_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Exported table (CSV): {csv_path}")
    
    if include_latex:
        latex_path = TABLES_DIR / f"{table_name}.tex"
        with open(latex_path, "w") as f:
            f.write(df.to_latex(index=False))
        print(f"Exported table (LaTeX): {latex_path}")

# %%
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# %% [markdown]
# # 3. Defining the Trips

# %% [markdown]
# ## 3.1 Loading the Map Boundaries

# %%
vienna = gpd.read_file("resources/vienna.json")
vienna_polygon = vienna.geometry.union_all()

minx, miny, maxx, maxy = vienna.total_bounds

print(f"West:  {minx}")
print(f"South: {miny}")
print(f"East:  {maxx}")
print(f"North: {maxy}")

vienna_center = [
    (miny + maxy) / 2,
    (minx + maxx) / 2,
]

# %%
vienna_map = folium.Map(
    location=vienna_center,
    zoom_start=11,
)

folium.GeoJson(vienna.geometry.to_json()).add_to(vienna_map)

vienna_map

# %%
export_map(vienna_map, "01_vienna_boundaries")
# ## 3.2 Trip Sampling

# %% jupyter={"source_hidden": true}
def sample_point_in_polygon(polygon):
    """Sample one random point inside a polygon."""
    minx, miny, maxx, maxy = polygon.bounds

    while True:
        point = Point(
            random.uniform(minx, maxx),
            random.uniform(miny, maxy),
        )

        if polygon.contains(point):
            return point


def distance_group_counts(distance_groups, total_count):
    """Convert distance group shares into integer OD pair counts."""
    groups_with_counts = []

    for group in distance_groups:
        exact_count = group["share"] * total_count
        count = int(np.floor(exact_count))

        groups_with_counts.append({
            **group,
            "exact_count": exact_count,
            "count": count,
            "remainder": exact_count - count,
        })

    assigned_count = sum(
        group["count"]
        for group in groups_with_counts
    )

    remaining_count = total_count - assigned_count

    groups_with_counts = sorted(
        groups_with_counts,
        key=lambda group: group["remainder"],
        reverse=True,
    )

    for i in range(remaining_count):
        groups_with_counts[i]["count"] += 1

    return groups_with_counts


def sample_od_pair_for_distance_group(
    polygon,
    distance_group,
    max_attempts=1000,
):
    """Sample one origin-destination pair within a fixed distance group."""
    for _ in range(max_attempts):
        origin_point = sample_point_in_polygon(polygon)

        origin_lat = origin_point.y
        origin_lon = origin_point.x

        distance_km = random.uniform(
            distance_group["min_km"],
            distance_group["max_km"],
        )

        bearing = random.uniform(0, 360)

        destination = geodesic(kilometers=distance_km).destination(
            (origin_lat, origin_lon),
            bearing,
        )

        destination_point = Point(
            destination.longitude,
            destination.latitude,
        )

        if polygon.contains(destination_point):
            return {
                "origin_point": origin_point,
                "destination_point": destination_point,
                "distance_group": distance_group["label"],
                "sampled_distance_km": distance_km,
            }

    raise ValueError(
        f"Could not sample OD pair for distance group {distance_group['label']}"
    )


def generate_od_pairs_stratified(
    polygon,
    count,
    distance_groups,
):
    """Generate OD pairs with exact counts per distance group."""
    od_pairs = []

    groups_with_counts = distance_group_counts(
        distance_groups=distance_groups,
        total_count=count,
    )

    for group in groups_with_counts:
        for _ in range(group["count"]):
            od_pair = sample_od_pair_for_distance_group(
                polygon=polygon,
                distance_group=group,
            )

            od_pairs.append(od_pair)

    random.shuffle(od_pairs)

    return od_pairs


def point_to_lat_lon(point):
    """Convert shapely Point with x=lon, y=lat to (lat, lon)."""
    return point.y, point.x


# %%
od_pairs = generate_od_pairs_stratified(
    polygon=vienna_polygon,
    count=OD_PAIRS_COUNT,
    distance_groups=OD_DISTANCE_GROUPS,
)

len(od_pairs)


# %%
od_pairs_df = pd.DataFrame([
    {
        "od_id": od_id,
        "distance_group": od_pair["distance_group"],
        "sampled_distance_km": od_pair["sampled_distance_km"],
    }
    for od_id, od_pair in enumerate(od_pairs)
])

od_pairs_df

# %%
export_table(od_pairs_df, "02_od_pairs")


# %%
od_distance_summary_df = (
    od_pairs_df["distance_group"]
    .value_counts()
    .rename("count")
    .reset_index()
    .rename(columns={"index": "distance_group"})
)

od_distance_summary_df["share"] = (
    od_distance_summary_df["count"]
    / len(od_pairs_df)
)

od_distance_summary_df

# %%
export_table(od_distance_summary_df, "03_od_distance_summary")


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 3.3 Showing the Trips on the Map

# %% jupyter={"source_hidden": true}
od_map = folium.Map(
    location=vienna_center,
    zoom_start=11,
)

folium.GeoJson(vienna.geometry.to_json()).add_to(od_map)

for od_id, od_pair in enumerate(od_pairs):
    origin_point = od_pair["origin_point"]
    destination_point = od_pair["destination_point"]

    origin_location = [origin_point.y, origin_point.x]
    destination_location = [destination_point.y, destination_point.x]

    midpoint_location = [
        (origin_point.y + destination_point.y) / 2,
        (origin_point.x + destination_point.x) / 2,
    ]

    popup_text = (
        f"OD {od_id}<br>"
        f"distance group: {od_pair['distance_group']}<br>"
        f"sampled distance: {od_pair['sampled_distance_km']:.2f} km"
    )

    folium.CircleMarker(
        location=origin_location,
        radius=3,
        color="green",
        fill=True,
        fill_opacity=0.8,
        popup=f"origin {od_id}<br>{popup_text}",
    ).add_to(od_map)

    folium.CircleMarker(
        location=destination_location,
        radius=3,
        color="red",
        fill=True,
        fill_opacity=0.8,
        popup=f"destination {od_id}<br>{popup_text}",
    ).add_to(od_map)

    folium.PolyLine(
        locations=[
            origin_location,
            destination_location,
        ],
        color="blue",
        weight=2,
        opacity=0.5,
        tooltip=popup_text,
    ).add_to(od_map)

    folium.Marker(
        location=midpoint_location,
        icon=folium.DivIcon(
            icon_size=(34, 22),
            icon_anchor=(17, 11),
            html=f"""
            <div style="
                width: 30px;
                height: 20px;
                line-height: 20px;
                font-size: 11px;
                font-weight: bold;
                color: black;
                background-color: white;
                border-radius: 4px;
                text-align: center;
            ">
                {od_id}
            </div>
            """
        ),
    ).add_to(od_map)

od_map

# %%
export_map(od_map, "04_od_pairs_map")


# %% [markdown]
# # 4. Experiment Execution
#
# For each OD pair, first the baseline route is requested.
# Then one route is requested for each sampled theta vector.

# %% [markdown]
# ## 4.1 Routing Parameters and Theta Sampling

# %%
def route_params(
        cycleway_lane=1.0,
        cycleway_track=1.0,
        road_class_cycleway=1.0,
        road_class_primary_secondary_trunk=1.0,
        road_class_residential=1.0,
        road_class_path=1.0,
        road_class_footway=1.0,
        surface_cobblestone_gravel_unpaved=1.0,
        incline_avg_above_four_percent=1.0,
        decline_avg_above_four_percent=1.0,
        no_car_access=1.0,
        bike_road_access_designated=1.0,
        bike_road_access_dismount_or_get_off_bike=1.0,
        max_speed_above_thirty=1.0,
):
    """Create a routing preference vector."""
    return {
        "cyclewayLane": cycleway_lane,
        "cyclewayTrack": cycleway_track,
        "roadClassCycleway": road_class_cycleway,
        "roadClassPrimarySecondaryTrunk": road_class_primary_secondary_trunk,
        "roadClassResidential": road_class_residential,
        "roadClassPath": road_class_path,
        "roadClassFootway": road_class_footway,
        "surfaceCobblestoneGravelUnpaved": surface_cobblestone_gravel_unpaved,
        "inclineAvgAboveFourPercent": incline_avg_above_four_percent,
        "declineAvgAboveFourPercent": decline_avg_above_four_percent,
        "noCarAccess": no_car_access,
        "bikeRoadAccessDesignated": bike_road_access_designated,
        "bikeRoadAccessDismountOrGetOffBike": bike_road_access_dismount_or_get_off_bike,
        "maxSpeedAboveThirty": max_speed_above_thirty,
    }


BASE_THETA = route_params()
PARAMETER_COLUMNS = list(BASE_THETA.keys())

def sample_theta(theta_min=THETA_MIN, theta_max=THETA_MAX):
    """Sample one random routing preference vector."""
    return route_params(
        cycleway_lane=random.uniform(theta_min, theta_max),
        cycleway_track=random.uniform(theta_min, theta_max),
        road_class_cycleway=random.uniform(theta_min, theta_max),
        road_class_primary_secondary_trunk=random.uniform(theta_min, theta_max),
        road_class_residential=random.uniform(theta_min, theta_max),
        road_class_path=random.uniform(theta_min, theta_max),
        road_class_footway=random.uniform(theta_min, theta_max),
        surface_cobblestone_gravel_unpaved=random.uniform(theta_min, theta_max),
        incline_avg_above_four_percent=random.uniform(theta_min, theta_max),
        decline_avg_above_four_percent=random.uniform(theta_min, theta_max),
        no_car_access=random.uniform(theta_min, theta_max),
        bike_road_access_designated=random.uniform(theta_min, theta_max),
        bike_road_access_dismount_or_get_off_bike=random.uniform(theta_min, theta_max),
        max_speed_above_thirty=random.uniform(theta_min, theta_max),
    )


def generate_theta_samples(count):
    """Generate theta vectors used for all OD pairs."""
    rows = []

    for sample_id in range(count):
        theta = sample_theta()

        rows.append({
            "sample_id": sample_id,
            **theta,
        })

    return pd.DataFrame(rows)


# %%
theta_samples_df = generate_theta_samples(ROUTES_PER_OD)

theta_samples_df.head()


# %% [markdown]
# ## 4.2 Helpers

# %%
def route_metric_row(route, od_id, sample_id, is_baseline, theta):
    """Create one row for route-level metrics."""
    properties = route["properties"]

    return {
        "od_id": od_id,
        "sample_id": sample_id,
        "is_baseline": is_baseline,
        "route_length_m": properties.get("distance"),
        "route_time_min": properties.get("time"),
        "route_ascend_m": properties.get("ascend"),
        "route_descend_m": properties.get("descend"),
        **theta,
    }


def route_features_long(route, od_id, sample_id, is_baseline):
    """Extract route features in long format.

    No artificial missing values are added here.
    """
    rows = []

    route_features = route["properties"]["routeFeatures"]

    for feature_group, values in route_features.items():
        for feature_value, share in values.items():
            rows.append({
                "od_id": od_id,
                "sample_id": sample_id,
                "is_baseline": is_baseline,
                "feature_group": feature_group,
                "feature_value": str(feature_value),
                "share": share,
            })

    return rows


def route_coordinates(route):
    """Convert GeoJSON route coordinates to folium-compatible coordinates."""
    return [
        (lat, lon)
        for lon, lat in route["geometry"]["coordinates"]
    ]


# %% [markdown]
# ## 4.3 API Request

# %%
def get_route(origin, destination, theta):
    """Request one route from the routing API."""
    payload = {
        "points": [
            {
                "lat": origin[0],
                "lon": origin[1],
            },
            {
                "lat": destination[0],
                "lon": destination[1],
            },
        ],
        "profile": "BASE",
        "mode": "CUSTOM",
        "withInstructions": False,
        "preferencesDto": theta,
    }

    response = requests.post(
        API_URL,
        json=payload,
    )

    response.raise_for_status()

    return response.json()


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 4.4 Route Generation

# %%
route_rows = []
feature_rows = []
route_records = []

for od_id, od_pair in enumerate(od_pairs):
    origin = point_to_lat_lon(od_pair["origin_point"])
    destination = point_to_lat_lon(od_pair["destination_point"])

    try:
        base_route = get_route(
            origin=origin,
            destination=destination,
            theta=BASE_THETA,
        )

        route_rows.append(
            route_metric_row(
                route=base_route,
                od_id=od_id,
                sample_id=BASE_SAMPLE_ID,
                is_baseline=True,
                theta=BASE_THETA,
            )
        )

        feature_rows.extend(
            route_features_long(
                route=base_route,
                od_id=od_id,
                sample_id=BASE_SAMPLE_ID,
                is_baseline=True,
            )
        )

        route_records.append({
            "od_id": od_id,
            "sample_id": BASE_SAMPLE_ID,
            "is_baseline": True,
            "route": base_route,
        })

        for _, theta_row in theta_samples_df.iterrows():
            sample_id = int(theta_row["sample_id"])

            theta = {
                parameter: theta_row[parameter]
                for parameter in PARAMETER_COLUMNS
            }

            route = get_route(
                origin=origin,
                destination=destination,
                theta=theta,
            )

            route_rows.append(
                route_metric_row(
                    route=route,
                    od_id=od_id,
                    sample_id=sample_id,
                    is_baseline=False,
                    theta=theta,
                )
            )

            feature_rows.extend(
                route_features_long(
                    route=route,
                    od_id=od_id,
                    sample_id=sample_id,
                    is_baseline=False,
                )
            )

            route_records.append({
                "od_id": od_id,
                "sample_id": sample_id,
                "is_baseline": False,
                "route": route,
            })

    except Exception as error:
        print(f"Failed OD {od_id}: {error}")

# %% [markdown]
# # 5. Result Tables

# %%
routes_df = pd.DataFrame(route_rows)
features_df = pd.DataFrame(feature_rows)
route_records_df = pd.DataFrame(route_records)

# %%
dataset_overview = pd.Series({
    "od_pairs": routes_df["od_id"].nunique(),
    "routes_total": len(routes_df),
    "baseline_routes": int(routes_df["is_baseline"].sum()),
    "sampled_routes": int((~routes_df["is_baseline"]).sum()),
    "theta_samples": len(theta_samples_df),
    "feature_groups": features_df["feature_group"].nunique(),
    "feature_values": (
        features_df[["feature_group", "feature_value"]]
        .drop_duplicates()
        .shape[0]
    ),
})

dataset_overview

# %%
export_table(dataset_overview.reset_index().rename(columns={0: "value", "index": "metric"}), "05_dataset_overview")


# %% [markdown]
# # 6. Evaluations

# %% [markdown]
# ## 6.1 Functions

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### 6.1.1 Trip Summary
#
# This section summarizes all generated routes for one OD pair.
#
# The baseline route is included in min, median, max, and range.
# It is also shown separately as its own column.

# %%
def od_metric_summary(routes_df, od_id):
    """Summarize simple route metrics for one OD pair.

    The baseline route is included in all statistics and also shown separately.
    """
    od_routes_df = routes_df[
        routes_df["od_id"] == od_id
        ].copy()

    baseline_row = od_routes_df[
        od_routes_df["is_baseline"]
    ].iloc[0]

    metric_columns = {
        "distance": "route_length_m",
        "time": "route_time_min",
        "ascend": "route_ascend_m",
        "descend": "route_descend_m",
    }

    rows = []

    for metric_name, column in metric_columns.items():
        values = od_routes_df[column]

        rows.append({
            "metric": metric_name,
            "baseline": baseline_row[column],
            "min": values.min(),
            "median": values.median(),
            "max": values.max(),
            "range": values.max() - values.min(),
        })

    return pd.DataFrame(rows)


def od_feature_variability_summary(features_df, routes_df, od_id, top_n=10):
    """Show the most variable feature values for one OD pair.

    The baseline route is included in min, median, max, and range.
    It is also shown separately as its own column.

    Missing feature values in a route are treated as 0 for this summary.
    """
    od_features_df = features_df[
        features_df["od_id"] == od_id
        ].copy()

    od_routes_df = routes_df[
        routes_df["od_id"] == od_id
        ].copy()

    sample_ids = od_routes_df["sample_id"].tolist()

    wide_df = (
        od_features_df
        .pivot_table(
            index="sample_id",
            columns=["feature_group", "feature_value"],
            values="share",
            aggfunc="sum",
            fill_value=0,
        )
    )

    wide_df = wide_df.reindex(
        sample_ids,
        fill_value=0,
    )

    baseline_sample_id = od_routes_df[
        od_routes_df["is_baseline"]
    ]["sample_id"].iloc[0]

    rows = []

    for feature_group, feature_value in wide_df.columns:
        values = wide_df[(feature_group, feature_value)]

        rows.append({
            "group": feature_group,
            "value": feature_value,
            "baseline": values.loc[baseline_sample_id],
            "min": values.min(),
            "median": values.median(),
            "max": values.max(),
            "range": values.max() - values.min(),
        })

    summary_df = pd.DataFrame(rows)

    summary_df = (
        summary_df
        .sort_values("range", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return summary_df


def plot_all_routes_for_od(
        route_records_df,
        od_id,
        route_opacity=0.10,
        route_weight=3,
        baseline_weight=3,
        highlight_sample_ids=None,
        show_baseline=True,
        width="900px",
        height="500px",
):
    """Plot all routes for one OD pair on one map.

    Sampled routes are drawn with low opacity so overlapping corridors become visible.

    Optionally, one or multiple sampled routes can be highlighted.
    The baseline route can also be shown or hidden.
    """
    od_routes_df = route_records_df[
        route_records_df["od_id"] == od_id
    ].copy()

    if od_routes_df.empty:
        raise ValueError(f"No routes found for OD {od_id}")

    if highlight_sample_ids is None:
        highlight_sample_ids = []
    elif isinstance(highlight_sample_ids, int):
        highlight_sample_ids = [highlight_sample_ids]
    else:
        highlight_sample_ids = list(highlight_sample_ids)

    baseline_row = od_routes_df[
        od_routes_df["is_baseline"]
    ]

    if baseline_row.empty:
        first_route = od_routes_df.iloc[0]["route"]
    else:
        first_route = baseline_row.iloc[0]["route"]

    first_coordinates = route_coordinates(first_route)

    route_map = folium.Map(
        location=first_coordinates[0],
        zoom_start=13,
        width=width,
        height=height,
    )

    sampled_routes_df = od_routes_df[
        ~od_routes_df["is_baseline"]
    ]

    # Draw sampled routes first, with low opacity.
    for _, row in sampled_routes_df.iterrows():
        sample_id = int(row["sample_id"])
        coordinates = route_coordinates(row["route"])

        # Skip highlighted routes here. They are drawn later on top.
        if sample_id in highlight_sample_ids:
            continue

        # Broad transparent layer: shows overlap density.
        folium.PolyLine(
            locations=coordinates,
            color="blue",
            weight=6,
            opacity=0.05,
            tooltip=f"sample {sample_id}",
        ).add_to(route_map)

        # Thin visible layer: keeps solitary routes recognizable.
        folium.PolyLine(
            locations=coordinates,
            color="blue",
            weight=3,
            opacity=0.22,
            tooltip=f"sample {sample_id}",
        ).add_to(route_map)

    # Optionally draw baseline route.
    if show_baseline and not baseline_row.empty:
        baseline_route = baseline_row.iloc[0]["route"]
        baseline_coordinates = route_coordinates(baseline_route)

        folium.PolyLine(
            locations=baseline_coordinates,
            color="red",
            weight=baseline_weight,
            opacity=0.7,
            tooltip="baseline",
        ).add_to(route_map)

    # Draw highlighted sampled routes last.
    highlight_colors = [
        "orange",
        "purple",
        "green",
        "black",
        "cadetblue",
        "darkred",
        "darkblue",
    ]

    for i, highlight_sample_id in enumerate(highlight_sample_ids):
        highlight_row = sampled_routes_df[
            sampled_routes_df["sample_id"] == highlight_sample_id
        ]

        if highlight_row.empty:
            raise ValueError(
                f"No sampled route found for OD {od_id}, sample {highlight_sample_id}"
            )

        highlight_route = highlight_row.iloc[0]["route"]
        highlight_coordinates = route_coordinates(highlight_route)

        folium.PolyLine(
            locations=highlight_coordinates,
            color=highlight_colors[i % len(highlight_colors)],
            weight=route_weight,
            opacity=0.9,
            tooltip=f"highlighted sample {highlight_sample_id}",
        ).add_to(route_map)

    folium.CircleMarker(
        location=first_coordinates[0],
        radius=4,
        color="green",
        fill=True,
        fill_opacity=0.9,
        tooltip="origin",
    ).add_to(route_map)

    folium.CircleMarker(
        location=first_coordinates[-1],
        radius=4,
        color="red",
        fill=True,
        fill_opacity=0.9,
        tooltip="destination",
    ).add_to(route_map)

    return route_map


def plot_od_feature_ranges(features_df, routes_df, od_id, top_n=20):
    """Plot feature share ranges for one OD pair.

    Shows the top_n most variable feature values.
    Each line shows min-max range.
    Each small dot shows one generated route sample.
    Baseline and median are highlighted separately.

    Missing feature values are treated as 0.
    """
    summary_df = od_feature_variability_summary(
        features_df=features_df,
        routes_df=routes_df,
        od_id=od_id,
        top_n=top_n,
    ).copy()

    summary_df["feature"] = (
            summary_df["group"].astype(str)
            + ": "
            + summary_df["value"].astype(str)
    )

    # Largest range at the top.
    summary_df = (
        summary_df
        .sort_values("range", ascending=True)
        .reset_index(drop=True)
    )

    od_routes_df = routes_df[
        routes_df["od_id"] == od_id
        ].copy()

    sample_ids = od_routes_df["sample_id"].tolist()

    od_features_df = features_df[
        features_df["od_id"] == od_id
        ].copy()

    wide_df = (
        od_features_df
        .pivot_table(
            index="sample_id",
            columns=["feature_group", "feature_value"],
            values="share",
            aggfunc="sum",
            fill_value=0,
        )
    )

    wide_df = wide_df.reindex(
        sample_ids,
        fill_value=0,
    )

    baseline_sample_id = od_routes_df[
        od_routes_df["is_baseline"]
    ]["sample_id"].iloc[0]

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.38)))

    for y_pos, row in summary_df.iterrows():
        feature_group = row["group"]
        feature_value = row["value"]

        values = wide_df[(feature_group, feature_value)]

        # Min-max range line.
        ax.hlines(
            y=y_pos,
            xmin=row["min"],
            xmax=row["max"],
            linewidth=5,
            alpha=0.3,
        )

        # Individual route sample dots.
        ax.scatter(
            values,
            [y_pos] * len(values),
            s=25,
            alpha=0.35,
            color='orange',
            label="route samples" if y_pos == 0 else None,
            zorder=3,
        )

        # Baseline point.
        ax.scatter(
            values.loc[baseline_sample_id],
            y_pos,
            s=30,
            color='green',
            marker="D",
            label="baseline" if y_pos == 0 else None,
            zorder=5,
        )

        # Median point.
        ax.scatter(
            row["median"],
            y_pos,
            s=45,
            color='red',
            marker="x",
            label="median" if y_pos == 0 else None,
            zorder=5,
        )

    ax.set_yticks(range(len(summary_df)))
    ax.set_yticklabels(summary_df["feature"])

    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of route")
    ax.set_ylabel("Feature value")
    ax.set_title(f"Feature Share Distributions, OD {od_id}")

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )

    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    export_plot(fig, f"plot_od_feature_ranges_od{od_id}")
    plt.show()

    return summary_df


def show_od_summary(od_id, top_n=20):
    """Display metric summary, feature variability summary, and route-overlap map for one OD pair."""
    display(HTML(f"<h3>OD {od_id}</h3>"))

    display(HTML("<h4>Route metric summary</h4>"))
    metric_table = od_metric_summary(
        routes_df=routes_df,
        od_id=od_id,
    ).round(2)
    export_table(metric_table, f"summary_od{od_id}_metrics")
    display(metric_table)

    display(HTML("<h4>Most variable route features</h4>"))
    feature_table = od_feature_variability_summary(
        features_df=features_df,
        routes_df=routes_df,
        od_id=od_id,
        top_n=top_n,
    ).round(3)
    export_table(feature_table, f"summary_od{od_id}_feature_variability")
    display(feature_table)

    display(HTML("<h4>Route feature ranges plot</h4>"))
    display(
        plot_od_feature_ranges(
            features_df=features_df,
            routes_df=routes_df,
            od_id=od_id,
            top_n=top_n,
        )
    )

    display(HTML("<h4>All generated routes</h4>"))
    route_map = plot_all_routes_for_od(
        route_records_df=route_records_df,
        od_id=od_id,
    )
    export_map(route_map, f"summary_od{od_id}_all_routes")
    display(route_map)

#show_od_summary(
#    od_id=16,
#    top_n=20,
#)


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### 6.1.2 Single Route Summary
#
# This section shows one generated route.
#
# It contains:
#
# 1. A table with simple route metrics.
# 2. A stacked percentage bar plot where each feature group is one horizontal bar.

# %%
def plot_single_route_on_map(route_records_df, od_id, sample_id, width="500px", height="350px"):
    """Plot one selected route on a map."""
    route_row = route_records_df[
        (route_records_df["od_id"] == od_id)
        & (route_records_df["sample_id"] == sample_id)
        ]

    route = route_row.iloc[0]["route"]

    coordinates = route_coordinates(route)

    route_map = folium.Map(
        location=coordinates[0],
        zoom_start=13,
        width=width,
        height=height,
    )

    folium.PolyLine(
        locations=coordinates,
        color="blue",
        weight=4,
        opacity=0.8,
        tooltip=f"OD {od_id}, sample {sample_id}",
    ).add_to(route_map)

    folium.CircleMarker(
        location=coordinates[0],
        radius=3,
        color="green",
        fill=True,
        fill_opacity=0.9,
        tooltip="origin",
    ).add_to(route_map)

    folium.CircleMarker(
        location=coordinates[-1],
        radius=3,
        color="red",
        fill=True,
        fill_opacity=0.9,
        tooltip="destination",
    ).add_to(route_map)

    return route_map


def single_route_metric_table(routes_df, od_id, sample_id):
    """Show simple route metrics for one route."""
    route_row = routes_df[
        (routes_df["od_id"] == od_id)
        & (routes_df["sample_id"] == sample_id)
        ].iloc[0]

    return pd.DataFrame({
        "metric": [
            "distance",
            "time",
            "ascend",
            "descend",
        ],
        "value": [
            route_row["route_length_m"],
            route_row["route_time_min"],
            route_row["route_ascend_m"],
            route_row["route_descend_m"],
        ],
        "unit": [
            "m",
            "min",
            "m",
            "m",
        ],
    })


def single_route_feature_plot_df(features_df, od_id, sample_id):
    """Prepare feature shares for one route.

    Missing shares are added only for the plot and not stored in features_df.
    """
    route_features_df = features_df[
        (features_df["od_id"] == od_id)
        & (features_df["sample_id"] == sample_id)
        ].copy()

    missing_rows = []

    group_sums = (
        route_features_df
        .groupby("feature_group")["share"]
        .sum()
    )

    for feature_group, group_sum in group_sums.items():
        missing_share = max(0, 1.0 - group_sum)

        if missing_share > 1e-9:
            missing_rows.append({
                "od_id": od_id,
                "sample_id": sample_id,
                "is_baseline": sample_id == BASE_SAMPLE_ID,
                "feature_group": feature_group,
                "feature_value": "missing",
                "share": missing_share,
            })

    if missing_rows:
        route_features_df = pd.concat(
            [
                route_features_df,
                pd.DataFrame(missing_rows),
            ],
            ignore_index=True,
        )

    return route_features_df


def text_color_for_background(color):
    """Return black or white depending on background brightness."""
    r, g, b, _ = color

    brightness = 0.299 * r + 0.587 * g + 0.114 * b

    if brightness < 0.5:
        return "white"

    return "black"


def plot_single_route_features(features_df, od_id, sample_id, inside_label_min_share=0.08):
    """Plot one route's grouped features as stacked horizontal percentage bars.

    Within each feature group, segments are ordered from largest share on the left
    to smallest share on the right.

    Large segments are labeled inside the bar.
    Small segments are labeled outside the bar with connector lines and anchor dots.

    Missing shares are added only for the plot and not stored in features_df.
    """
    plot_source_df = single_route_feature_plot_df(
        features_df=features_df,
        od_id=od_id,
        sample_id=sample_id,
    )

    feature_groups = sorted(plot_source_df["feature_group"].unique())

    feature_values = sorted(plot_source_df["feature_value"].unique())
    cmap = plt.get_cmap("tab20")

    color_by_value = {
        value: cmap(i % cmap.N)
        for i, value in enumerate(feature_values)
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for y_pos, feature_group in enumerate(feature_groups):
        group_df = (
            plot_source_df[
                plot_source_df["feature_group"] == feature_group
                ]
            .sort_values("share", ascending=False)
            .reset_index(drop=True)
        )

        left = 0.0
        outside_label_number = 0

        for _, row in group_df.iterrows():
            feature_value = row["feature_value"]
            share = row["share"]

            if share <= 0:
                continue

            color = color_by_value[feature_value]

            ax.barh(
                y=y_pos,
                width=share,
                left=left,
                height=0.75,
                color=color,
                edgecolor="white",
                linewidth=0.8,
            )

            center = left + share / 2
            label = f"{feature_value} ({share:.0%})"

            if share >= inside_label_min_share:
                ax.text(
                    x=center,
                    y=y_pos,
                    s=label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color_for_background(color),
                    fontweight="bold",
                )
            else:
                y_offset = (outside_label_number % 3 - 1) * 0.18
                label_x = 1.03 + 0.10 * (outside_label_number // 3)
                label_y = y_pos + y_offset

                # Anchor dot makes it clear which small segment the label belongs to.
                ax.scatter(
                    center,
                    y_pos,
                    s=18,
                    color=color,
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=5,
                )

                ax.annotate(
                    text=label,
                    xy=(center, y_pos),
                    xytext=(label_x, label_y),
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="black",
                    arrowprops={
                        "arrowstyle": "-",
                        "linewidth": 0.9,
                        "color": color,
                        "shrinkA": 0,
                        "shrinkB": 4,
                    },
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": color,
                        "linewidth": 0.7,
                        "alpha": 0.95,
                    },
                )

                outside_label_number += 1

            left += share

    ax.set_yticks(range(len(feature_groups)))
    ax.set_yticklabels(feature_groups)

    ax.set_xlim(0, 1.35)
    ax.set_xlabel("Share of route")
    ax.set_ylabel("Feature group")
    # ax.set_title(f"Route Feature Composition, OD {od_id}, Sample {sample_id}")

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}" if value <= 1 else "")
    )

    plt.tight_layout()
    export_plot(fig, f"plot_single_route_features_od{od_id}_sample{sample_id}")
    plt.show()

    return plot_source_df


def show_route_summary(od_id, sample_id):
    """Display metric summary, feature variability summary, and route-overlap map for one OD pair."""
    display(HTML(f"<h3>OD {od_id} Route {sample_id}</h3>"))

    display(HTML("<h4>Route metrics</h4>"))
    metric_table = single_route_metric_table(
        routes_df=routes_df,
        od_id=od_id,
        sample_id=sample_id,
    ).round(2)
    export_table(metric_table, f"route_od{od_id}_sample{sample_id}_metrics")
    display(metric_table)

    display(HTML("<h4>Route features</h4>"))
    plot_single_route_features(
        features_df=features_df,
        od_id=od_id,
        sample_id=sample_id,
        inside_label_min_share=0.10,
    )

    display(HTML("<h4>Route map</h4>"))
    route_map = plot_all_routes_for_od(
        route_records_df=route_records_df,
        od_id=od_id,
        highlight_sample_ids=[sample_id],
        show_baseline=True,
    )
    export_map(route_map, f"route_od{od_id}_sample{sample_id}")
    display(route_map)


# %%
# Select routes with a min or max feature share for an OD

def select_route_by_feature_share(
        features_df,
        routes_df,
        od_id,
        feature_group,
        feature_value,
        mode="max",
):
    """Select the route with the max or min share of a given feature.

    Missing feature values are treated as 0.
    """
    od_routes_df = routes_df[
        routes_df["od_id"] == od_id
        ].copy()

    sample_ids = od_routes_df["sample_id"].tolist()

    selected_feature_df = features_df[
        (features_df["od_id"] == od_id)
        & (features_df["feature_group"] == feature_group)
        & (features_df["feature_value"] == str(feature_value))
        ]

    feature_shares = (
        selected_feature_df
        .set_index("sample_id")["share"]
        .reindex(sample_ids, fill_value=0)
    )

    if mode == "max":
        selected_sample_id = feature_shares.idxmax()
    elif mode == "min":
        selected_sample_id = feature_shares.idxmin()
    else:
        raise ValueError("mode must be either 'max' or 'min'")

    selected_share = feature_shares.loc[selected_sample_id]

    return int(selected_sample_id), selected_share


# %%
#sample_id, share = select_route_by_feature_share(
#    features_df=features_df,
#    routes_df=routes_df,
#    od_id=EXAMPLE_OD_ID,
#    feature_group="average_slope",
#    feature_value="steep_down",
#    mode="max",
#)

#sample_id, share

#show_route_summary(
#    od_id=EXAMPLE_OD_ID,
#    sample_id=sample_id
#)

# %% [markdown]
# ## 6.2 Route Data Evaluation

# %% [markdown]
# ### 6.2.1 Mean Feature Ranges across Trips

# %% jupyter={"source_hidden": true}
def feature_variability_across_ods(features_df, routes_df):
    """Summarize feature variability across all OD pairs.

    For each OD pair and each feature value:
    - missing feature values are treated as 0
    - range = max share - min share

    Then ranges are aggregated across OD pairs.
    """
    rows = []

    for od_id, od_routes_df in routes_df.groupby("od_id"):
        od_features_df = features_df[
            features_df["od_id"] == od_id
            ].copy()

        sample_ids = od_routes_df["sample_id"].tolist()

        wide_df = (
            od_features_df
            .pivot_table(
                index="sample_id",
                columns=["feature_group", "feature_value"],
                values="share",
                aggfunc="sum",
                fill_value=0,
            )
        )

        wide_df = wide_df.reindex(
            sample_ids,
            fill_value=0,
        )

        baseline_sample_id = od_routes_df[
            od_routes_df["is_baseline"]
        ]["sample_id"].iloc[0]

        for feature_group, feature_value in wide_df.columns:
            values = wide_df[(feature_group, feature_value)]

            rows.append({
                "od_id": od_id,
                "feature_group": feature_group,
                "feature_value": feature_value,
                "baseline_share": values.loc[baseline_sample_id],
                "min": values.min(),
                "median": values.median(),
                "max": values.max(),
                "range": values.max() - values.min(),
            })

    feature_od_df = pd.DataFrame(rows)

    summary_df = (
        feature_od_df
        .groupby(["feature_group", "feature_value"])
        .agg(
            mean_range=("range", "mean"),
            median_range=("range", "median"),
            max_range=("range", "max"),
            mean_baseline_share=("baseline_share", "mean"),
        )
        .reset_index()
        .sort_values("mean_range", ascending=False)
        .reset_index(drop=True)
    )

    return summary_df, feature_od_df


# %%
feature_variability_summary_df, feature_variability_od_df = feature_variability_across_ods(
    features_df=features_df,
    routes_df=routes_df,
)

feature_variability_summary_df.head(50).round(3)


# %% [markdown]
# ### 6.2.2 Trips Ranked by Overall Mean Feture Range

# %% jupyter={"source_hidden": true}
def od_variability_ranking_by_feature_ranges(features_df, routes_df):
    """Rank OD pairs by their overall feature variability.

    For each OD pair:
    - each feature value gets a range: max share - min share
    - missing feature values are treated as 0
    - the OD score is the mean range across all feature values
    """
    rows = []

    for od_id, od_routes_df in routes_df.groupby("od_id"):
        od_features_df = features_df[
            features_df["od_id"] == od_id
            ].copy()

        sample_ids = od_routes_df["sample_id"].tolist()

        wide_df = (
            od_features_df
            .pivot_table(
                index="sample_id",
                columns=["feature_group", "feature_value"],
                values="share",
                aggfunc="sum",
                fill_value=0,
            )
        )

        wide_df = wide_df.reindex(
            sample_ids,
            fill_value=0,
        )

        feature_ranges = wide_df.max(axis=0) - wide_df.min(axis=0)

        rows.append({
            "od_id": od_id,
            "mean_feature_range": feature_ranges.mean(),
            "median_feature_range": feature_ranges.median(),
            "max_feature_range": feature_ranges.max(),
            "feature_count": len(feature_ranges),
        })

    ranking_df = (
        pd.DataFrame(rows)
        .sort_values("mean_feature_range", ascending=False)
        .reset_index(drop=True)
    )

    ranking_df["rank"] = ranking_df.index + 1

    return ranking_df[
        [
            "rank",
            "od_id",
            "mean_feature_range",
            "median_feature_range",
            "max_feature_range",
            "feature_count",
        ]
    ]


# %% jupyter={"source_hidden": true}
od_range_ranking_df = od_variability_ranking_by_feature_ranges(
    features_df=features_df,
    routes_df=routes_df,
)

od_range_ranking_df.round(3)

# %%
export_table(od_range_ranking_df, "06_od_variability_ranking")

# %% [markdown]
# ### 6.2.3 Visualization of Selected Trips

# %%
top_2_ods = od_range_ranking_df.head(2)

middle_start = len(od_range_ranking_df) // 2 - 1
middle_2_ods = od_range_ranking_df.iloc[middle_start:middle_start + 2]

bottom_2_ods = od_range_ranking_df.tail(2)

selected_ods_df = pd.concat(
    [
        top_2_ods.assign(selection_group="top"),
        middle_2_ods.assign(selection_group="middle"),
        bottom_2_ods.assign(selection_group="bottom"),
    ],
    ignore_index=True,
)

selected_ods_df[
    [
        "selection_group",
        "rank",
        "od_id",
        "mean_feature_range",
        "median_feature_range",
        "max_feature_range",
    ]
].round(3)

# %%
export_table(selected_ods_df[
    [
        "selection_group",
        "rank",
        "od_id",
        "mean_feature_range",
        "median_feature_range",
        "max_feature_range",
    ]
], "07_selected_ods_summary")

selected_od_ids = selected_ods_df["od_id"].tolist()

selected_od_ids

# %%
for _, row in selected_ods_df.iterrows():
    display(
        HTML(
            f"<h2>{row['selection_group'].capitalize()} variability — rank {row['rank']}/{OD_PAIRS_COUNT}</h2>"
        )
    )

    show_od_summary(
        od_id=int(row["od_id"]),
        top_n=20,
    )


# %% [markdown]
# ## 6.3 Persona Scoring

# %% [markdown]
# ### 6.3.1 Helpers

# %%
def min_max_normalize_by_od(df, column):
    """Min-max normalize one column within each OD pair."""
    min_values = df.groupby("od_id")[column].transform("min")
    max_values = df.groupby("od_id")[column].transform("max")

    denominator = max_values - min_values

    return pd.Series(
        np.where(
            denominator > 0,
            (df[column] - min_values) / denominator,
            0.5,
        ),
        index=df.index,
    )


def compute_persona_scores(
        df,
        persona_name,
        positive_features,
        negative_features,
):
    """Compute persona score from positive and negative feature columns.

    Positive features are preferred high.
    Negative features are preferred low.

    All features are normalized within each OD pair.
    """
    scored_df = df.copy()
    score_components = []

    persona_features = positive_features + negative_features

    for feature in persona_features:
        if feature not in scored_df.columns:
            print(f"Warning: missing feature column '{feature}', setting it to 0")
            scored_df[feature] = 0

    for feature in positive_features:
        score_column = f"{feature}_score"

        scored_df[score_column] = min_max_normalize_by_od(
            df=scored_df,
            column=feature,
        )

        score_components.append(score_column)

    for feature in negative_features:
        score_column = f"{feature}_score"

        scored_df[score_column] = (
                1
                - min_max_normalize_by_od(
            df=scored_df,
            column=feature,
        )
        )

        score_components.append(score_column)

    scored_df[f"{persona_name}_score"] = (
        scored_df[score_components]
        .mean(axis=1)
    )

    return scored_df


def best_persona_routes_per_od(scored_df, score_column, include_baseline=False):
    """Select the best-scoring route for each OD pair."""
    candidate_df = scored_df.copy()

    if not include_baseline:
        candidate_df = candidate_df[
            ~candidate_df["is_baseline"]
        ].copy()

    return (
        candidate_df
        .sort_values(["od_id", score_column], ascending=[True, False])
        .groupby("od_id")
        .head(1)
        .sort_values("od_id")
        .reset_index(drop=True)
    )


def add_baseline_comparison(best_routes_df, scored_df, persona_name):
    """Add baseline time, relative time change, and baseline persona score."""
    baseline_df = (
        scored_df[
            scored_df["is_baseline"]
        ][
            [
                "od_id",
                "route_time_min",
                f"{persona_name}_score",
            ]
        ]
        .rename(
            columns={
                "route_time_min": "base_route_time_min",
                f"{persona_name}_score": f"base_{persona_name}_score",
            }
        )
    )

    result_df = best_routes_df.merge(
        baseline_df,
        on="od_id",
        how="left",
    )

    result_df["relative_time_change_to_base"] = (
        result_df["route_time_min"]
        - result_df["base_route_time_min"]
    ) / result_df["base_route_time_min"]

    return result_df


# %%
def build_route_feature_matrix(features_df, routes_df, include_baseline=False):
    """Create one row per route with feature shares as columns.

    Missing feature values are treated as 0.
    """
    route_info_df = routes_df.copy()

    if not include_baseline:
        route_info_df = route_info_df[
            ~route_info_df["is_baseline"]
        ].copy()

    wide_features_df = (
        features_df
        .pivot_table(
            index=["od_id", "sample_id"],
            columns=["feature_group", "feature_value"],
            values="share",
            aggfunc="sum",
            fill_value=0,
        )
    )

    wide_features_df.columns = [
        f"{feature_group}:{feature_value}"
        for feature_group, feature_value in wide_features_df.columns
    ]

    wide_features_df = wide_features_df.reset_index()

    matrix_df = route_info_df.merge(
        wide_features_df,
        on=["od_id", "sample_id"],
        how="left",
    )

    feature_columns = [
        column
        for column in wide_features_df.columns
        if column not in ["od_id", "sample_id"]
    ]

    matrix_df[feature_columns] = matrix_df[feature_columns].fillna(0)

    return matrix_df, feature_columns


# %%
route_feature_matrix_df, feature_columns = build_route_feature_matrix(
    features_df=features_df,
    routes_df=routes_df,
    include_baseline=False,
)


# %%
persona_df, persona_feature_columns = build_route_feature_matrix(
    features_df=features_df,
    routes_df=routes_df,
    include_baseline=True,
)

#persona_df

# %%
def evaluate_persona(persona_df, persona_name, positive_features, negative_features):
    scored_df = compute_persona_scores(
        df=persona_df,
        persona_name=persona_name,
        positive_features=positive_features,
        negative_features=negative_features,
    )

    best_routes_per_od_df = best_persona_routes_per_od(
        scored_df=scored_df,
        score_column=f"{persona_name}_score",
        include_baseline=False,
    )

    best_routes_per_od_df = add_baseline_comparison(
        best_routes_df=best_routes_per_od_df,
        scored_df=scored_df,
        persona_name=persona_name,
    )

    best_score_per_od_table = best_routes_per_od_df[
        [
            "od_id",
            "sample_id",
            "route_time_min",
            "relative_time_change_to_base",
            "base_route_time_min",
            f"{persona_name}_score",
            f"base_{persona_name}_score",
            *PARAMETER_COLUMNS,
        ]
    ].copy()

    best_score_per_od_table = best_score_per_od_table.rename(
        columns={
            "route_time_min": "route_time",
            "base_route_time_min": "base_route_time",
        }
    )

    export_table(best_score_per_od_table, f"persona_{persona_name}_best_routes")

    display(
        best_score_per_od_table.style.format({
            "route_time": "{:.2f}",
            "relative_time_change_to_base": "{:+.0%}",
            "base_route_time": "{:.2f}",
            f"{persona_name}_score": "{:.3f}",
            f"base_{persona_name}_score": "{:.3f}",
            **{
                parameter: "{:.3f}"
                for parameter in PARAMETER_COLUMNS
            },
        })
    )

    best_parameter_summary_df = (
        best_routes_per_od_df[PARAMETER_COLUMNS]
        .agg(["mean", "median", "std"])
        .T
        .sort_values("mean", ascending=False)
    )

    export_table(best_parameter_summary_df.reset_index().rename(columns={"index": "parameter"}), f"persona_{persona_name}_parameter_summary")

    display(best_parameter_summary_df.round(3))


# %%
def evaluate_persona_time_adjusted(
    persona_df,
    persona_name,
    positive_features,
    negative_features,
    max_relative_time_increase=0.40,
):
    """Evaluate persona routes with a maximum allowed time increase.

    For each OD pair, the selected route is the highest-scoring sampled route
    whose travel time is not more than max_relative_time_increase above the
    baseline route time.

    Example:
    max_relative_time_increase=0.40 allows routes up to +40% longer than baseline.
    """
    scored_df = compute_persona_scores(
        df=persona_df,
        persona_name=persona_name,
        positive_features=positive_features,
        negative_features=negative_features,
    )

    candidate_routes_df = scored_df[
        ~scored_df["is_baseline"]
    ].copy()

    candidate_routes_df = add_baseline_comparison(
        best_routes_df=candidate_routes_df,
        scored_df=scored_df,
        persona_name=persona_name,
    )

    feasible_routes_df = candidate_routes_df[
        candidate_routes_df["relative_time_change_to_base"]
        <= max_relative_time_increase
    ].copy()

    best_routes_per_od_df = (
        feasible_routes_df
        .sort_values(
            ["od_id", f"{persona_name}_score"],
            ascending=[True, False],
        )
        .groupby("od_id")
        .head(1)
        .sort_values("od_id")
        .reset_index(drop=True)
    )

    missing_od_ids = sorted(
        set(scored_df["od_id"].unique())
        - set(best_routes_per_od_df["od_id"].unique())
    )

    if missing_od_ids:
        print(
            "Warning: no feasible sampled route found for these OD pairs "
            f"with max_relative_time_increase={max_relative_time_increase:.0%}: "
            f"{missing_od_ids}"
        )

    best_score_per_od_table = best_routes_per_od_df[
        [
            "od_id",
            "sample_id",
            "route_time_min",
            "relative_time_change_to_base",
            "base_route_time_min",
            f"{persona_name}_score",
            f"base_{persona_name}_score",
            *PARAMETER_COLUMNS,
        ]
    ].copy()

    best_score_per_od_table = best_score_per_od_table.rename(
        columns={
            "route_time_min": "route_time",
            "base_route_time_min": "base_route_time",
        }
    )

    export_table(best_score_per_od_table, f"persona_{persona_name}_time_adjusted_best_routes")

    display(
        best_score_per_od_table.style.format({
            "route_time": "{:.2f}",
            "relative_time_change_to_base": "{:+.0%}",
            "base_route_time": "{:.2f}",
            f"{persona_name}_score": "{:.3f}",
            f"base_{persona_name}_score": "{:.3f}",
            **{
                parameter: "{:.3f}"
                for parameter in PARAMETER_COLUMNS
            },
        })
    )

    best_parameter_summary_df = (
        best_routes_per_od_df[PARAMETER_COLUMNS]
        .agg(["mean", "median", "std"])
        .T
        .sort_values("mean", ascending=False)
    )

    export_table(best_parameter_summary_df.reset_index().rename(columns={"index": "parameter"}), f"persona_{persona_name}_time_adjusted_parameter_summary")

    display(best_parameter_summary_df.round(3))

    return best_routes_per_od_df

# %% [markdown]
# ### 6.3.2 Safe


# %%
SAFE_POSITIVE_FEATURES = [
    "cycleway:lane",
    "cycleway:track",
    "road_class:cycleway",
    "road_class:residential",
]

SAFE_NEGATIVE_FEATURES = [
    "cycleway:no",
    "cycleway:shared_lane",
    "road_class:primary",
    "road_class:secondary",
    "max_speed:50.0",
]

evaluate_persona(
    persona_df=persona_df,
    persona_name="safe",
    positive_features=SAFE_POSITIVE_FEATURES,
    negative_features=SAFE_NEGATIVE_FEATURES
)


# %%
best_safe_routes_per_od_df = evaluate_persona_time_adjusted(
    persona_df=persona_df,
    persona_name="safe",
    positive_features=SAFE_POSITIVE_FEATURES,
    negative_features=SAFE_NEGATIVE_FEATURES,
    max_relative_time_increase=0.50,
)

# %% [markdown]
# ### 6.3.3 Comfort


# %%
COMFORT_POSITIVE_FEATURES = [
    "average_slope:moderate_down",
]

COMFORT_NEGATIVE_FEATURES = [
    "average_slope:moderate_up",
    "average_slope:steep_up",
    "surface:gravel",
    "surface:gravel",
]

evaluate_persona(
    persona_df=persona_df,
    persona_name="comfort",
    positive_features=COMFORT_POSITIVE_FEATURES,
    negative_features=COMFORT_NEGATIVE_FEATURES
)


# %%
best_comfort_routes_per_od_df = evaluate_persona_time_adjusted(
    persona_df=persona_df,
    persona_name="comfort",
    positive_features=COMFORT_POSITIVE_FEATURES,
    negative_features=COMFORT_NEGATIVE_FEATURES,
    max_relative_time_increase=0.50,
)

# %% [markdown]
# ### 6.3.4 Offroad


# %%
OFFROAD_POSITIVE_FEATURES = [
    "surface:ground",
    "surface:gravel",
    "road_class:path",
]

OFFROAD_NEGATIVE_FEATURES = [
    "surface:asphalt",
]

evaluate_persona(
    persona_df=persona_df,
    persona_name="offroad",
    positive_features=OFFROAD_POSITIVE_FEATURES,
    negative_features=OFFROAD_NEGATIVE_FEATURES
)

# %%
best_offroad_routes_per_od_df = evaluate_persona_time_adjusted(
    persona_df=persona_df,
    persona_name="offroad",
    positive_features=OFFROAD_POSITIVE_FEATURES,
    negative_features=OFFROAD_NEGATIVE_FEATURES,
    max_relative_time_increase=0.50,
)

# %% [markdown]
# ### 6.3.5 Show Persona Routes Across Selected Trips


# %%
def selected_persona_sample_ids_for_od(od_id, persona_best_route_dfs):
    """Return selected sample IDs for one OD from multiple persona result dataframes.

    persona_best_route_dfs should map persona names to their best-routes-per-OD dataframe.
    """
    selected_sample_ids = {}

    for persona_name, best_routes_df in persona_best_route_dfs.items():
        selected_row = best_routes_df[
            best_routes_df["od_id"] == od_id
        ]

        if selected_row.empty:
            selected_sample_ids[persona_name] = None
        else:
            selected_sample_ids[persona_name] = int(
                selected_row["sample_id"].iloc[0]
            )

    return selected_sample_ids


# %%
PERSONA_BEST_ROUTE_DFS = {
    "safe": best_safe_routes_per_od_df,
    "comfort": best_comfort_routes_per_od_df,
    "offroad": best_offroad_routes_per_od_df,
}

for od_id in selected_od_ids:
    selected_sample_ids_by_persona = selected_persona_sample_ids_for_od(
        od_id=od_id,
        persona_best_route_dfs=PERSONA_BEST_ROUTE_DFS,
    )

    highlight_sample_ids = [
        sample_id
        for sample_id in selected_sample_ids_by_persona.values()
        if sample_id is not None
    ]

    display(HTML(f"<h3>OD {od_id}</h3>"))
    display(selected_sample_ids_by_persona)

    display(
        plot_all_routes_for_od(
            route_records_df=route_records_df,
            od_id=od_id,
            highlight_sample_ids=highlight_sample_ids,
            route_weight=5,
            show_baseline=False,
        )
    )
