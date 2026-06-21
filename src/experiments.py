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
#

# %%
import random
import numpy as np
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from geopy.distance import geodesic
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# %% [markdown]
# # 2. Configuration
#

# %%
OD_PAIRS_COUNT = 20
ROUTES_PER_OD = 200

THETA_MIN = 0.2
THETA_MAX = 2.0

N_CLUSTERS = 4
EXAMPLE_OD = 3

API_URL = "http://localhost:8080/api/route"

EXAMPLE_START = (48.18051, 16.33375)
EXAMPLE_END = (48.23336, 16.37512)


# %% [markdown]
# # 3. Load Vienna Boundary
#

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

vienna_map = folium.Map(
    location=vienna_center,
    zoom_start=11,
)

folium.GeoJson(vienna.geometry.to_json()).add_to(vienna_map)

vienna_map


# %% [markdown]
# # 4. Origin-Destination Pair Sampling
#

# %%
def sample_point_in_polygon(polygon):
    """Sample one random Point inside a polygon by rejection sampling."""
    minx, miny, maxx, maxy = polygon.bounds

    while True:
        point = Point(
            random.uniform(minx, maxx),
            random.uniform(miny, maxy),
        )

        if polygon.contains(point):
            return point


def sample_od_pair(polygon, min_distance_km=2, max_distance_km=10):
    """Sample one origin-destination pair fully contained in the polygon."""
    origin_point = sample_point_in_polygon(polygon)

    origin_lat = origin_point.y
    origin_lon = origin_point.x

    distance_km = random.uniform(min_distance_km, max_distance_km)
    bearing = random.uniform(0, 360)

    destination = geodesic(kilometers=distance_km).destination(
        (origin_lat, origin_lon),
        bearing,
    )

    destination_point = Point(destination.longitude, destination.latitude)

    if polygon.contains(destination_point):
        return origin_point, destination_point

    return sample_od_pair(
        polygon,
        min_distance_km=min_distance_km,
        max_distance_km=max_distance_km,
    )


def generate_od_pairs(polygon, count):
    """Generate a list of origin-destination point pairs."""
    od_pairs = []

    while len(od_pairs) < count:
        od_pairs.append(sample_od_pair(polygon))

    return od_pairs


def point_to_lat_lon(point):
    """Convert a shapely Point with x=lon, y=lat to a (lat, lon) tuple."""
    return point.y, point.x



# %%
od_pairs = generate_od_pairs(
    polygon=vienna_polygon,
    count=OD_PAIRS_COUNT,
)

len(od_pairs)


# %%
od_map = folium.Map(
    location=vienna_center,
    zoom_start=11,
)

folium.GeoJson(vienna.geometry.to_json()).add_to(od_map)

for od_id, (origin_point, destination_point) in enumerate(od_pairs):
    folium.CircleMarker(
        location=[origin_point.y, origin_point.x],
        radius=3,
        color="green",
        fill=True,
        fill_opacity=0.8,
        popup=f"origin {od_id}",
    ).add_to(od_map)

    folium.CircleMarker(
        location=[destination_point.y, destination_point.x],
        radius=3,
        color="red",
        fill=True,
        fill_opacity=0.8,
        popup=f"destination {od_id}",
    ).add_to(od_map)

    folium.PolyLine(
        locations=[
            [origin_point.y, origin_point.x],
            [destination_point.y, destination_point.x],
        ],
        color="blue",
        weight=2,
        opacity=0.5,
    ).add_to(od_map)

od_map


# %% [markdown]
# # 5. Routing API
#

# %%
def get_route(origin, destination, theta):
    """Request a route for one origin-destination pair and one preference vector."""
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



# %% [markdown]
# # 6. Preference Vectors
#

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
    """Create a routing preference vector in API field naming."""
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

SAFE_THETA = route_params(
    road_class_cycleway=1.5,
    # no_car_access=1.5,
    # road_class_primary_secondary_trunk=0.5,
    # max_speed_above_thirty=0.5,
)


# %%
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



# %% [markdown]
# # 7. Feature Extraction and Distance Metrics
#

# %%
def extract_feature_vector(route):
    """Flatten routeFeatures into one feature-share dictionary."""
    feature_vector = {}

    route_features = route["properties"]["routeFeatures"]

    for category, values in route_features.items():
        for value, share in values.items():
            feature_vector[f"{category}:{value}"] = share

    return feature_vector


def l1_distance(features_a, features_b):
    """Compute L1 distance between sparse feature-share dictionaries."""
    feature_keys = set(features_a.keys()) | set(features_b.keys())

    return sum(
        abs(features_a.get(feature, 0) - features_b.get(feature, 0))
        for feature in feature_keys
    )


def route_coordinates(route):
    """Convert GeoJSON [lon, lat] route coordinates to folium-compatible (lat, lon)."""
    return [
        (lat, lon)
        for lon, lat in route["geometry"]["coordinates"]
    ]



# %% [markdown]
# # 8. Baseline Example
#

# %%
base_route = get_route(
    EXAMPLE_START,
    EXAMPLE_END,
    BASE_THETA,
)

safe_route = get_route(
    EXAMPLE_START,
    EXAMPLE_END,
    SAFE_THETA,
)


# %%
def plot_route_comparison(route_a, route_b, label_a="BASE", label_b="SAFE"):
    """Plot two routes on one folium map."""
    coordinates_a = route_coordinates(route_a)
    coordinates_b = route_coordinates(route_b)

    route_map = folium.Map(
        location=coordinates_a[0],
        zoom_start=14,
    )

    folium.PolyLine(
        coordinates_a,
        color="blue",
        weight=6,
        tooltip=label_a,
    ).add_to(route_map)

    folium.PolyLine(
        coordinates_b,
        color="red",
        weight=4,
        tooltip=label_b,
    ).add_to(route_map)

    return route_map


plot_route_comparison(
    base_route,
    safe_route,
)


# %%
base_features = extract_feature_vector(base_route)
safe_features = extract_feature_vector(safe_route)

comparison_df = pd.DataFrame({
    "BASE": base_features,
    "SAFE": safe_features,
}).fillna(0)

comparison_df


# %%
comparison_df.plot(
    kind="bar",
    figsize=(16, 5),
)

plt.ylabel("Share")
plt.title("Route Feature Comparison")
plt.tight_layout()
plt.show()


# %%
absolute_feature_difference = (
    comparison_df["BASE"]
    .sub(comparison_df["SAFE"])
    .abs()
)

absolute_feature_difference.sort_values(ascending=False)


# %%
absolute_feature_difference.sort_values().plot.barh(
    figsize=(8, 6),
)

plt.title("Absolute Feature Differences")
plt.show()


# %%
example_feature_distance = l1_distance(
    base_features,
    safe_features,
)

print(f"L1 feature distance: {example_feature_distance:.4f}")


# %% [markdown]
# # 9. Experiment Execution
#
# This section generates the route dataset used in the results chapter.
# For each OD pair, one baseline route is generated with BASE_THETA and
# additional routes are generated with randomly sampled parameter vectors.

# %%
def run_od_experiment(origin, destination, theta):
    """Run one OD experiment and return both the route and its feature vector."""
    candidate_route = get_route(origin, destination, theta)
    candidate_features = extract_feature_vector(candidate_route)

    return candidate_route, candidate_features


def route_metadata(route):
    """Extract route-level metadata from a GraphHopper route response.

    Note:
    route_time_min is assumed to already be returned in minutes.
    """
    properties = route["properties"]

    return {
        "route_length_m": properties.get("distance"),
        "route_time_min": properties.get("time"),
        "route_ascend_m": properties.get("ascend"),
        "route_descend_m": properties.get("descend"),
    }


def make_experiment_row(
    od_id,
    sample_id,
    theta,
    route,
    features,
    feature_distance,
    is_baseline,
):
    """Create one results row from route metadata, parameters, and features."""
    row = {
        "od_id": od_id,
        "sample_id": sample_id,
        "is_baseline": is_baseline,
        **route_metadata(route),
        **theta,
        "feature_distance": feature_distance,
    }

    for feature_name, feature_value in features.items():
        row[feature_name] = feature_value

    return row


# %%
experiment_rows = []
route_records = []
base_feature_vectors = {}

for od_id, (origin_point, destination_point) in enumerate(od_pairs):
    origin = point_to_lat_lon(origin_point)
    destination = point_to_lat_lon(destination_point)

    try:
        base_route = get_route(
            origin,
            destination,
            BASE_THETA,
        )
        base_features = extract_feature_vector(base_route)
        base_feature_vectors[od_id] = base_features

        base_row = make_experiment_row(
            od_id=od_id,
            sample_id=-1,
            theta=BASE_THETA,
            route=base_route,
            features=base_features,
            feature_distance=0.0,
            is_baseline=True,
        )

        experiment_rows.append(base_row)
        route_records.append({
            "od_id": od_id,
            "sample_id": -1,
            "is_baseline": True,
            "route": base_route,
        })

        for sample_id in range(ROUTES_PER_OD):
            random_theta = sample_theta()

            candidate_route, candidate_features = run_od_experiment(
                origin,
                destination,
                random_theta,
            )

            feature_distance = l1_distance(
                base_features,
                candidate_features,
            )

            experiment_row = make_experiment_row(
                od_id=od_id,
                sample_id=sample_id,
                theta=random_theta,
                route=candidate_route,
                features=candidate_features,
                feature_distance=feature_distance,
                is_baseline=False,
            )

            experiment_rows.append(experiment_row)
            route_records.append({
                "od_id": od_id,
                "sample_id": sample_id,
                "is_baseline": False,
                "route": candidate_route,
            })

    except Exception as error:
        print(f"failed OD {od_id}: {error}")


# %% [markdown]
# # 10. Results Dataset
#
# This section constructs the main results dataframe and defines reusable
# column groups for the following analysis.

# %%
results_df = pd.DataFrame(experiment_rows)
route_records_df = pd.DataFrame(route_records)

PARAMETER_COLUMNS = list(BASE_THETA.keys())

METADATA_COLUMNS = [
    "od_id",
    "sample_id",
    "is_baseline",
    "route_length_m",
    "route_time_min",
    "route_ascend_m",
    "route_descend_m",
    "feature_distance",
]

FEATURE_COLUMNS = [
    column
    for column in results_df.columns
    if column not in METADATA_COLUMNS
    and column not in PARAMETER_COLUMNS
]

results_df[FEATURE_COLUMNS] = (
    results_df[FEATURE_COLUMNS]
    .fillna(0)
)

results_df.head()


# %%
dataset_overview = pd.Series({
    "od_pairs": results_df["od_id"].nunique(),
    "routes_total": len(results_df),
    "baseline_routes": results_df["is_baseline"].sum(),
    "sampled_routes": (~results_df["is_baseline"]).sum(),
    "features": len(FEATURE_COLUMNS),
    "parameters": len(PARAMETER_COLUMNS),
})

dataset_overview


# %%
# todo this is useless

route_length_time_summary = results_df[
    [
        "route_length_m",
        "route_time_min",
        "route_ascend_m",
        "route_descend_m",
        "feature_distance",
    ]
].describe()

route_length_time_summary


# %% [markdown]
# # 11. Feature Variability
#
# This section follows the thesis structure for the feature variability results:
#
# 1. Visual example of one OD pair and selected routes.
# 2. Feature ranges for three example OD pairs.
# 3. Aggregate feature variability across all OD pairs.
# 4. Heatmap of feature variability.
# 5. L1 distance summaries.


# %% [markdown]
# ## Selecting example OD pairs


# %%
feature_std_by_od_df = (
    results_df
    .groupby("od_id")[FEATURE_COLUMNS]
    .std()
    .fillna(0)
)

mean_feature_std_by_od = (
    feature_std_by_od_df
    .mean(axis=1)
    .sort_values()
)

MAP_OD_ID = int(mean_feature_std_by_od.idxmax())

MAP_OD_ID


# %%
def select_low_medium_high_od_ids(score_by_od):
    """Select low-, medium-, and high-variability OD pairs."""
    sorted_scores = score_by_od.sort_values()

    low_od = int(sorted_scores.index[0])
    medium_od = int(sorted_scores.index[len(sorted_scores) // 2])
    high_od = int(sorted_scores.index[-1])

    selected = []
    for od_id in [low_od, medium_od, high_od]:
        if od_id not in selected:
            selected.append(od_id)

    return selected


EXAMPLE_OD_IDS = select_low_medium_high_od_ids(mean_feature_std_by_od)

EXAMPLE_OD_IDS


# %% [markdown]
# ## Visualize example routes

# %%
def get_route_from_records(od_id, sample_id):
    """Return the stored route geometry for a given OD pair and sample ID."""
    matches = route_records_df[
        (route_records_df["od_id"] == od_id)
        & (route_records_df["sample_id"] == sample_id)
    ]

    if matches.empty:
        raise ValueError(f"No route found for od_id={od_id}, sample_id={sample_id}")

    return matches.iloc[0]["route"]


def selected_routes_for_map(od_id, top_n=5):
    """Select baseline plus top-N routes by L1 feature distance for one OD pair."""
    od_df = results_df[
        results_df["od_id"] == od_id
    ]

    baseline_row = od_df[
        od_df["is_baseline"]
    ].iloc[0]

    top_rows = (
        od_df[
            ~od_df["is_baseline"]
        ]
        .sort_values("feature_distance", ascending=False)
        .head(top_n)
    )

    selected_rows = pd.concat([
        baseline_row.to_frame().T,
        top_rows,
    ])

    selected_routes = []

    for _, row in selected_rows.iterrows():
        route = get_route_from_records(
            od_id=int(row["od_id"]),
            sample_id=int(row["sample_id"]),
        )

        selected_routes.append({
            "label": (
                "baseline"
                if row["is_baseline"]
                else f"sample {int(row['sample_id'])}"
            ),
            "feature_distance": row["feature_distance"],
            "route": route,
        })

    return selected_routes


def plot_route_overlay(selected_routes):
    """Plot selected routes on one folium map."""
    first_route_coordinates = route_coordinates(selected_routes[0]["route"])

    route_map = folium.Map(
        location=first_route_coordinates[0],
        zoom_start=13,
    )

    colors = ["blue", "red", "green", "purple", "orange", "black"]

    for route_id, route_record in enumerate(selected_routes):
        coordinates = route_coordinates(route_record["route"])

        tooltip = (
            f"{route_record['label']} "
            f"(L1={route_record['feature_distance']:.3f})"
        )

        folium.PolyLine(
            coordinates,
            color=colors[route_id % len(colors)],
            weight=4 if route_record["label"] == "baseline" else 3,
            opacity=0.8,
            tooltip=tooltip,
        ).add_to(route_map)

    return route_map


# %%
map_routes = selected_routes_for_map(
    od_id=MAP_OD_ID,
    top_n=5,
)

plot_route_overlay(map_routes)


# %% [markdown]
# ## Feature variability across all OD pairs


# %%
behavior_rows = []

for od_id in sorted(results_df["od_id"].unique()):
    od_df = results_df[
        results_df["od_id"] == od_id
    ]

    base_features = base_feature_vectors[od_id]

    od_behavior_space_df = pd.DataFrame({
        "min_share": od_df[FEATURE_COLUMNS].min(),
        "mean_share": od_df[FEATURE_COLUMNS].mean(),
        "max_share": od_df[FEATURE_COLUMNS].max(),
    })

    od_behavior_space_df["base_share"] = [
        base_features.get(feature, 0)
        for feature in od_behavior_space_df.index
    ]

    od_behavior_space_df["range"] = (
        od_behavior_space_df["max_share"]
        - od_behavior_space_df["min_share"]
    )

    od_behavior_space_df["mean_delta_to_base"] = (
        od_behavior_space_df["mean_share"]
        - od_behavior_space_df["base_share"]
    )

    mean_abs_delta_to_base = {}

    for feature in FEATURE_COLUMNS:
        base_value = base_features.get(feature, 0)
        mean_abs_delta_to_base[feature] = (
            od_df[feature]
            .sub(base_value)
            .abs()
            .mean()
        )

    od_behavior_space_df["mean_abs_delta_to_base"] = (
        pd.Series(mean_abs_delta_to_base)
    )

    od_behavior_space_df["od_id"] = od_id
    od_behavior_space_df["feature"] = od_behavior_space_df.index

    behavior_rows.append(
        od_behavior_space_df.reset_index(drop=True)
    )


behavior_by_od_df = pd.concat(
    behavior_rows,
    ignore_index=True,
)

behavior_summary_df = (
    behavior_by_od_df
    .groupby("feature")
    .agg({
        "min_share": "mean",
        "base_share": "mean",
        "mean_share": "mean",
        "max_share": "mean",
        "range": "mean",
        "mean_delta_to_base": "mean",
        "mean_abs_delta_to_base": "mean",
    })
    .rename(columns={
        "min_share": "mean_min_share",
        "base_share": "mean_base_share",
        "mean_share": "mean_mean_share",
        "max_share": "mean_max_share",
        "range": "mean_range",
        "mean_delta_to_base": "mean_delta_to_base",
        "mean_abs_delta_to_base": "mean_abs_delta_to_base",
    })
    .sort_values("mean_range", ascending=False)
)

behavior_summary_df

# todo no mean_mean_share


# %%
# Compact thesis table candidate.
behavior_summary_df[
    [
        "mean_min_share",
        "mean_base_share",
        "mean_mean_share",
        "mean_max_share",
        "mean_range",
        "mean_delta_to_base",
        "mean_abs_delta_to_base",
    ]
].head(25)


# %%
plt.figure(figsize=(10, 8))

behavior_summary_df["mean_range"].sort_values().plot.barh()

plt.xlabel("Mean Reachable Range")
plt.title("Aggregate Feature Variability Across OD Pairs")
plt.show()


# %%
plt.figure(figsize=(10, 8))

behavior_summary_df["mean_abs_delta_to_base"].sort_values().plot.barh()

plt.xlabel("Mean Absolute Delta to Baseline")
plt.title("Mean Feature Deviation from Baseline")
plt.show()


# %% [markdown]
# ## Heatmap


# %%
feature_range_matrix = (
    behavior_by_od_df
    .pivot(
        index="od_id",
        columns="feature",
        values="range",
    )
    .fillna(0)
)

# Reorder columns by global mean range.
feature_range_matrix = feature_range_matrix[
    behavior_summary_df.index
]

plt.figure(figsize=(16, 8))

sns.heatmap(
    feature_range_matrix,
    cmap="viridis",
)

plt.xlabel("Feature")
plt.ylabel("OD Pair")
plt.title("Feature Variability Across OD Pairs")
plt.show()


# %% [markdown]
# ## Feature Ranges for example OD pairs

# %%
def behavior_space_for_od(od_id):
    """Compute min, mean, max, baseline, and range per feature for one OD pair."""
    od_df = results_df[
        results_df["od_id"] == od_id
        ]

    base_features = base_feature_vectors[od_id]

    behavior_space_df = pd.DataFrame({
        "min_share": od_df[FEATURE_COLUMNS].min(),
        "mean_share": od_df[FEATURE_COLUMNS].mean(),
        "max_share": od_df[FEATURE_COLUMNS].max(),
    })

    behavior_space_df["base_share"] = [
        base_features.get(feature, 0)
        for feature in behavior_space_df.index
    ]

    behavior_space_df["range"] = (
            behavior_space_df["max_share"]
            - behavior_space_df["min_share"]
    )

    behavior_space_df = (
        behavior_space_df
        .sort_values("range", ascending=False)
    )

    return behavior_space_df


def plot_behavior_space_dots(od_id, top_n=15):
    """Plot sampled feature values with baseline marker for one OD pair."""
    behavior_space_df = behavior_space_for_od(od_id)
    top_features = behavior_space_df.head(top_n).index.tolist()

    od_samples_df = results_df[results_df["od_id"] == od_id]

    plt.figure(figsize=(10, 8))

    for y_pos, feature in enumerate(top_features):
        sample_values = od_samples_df[feature].dropna()

        # background range
        plt.hlines(
            y=y_pos,
            xmin=sample_values.min(),
            xmax=sample_values.max(),
            alpha=0.2,
            linewidth=4,
            zorder=1,
        )

        # sampled routes
        plt.scatter(
            sample_values,
            np.full(len(sample_values), y_pos),
            alpha=0.5,
            s=20,
            zorder=2,
            label="Sample Routes" if y_pos == 0 else None,
        )

        # baseline
        plt.scatter(
            behavior_space_df.loc[feature, "base_share"],
            y_pos,
            s=60,
            edgecolor="black",
            linewidth=1.5,
            zorder=3,
            label="Baseline Route" if y_pos == 0 else None,
        )

    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel("Feature Share")
    plt.title(f"Reachable Behavior Space Distribution (OD {od_id})")
    plt.legend()
    plt.show()

    return behavior_space_df


# %%
example_behavior_spaces = {}

for od_id in EXAMPLE_OD_IDS:
    example_behavior_spaces[od_id] = plot_behavior_space_dots(
        od_id=od_id,
        top_n=50,
    )

# todo decide on feature order, make consist across plots
# maybe dont show lines but instead points for every sample route


# %%
# Optional: inspect the table behind one example plot.
example_behavior_spaces[EXAMPLE_OD_IDS[-1]].head(20)


# %% [markdown]
# ## L1 Distance to Baseline
#
# The current `feature_distance` column stores the L1 distance between each
# sampled route feature vector and the baseline feature vector of the same
# OD pair. At this stage, the existing feature distance is used as computed
# during experiment execution.
#
# TODO: Replace or recompute this with normalized feature vectors once the
# final normalization strategy for L1 distance is fixed.

# %%
results_df["feature_distance"].describe()


# %%
l1_summary_by_od_df = (
    results_df
    .groupby("od_id")["feature_distance"]
    .agg(["min", "max", "mean", "std"])
    .sort_values("mean", ascending=False)
)

l1_summary_by_od_df


# %%
plt.figure(figsize=(8, 4))

plt.hist(
    results_df["feature_distance"],
    bins=30,
)

plt.xlabel("L1 Feature Distance to Baseline")
plt.ylabel("Count")
plt.title("Distribution of Route Feature Distances to Baseline")
plt.show()


# %%
plt.figure(figsize=(10, 5))

l1_summary_by_od_df["mean"].sort_values().plot.bar()

plt.ylabel("Mean L1 Feature Distance to Baseline")
plt.xlabel("OD Pair")
plt.title("Mean Route Feature Distance to Baseline per OD Pair")
plt.show()


# %%
plt.figure(figsize=(10, 5))

l1_summary_by_od_df["max"].sort_values().plot.bar()

plt.ylabel("Maximum L1 Feature Distance to Baseline")
plt.xlabel("OD Pair")
plt.title("Maximum Route Feature Distance to Baseline per OD Pair")
plt.show()


# %% [markdown]
# ## Route length and time variability


# %%
route_metric_summary_df = (
    results_df
    .groupby("od_id")
    .agg({
        "route_length_m": ["min", "max", "mean", "std"],
        "route_time_min": ["min", "max", "mean", "std"],
        "route_ascend_m": ["min", "max", "mean", "std"],
        "route_descend_m": ["min", "max", "mean", "std"],
    })
)

route_metric_summary_df


# %% [markdown]
# # Finding routes that match routing personas

# %%
def min_max_normalize(series):
    series_min = series.min()
    series_max = series.max()

    if series_max == series_min:
        return pd.Series(
            0.5,
            index=series.index
        )

    return (
        (series - series_min)
        / (series_max - series_min)
    )


def compute_persona_scores(
    df,
    persona_name,
    positive_features,
    negative_features
):
    scored_df = df.copy()

    score_components = []

    for feature in positive_features:
        score_column = f"{feature}_score"

        scored_df[score_column] = min_max_normalize(
            scored_df[feature]
        )

        score_components.append(score_column)

    for feature in negative_features:
        score_column = f"{feature}_score"

        scored_df[score_column] = (
            1
            - min_max_normalize(scored_df[feature])
        )

        score_components.append(score_column)

    scored_df[f"{persona_name}_score"] = (
        scored_df[score_components]
        .mean(axis=1)
    )

    return scored_df


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

SAFE_FEATURES = (
    SAFE_POSITIVE_FEATURES
    + SAFE_NEGATIVE_FEATURES
)

SAFE_SCORE_COLUMNS = [
    f"{feature}_score"
    for feature in SAFE_FEATURES
]

# %%
safe_scored_df = compute_persona_scores(
    df=results_df,
    persona_name="safe",
    positive_features=SAFE_POSITIVE_FEATURES,
    negative_features=SAFE_NEGATIVE_FEATURES
)

# %%
TOP_N = 5
OD_ID = EXAMPLE_OD

base_route_df = safe_scored_df[
    (safe_scored_df["od_id"] == OD_ID)
    & (safe_scored_df["sample_id"] == -1)
]

top_safe_routes_df = (
    safe_scored_df[
        (safe_scored_df["od_id"] == OD_ID)
        & (safe_scored_df["sample_id"] != -1)
    ]
    .sort_values("safe_score", ascending=False)
    .head(TOP_N)
)

comparison_df = pd.concat(
    [
        base_route_df,
        top_safe_routes_df,
    ],
    ignore_index=True
)

comparison_df[
    [
        "od_id",
        "sample_id",
        "route_length_m",
        "route_time_min",
        *SAFE_FEATURES,
        *SAFE_SCORE_COLUMNS,
        "safe_score",
    ]
]

# %%
PARAMETER_COLUMNS = [
    "cyclewayLane",
    "cyclewayTrack",
    "roadClassCycleway",
    "roadClassPrimarySecondaryTrunk",
    "roadClassResidential",
    "roadClassPath",
    "roadClassFootway",
    "surfaceCobblestoneGravelUnpaved",
    "inclineAvgAboveFourPercent",
    "declineAvgAboveFourPercent",
    "noCarAccess",
    "bikeRoadAccessDesignated",
    "bikeRoadAccessDismountOrGetOffBike",
    "maxSpeedAboveThirty",
]

top_safe_routes_df[
    [
        "od_id",
        "sample_id",
        "safe_score",
        *PARAMETER_COLUMNS,
    ]
]
