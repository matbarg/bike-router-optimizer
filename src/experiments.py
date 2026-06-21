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
OD_PAIRS_COUNT = 50
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

# %%
def run_od_experiment(origin, destination, theta):
    """Run one OD experiment and return both the route and its feature vector."""
    candidate_route = get_route(origin, destination, theta)
    candidate_features = extract_feature_vector(candidate_route)

    return candidate_route, candidate_features



# %%
experiment_rows = []
all_routes = []
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

        base_row = {
            "od_id": od_id,
            "sample_id": -1,
            "route_length_m": base_route["properties"]["distance"],
            "route_time_min": base_route["properties"]["time"],
            **BASE_THETA,
            "feature_distance": 0.0,
        }

        for feature_name, feature_share in base_features.items():
            base_row[feature_name] = feature_share

        experiment_rows.append(base_row)
        all_routes.append(base_route)

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

            experiment_row = {
                "od_id": od_id,
                "sample_id": sample_id,
                "route_length_m": candidate_route["properties"]["distance"],
                "route_time_min": candidate_route["properties"]["time"],
                **random_theta,
                "feature_distance": feature_distance,
            }

            for feature_name, feature_share in candidate_features.items():
                experiment_row[feature_name] = feature_share

            experiment_rows.append(experiment_row)
            all_routes.append(candidate_route)

    except Exception as error:
        print(f"failed OD {od_id}: {error}")

# %% [markdown]
# # 10. Results Dataset
#

# %%
results_df = pd.DataFrame(experiment_rows)

FEATURE_COLUMNS = [
    column
    for column in results_df.columns
    if ":" in column
]

results_df[FEATURE_COLUMNS] = (
    results_df[FEATURE_COLUMNS]
    .fillna(0)
)

results_df.head()


# %%
results_df["feature_distance"].describe()


# %%
results_df.groupby("od_id").mean(numeric_only=True)


# %%
results_df.groupby("od_id").std(numeric_only=True)


# %%
results_df.groupby("od_id").agg(["min", "max", "mean", "std"])


# %% [markdown]
# # 11. Route Overlay
#

# %%
def plot_route_overlay(routes):
    """Plot a collection of routes on one folium map."""
    route_map = folium.Map(
        location=[48.21, 16.34],
        zoom_start=13,
    )

    colors = ["blue", "red", "green", "purple", "orange"]

    for route_id, route in enumerate(routes):
        coordinates = route_coordinates(route)

        folium.PolyLine(
            coordinates,
            color=colors[route_id % len(colors)],
            weight=3,
            opacity=0.7,
        ).add_to(route_map)

    return route_map


plot_route_overlay(all_routes)


# %% [markdown]
# # 12. Summary Statistics
#

# %%
summary_df = (
    results_df
    .groupby("od_id")
    .agg({
        "route_length_m": ["min", "max", "mean", "std"],
        "route_time_min": ["min", "max", "mean", "std"],
        "feature_distance": ["min", "max", "mean", "std"],
    })
)

summary_df


# %%
plt.figure(figsize=(8, 4))

plt.hist(
    results_df["feature_distance"],
    bins=30,
)

plt.xlabel("L1 Feature Distance")
plt.ylabel("Count")
plt.title("Distribution of Route Feature Distances")
plt.show()


# %%
feature_variation = (
    results_df[FEATURE_COLUMNS]
    .std()
    .sort_values(ascending=False)
)

feature_variation


# %%
plt.figure(figsize=(10, 8))

feature_variation.plot.barh()

plt.xlabel("Standard Deviation")
plt.title("Global Feature Variability")
plt.show()


# %%
plt.figure(figsize=(10, 5))

results_df.boxplot(
    column="route_length_m",
    by="od_id",
)

plt.ylabel("Meters")
plt.title("Route Length Distribution")
plt.suptitle("")
plt.show()


# %%
plt.figure(figsize=(10, 5))

results_df.boxplot(
    column="route_time_min",
    by="od_id",
)

plt.ylabel("Milliseconds")
plt.title("Route Time Distribution")
plt.suptitle("")
plt.show()


# %%
if "cycleway:lane" in results_df.columns:
    plt.figure(figsize=(10, 5))

    results_df.boxplot(
        column="cycleway:lane",
        by="od_id",
    )

    plt.ylabel("Share")
    plt.title("Cycleway Lane Share Distribution")
    plt.suptitle("")
    plt.show()


# %%
feature_variation_matrix = (
    results_df
    .groupby("od_id")[FEATURE_COLUMNS]
    .std()
)

plt.figure(figsize=(16, 8))

sns.heatmap(
    feature_variation_matrix,
    cmap="viridis",
)

plt.title("Feature Variability Across OD Pairs")
plt.show()


# %%
diversity_score = (
    results_df
    .groupby("od_id")[FEATURE_COLUMNS]
    .std()
    .mean(axis=1)
)

plt.figure(figsize=(8, 4))

diversity_score.sort_values().plot.bar()

plt.ylabel("Mean Feature Std")
plt.xlabel("OD Pair")
plt.title("Route Diversity Score per OD Pair")
plt.show()


# %%
length_stats_df = (
    results_df
    .groupby("od_id")["route_length_m"]
    .agg(["mean", "std"])
)

length_stats_df["cv"] = (
    length_stats_df["std"]
    / length_stats_df["mean"]
)

length_stats_df["cv"].plot.bar()

plt.ylabel("Coefficient of Variation")
plt.title("Relative Route Length Variability")
plt.show()


# %% [markdown]
# # 13. Global Behavior Space
#

# %%
global_behavior_space_df = pd.DataFrame({
    "min": results_df[FEATURE_COLUMNS].min(),
    "max": results_df[FEATURE_COLUMNS].max(),
    "mean": results_df[FEATURE_COLUMNS].mean(),
    "std": results_df[FEATURE_COLUMNS].std(),
})

global_behavior_space_df["range"] = (
    global_behavior_space_df["max"]
    - global_behavior_space_df["min"]
)

global_behavior_space_df.sort_values(
    "range",
    ascending=False,
)


# %%
global_behavior_space_df["range"]     .sort_values()     .plot.barh(figsize=(10, 8))

plt.xlabel("Reachable Range")
plt.title("Behavior Space of Route Features")
plt.show()


# %% [markdown]
# # 14. Clustering
#

# %%
X = results_df[FEATURE_COLUMNS].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
)

results_df["cluster"] = kmeans.fit_predict(X_scaled)

results_df["cluster"].value_counts()


# %%
cluster_profiles_df = (
    results_df
    .groupby("cluster")[FEATURE_COLUMNS]
    .mean()
)

cluster_profiles_df


# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=results_df["cluster"],
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection of Route Feature Space")
plt.show()


# %% [markdown]
# # 15. Parameter Sensitivity
#

# %%
results_df["cycleway_track_bin"] = pd.cut(
    results_df["cyclewayTrack"],
    bins=5,
)

cycleway_track_response = (
    results_df
    .groupby("cycleway_track_bin", observed=False)["cycleway:lane"]
    .mean()
)

cycleway_track_response.plot(
    marker="o",
)

plt.ylabel("Mean cycleway:lane share")
plt.xlabel("cyclewayTrack bin")
plt.title("Sensitivity of cycleway:lane to cyclewayTrack")
plt.show()


# %% [markdown]
# # 16. Behavior Space for One OD Pair
#

# %%
example_od_df = results_df[
    results_df["od_id"] == EXAMPLE_OD
]

example_base_features = base_feature_vectors[EXAMPLE_OD]

example_behavior_space_df = pd.DataFrame({
    "min_share": example_od_df[FEATURE_COLUMNS].min(),
    "max_share": example_od_df[FEATURE_COLUMNS].max(),
})

example_behavior_space_df["base_share"] = [
    example_base_features.get(feature, 0)
    for feature in example_behavior_space_df.index
]

example_behavior_space_df["range"] = (
    example_behavior_space_df["max_share"]
    - example_behavior_space_df["min_share"]
)

example_behavior_space_df = (
    example_behavior_space_df
    .sort_values("range", ascending=False)
)

example_behavior_space_df


# %%
top_example_features = example_behavior_space_df.head(15)

plt.figure(figsize=(10, 8))

plt.hlines(
    y=top_example_features.index,
    xmin=top_example_features["min_share"],
    xmax=top_example_features["max_share"],
)

plt.scatter(
    top_example_features["base_share"],
    top_example_features.index,
    marker="o",
)

plt.xlabel("Feature Share")
plt.title(f"Reachable Behavior Space (OD {EXAMPLE_OD})")
plt.show()


# %% [markdown]
# # 17. Behavior Space for All OD Pairs
#

# %%
behavior_rows = []

for od_id in sorted(results_df["od_id"].unique()):
    od_df = results_df[results_df["od_id"] == od_id]
    base_features = base_feature_vectors[od_id]

    od_behavior_space_df = pd.DataFrame({
        "min_share": od_df[FEATURE_COLUMNS].min(),
        
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

    od_behavior_space_df["od_id"] = od_id
    od_behavior_space_df["feature"] = od_behavior_space_df.index

    behavior_rows.append(od_behavior_space_df.reset_index(drop=True))


behavior_by_od_df = pd.concat(
    behavior_rows,
    ignore_index=True
)


behavior_summary_df = (
    behavior_by_od_df
    .groupby("feature")
    .agg({
        "min_share": "mean",
        "max_share": "mean",
        "base_share": "mean",
        "range": "mean",
    })
    .rename(columns={
        "min_share": "mean_min_share",
        "max_share": "mean_max_share",
        "base_share": "mean_base_share",
        "range": "mean_range",
    })
    .sort_values("mean_range", ascending=False)
)

behavior_summary_df


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
