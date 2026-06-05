import random
from turtledemo.penrose import star

import requests
import gpxpy
from gpxpy.gpx import GPXTrackPoint

from routemap import RouteMap
from shapely.measurement import frechet_distance
from shapely import LineString

"""
request_body = {
        "profile": "FAST",
        "mode": "CUSTOM",
        "points": [
            {
                "lat": 48.21031811022499,
                "lon": 16.35246276855469
            },
            {
                "lat": 48.233249138277415,
                "lon": 16.34027481079102
            }
        ],
        "preferencesDto": {
            "bikeInfra": p1,
            "surfaces": p2,
            "hills": p3
        },
        "withInstructions": False
    }
"""


def get_route(start, end, theta):
    request_body = {
        "profile": "BASE",
        "mode": "CUSTOM",
        "points": [
            {"lat": start[0], "lon": start[1]},
            {"lat": end[0], "lon": end[1]}
        ],
        "preferencesDto": theta,
        "withInstructions": False
    }

    res = requests.post(RandomSearch.API_URL, json=request_body)
    return res.json()["geometry"]["coordinates"], res.json()


def route_params(bike_infra=1.0, surfaces=1.0, hills=1.0, car_free=1.0, main_roads=1.0, residential=1.0):
    return {
        "bikeInfra": bike_infra,
        "surfaces": surfaces,
        "hills": hills,
        "carFree": car_free,
        "mainRoads": main_roads,
        "residential": residential,
    }


def sample_theta():
    return route_params(
        bike_infra=random.uniform(0.3, 1.8),
        surfaces=random.uniform(0.3, 1.8),
        hills=random.uniform(0.3, 1.8),
        car_free=random.uniform(0.3, 1.8),
        main_roads=random.uniform(0.3, 1.8),
        residential=random.uniform(0.3, 1.8),
    )


class RandomSearch:
    API_URL = "http://localhost:8080/api/route"

    def __init__(self):
        self.map = RouteMap()

    def compute_loss(self, theta, dataset, show_route=False):
        total = 0.0

        for (start, end, preferred_route) in dataset:
            try:
                predicted_route, geojson = get_route(start, end, theta)

                if show_route:
                    self.map.add_geojson(geojson)

                print("Predicted route:", len(predicted_route))
                print("Preferred route:", len(preferred_route))
                dist = frechet_distance(LineString(predicted_route), LineString(preferred_route))
                print(f"Frechet distance: {dist:.6f}")
                total += dist
            except Exception as e:
                print("Routing failed:", e)
                total += 1e6  # penalty

        return total / len(dataset)

    def random_search(self, dataset, iterations=100):
        best_theta = None
        best_loss = float("inf")

        for i in range(iterations):
            show_route = i % 25 == 0

            theta = sample_theta()
            loss = self.compute_loss(theta, dataset, show_route)

            print(f"Iter {i}: loss={loss:.6f}, theta={theta}")


            if loss < best_loss:
                best_loss = loss
                best_theta = theta
                print("New best!")

        return best_theta, best_loss


def run_rs_with_generated_routes():
    rs = RandomSearch()

    target_params = route_params(
        bike_infra=1.0,
        surfaces=1.0,
        hills=0.5,
        car_free=1.8,
        main_roads=1.0,
        residential=1.5,
    )

    route_endpoints = [
        ((48.18051, 16.33375), (48.21890, 16.38062)),
        ((48.24354, 16.33221), (48.23199, 16.37392)),
        ((48.19768, 16.39778), (48.18566, 16.35358)),
        # ((48.18051, 16.33375), (48.23336, 16.37512))
    ]

    dataset = []

    for (start, end) in route_endpoints:
        route, geojson = get_route(start, end, target_params)
        rs.map.add_geojson(geojson, "green", weight=8)
        dataset.append((start, end, route))

    best_theta, best_loss = rs.random_search(dataset, iterations=100)

    print("Target parameters:", target_params)
    print("Best parameters:", best_theta)
    print("Best loss:", best_loss)

    for (start, end) in route_endpoints:
        route, geojson = get_route(start, end, best_theta)
        rs.map.add_geojson(geojson, "orange", weight=4)

    rs.map.save()


def get_data_from_gpx():
    with open("./resources/gpx-routes/2025-04-22_2181780448.gpx", "r") as f:
        gpx = gpxpy.parse(f)

    points = [(p.latitude, p.longitude) for p in gpx.tracks[0].segments[0].points]
    start = points[0]
    end = points[-1]

    return start, end, points


def run_rs_with_real_routes():
    rs = RandomSearch()

    dataset = [get_data_from_gpx()]

    for (start, end, points) in dataset:
        rs.map.add_point_list(points, "green", weight=8)

    best_theta, best_loss = rs.random_search(dataset, iterations=50)

    print("Best parameters:", best_theta)
    print("Best loss:", best_loss)

    for (start, end, points) in dataset:
        route, geojson = get_route(start, end, best_theta)
        rs.map.add_geojson(geojson, "orange", weight=4)

    rs.map.save()


def test_route():
    start = (48.18051, 16.33375)
    end = (48.23336, 16.37512)

    standard_params = route_params()

    custom_params = route_params(
        bike_infra=1.0,
        surfaces=1.0,
        hills=1.0,
        car_free=1.7,
        main_roads=1.0,
        residential=1.7,
    )

    route_map = RouteMap()

    standard_route, standard_geojson = get_route(start, end, standard_params)
    route_map.add_geojson(standard_geojson, "green", weight=8)
    custom_route, custom_geojson = get_route(start, end, custom_params)
    route_map.add_geojson(custom_geojson, "orange", weight=4)

    route_map.save()


if __name__ == "__main__":
    # test_route()
    # run_rs_with_generated_routes()
    run_rs_with_real_routes()
