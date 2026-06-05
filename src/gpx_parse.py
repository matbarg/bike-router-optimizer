import gpxpy
import json

from gpxpy.gpx import GPXTrackPoint

with open("input.gpx", "r") as f:
    gpx = gpxpy.parse(f)

points = gpx.tracks[0].segments[0].points

def gpx_points_to_linestring(points: list(GPXTrackPoint)):
    coords = []

    for point in points:
        coords.append([point.latitude, point.longitude])

    return {
        "geometry": {
            "type": "LineString",

        }
    }