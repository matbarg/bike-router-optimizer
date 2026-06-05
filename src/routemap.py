import folium
import itertools


class RouteMap:
    COLORS = itertools.cycle([
        'red', 'blue', 'green', 'purple', 'orange',
        'darkred', 'lightred', 'beige', 'darkblue'
    ])

    def __init__(self):
        self.map = folium.Map(location=[48.2, 16.37], zoom_start=10)

    def add_geojson(self, geojson, color=None, weight=5) -> None:
        folium.GeoJson(geojson,
                       style_function=lambda feature: {
                           "color": color or next(RouteMap.COLORS),
                           "weight": weight,
                           "opacity": 1.0
                       },
                       tooltip=folium.GeoJsonTooltip(
                           fields=["distance","cost"],
                           align=["Distance (km):", "Cost:"],
                       )).add_to(self.map)

    def add_point_list(self, points: list[tuple[float, float]], color=None, weight=5) -> None:
        folium.PolyLine(points,
                        color=color or next(RouteMap.COLORS),
                        weight=weight,
                        opacity=1.0).add_to(self.map)


    def save(self, file="map.html") -> None:
        self.map.save(file)
