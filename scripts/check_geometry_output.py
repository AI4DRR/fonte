import geopandas as gpd
gdf = gpd.read_parquet("outputs/test_50_gpt5_medium/event_geometries.parquet")

print(gdf.dtypes)            # geometry should appear as 'geometry'
print(gdf.geometry.iloc[0])  # prints WKT, e.g. MULTIPOLYGON (((...)))
m = gdf.explore(
    tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    max_zoom=19,
)
m.save("map.html")