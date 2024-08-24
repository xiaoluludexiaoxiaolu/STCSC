import os
class_list = ["airplane", "airport", "baseball_diamond", "basketball_court", "beach",
              "bridge", "chaparral", "church", "circular_farmland", "cloud",
              "commercial_area", "dense_residential", "desert", "forest", "freeway",
              "golf_course", "ground_track_field", "harbor", "industrial_area", "intersection",
              "island", "lake", "meadow", "medium_residential", "mobile_home_park",
              "mountain", "overpass", "palace", "parking_lot", "railway",
              "railway_station", "rectangular_farmland", "river", "roundabout", "runway",
              "sea_ice", "ship", "snowberg", "sparse_residential", "stadium",
              "storage_tank", "tennis_court", "terrace", "thermal_power_station", "wetland"]

for i in class_list:
    file_name = "train_val_data/trainval_threshold8/" + i
    print(file_name)
    os.mkdir(file_name)