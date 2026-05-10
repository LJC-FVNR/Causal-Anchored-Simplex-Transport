from __future__ import annotations

TRACK_NAME = "owid_dietary_composition_by_country"

CSV_URL = "https://ourworldindata.org/grapher/dietary-composition-by-country.csv"
GRAPHER_PAGE_URL = "https://ourworldindata.org/grapher/dietary-composition-by-country"
TOPIC_PAGE_URL = "https://ourworldindata.org/diet-compositions"

WINDOW_START_YEAR = 1961
WINDOW_END_YEAR = 2023
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

FOOD_COLUMNS = [
    "Miscellaneous",
    "Alcoholic beverages",
    "Animal fats",
    "Vegetable oils",
    "Oilcrops",
    "Fish and seafood",
    "Sugar crops",
    "Sugar and sweeteners",
    "Starchy roots",
    "Other meat",
    "Sheep and goat meat",
    "Pork",
    "Poultry meat",
    "Beef and buffalo meat",
    "Eggs",
    "Milk",
    "Nuts",
    "Fruit",
    "Vegetables",
    "Pulses",
    "Other cereals",
    "Barley",
    "Maize",
    "Rice",
    "Wheat",
]

