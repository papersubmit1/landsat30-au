# -*- coding: utf-8 -*-
"""
This script processes OpenStreetMap (OSM) data from a GeoJSON file,
classifies OSM tags into broader pattern classes, and merges the results
with a dataset of clean satellite images.
"""

import json
import re
import textwrap

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1. Detectable pattern classes
# ----------------------------------------------------------------------
PATTERN_CLASSES = [
    "water_body",
    "river_stream",
    "wetland",
    "cropland",
    "natural_vegetation",
    "bare_ground",
    "urban_fabric",
    "road_corridor",
    "open_pit_mine",
    "glacier_or_snow",
    "unknown_or_too_small",
]

# ----------------------------------------------------------------------
# 2. Keyword bank for each class (case-insensitive substrings)
# ----------------------------------------------------------------------
KEYWORDS = {
    "water_body": r"\b(reservoir|lake|pond|lagoon|dam|tailings_pond|water\b)|farm[_\s\-]?dam",
    "river_stream": r"\b(river|stream|creek|drain|canal|oxbow|tidal_channel)\b",
    "wetland": r"\b(wetland|swamp|marsh|bog)\b",
    "cropland": r"\b(farmland|farm|farmyard|paddock|cropland|orchard|plantation|vineyard|paddock|cotton_gin)\b",
    "natural_vegetation": r"\b(forest|woodland|trees?|grassland|meadow|scrub|shrub)\b",
    "bare_ground": r"\b(bare_ground|sand|rock|dune|desert|clearing|rocky|stone|gravel)\b",
    "urban_fabric": r"\b(residential|village|town|suburb|city|commercial|industrial|factory|warehouse|retail|parking|building|(?:ice_)?factory|bakery|university|hospital|school|clinic|shopping|mall|casino|pub|restaurant|hotel|clubhouse|office|gym|stadium|arena|terminal|depot)\b",
    "road_corridor": r"\b(road|street|motorway|highway|primary|secondary|tertiary|trunk|track|service|path|railway|cycleway)\b",
    "open_pit_mine": r"\b(quarry|mineshaft|mine|spoil_heap|tailings|stockpile|landfill|waste(?:_transfer|water)?|dump|open[_\-]?pit)\b",
    "glacier_or_snow": r"\b(glacier|ice|snowfield|icefield|snow)\b",
}


MAPPING_RAW = textwrap.dedent("""
raw_tag new_tag
sand_and_rocks 	bare_ground
storage urban_fabric
pathology 	 urban_fabric
changeroom 	urban_fabric
caldera bare_ground
loading_dock 	 	urban_fabric
greenset 	 	natural_vegetation
isolated_dwelling 	 urban_fabric
gardens urban_fabric
storage_rental 	urban_fabric
coast_guard urban_fabric
retirement_village 	urban_fabric
horse_riding 	 	urban_fabric
storage_tank 	 	urban_fabric
bowling_alley 	 urban_fabric
moat 	 	river_stream
car_wash 	 	urban_fabric
sport_centre 	 	urban_fabric
paved;dirt 	urban_fabric
nursing_home 	 	urban_fabric
reeds 	 wetland
dressing_room 	 urban_fabric
research 	 	urban_fabric
marine_rescue 	 urban_fabric
post_depot 	urban_fabric
playground 	urban_fabric
acting_school 	 urban_fabric
shooting_range 	urban_fabric
Softfall 	 	urban_fabric
optometrist;dentist urban_fabric
disc_golf_course 	 	urban_fabric
heritage 	 	urban_fabric
campground 	urban_fabric
planetarium urban_fabric
greenery 	 	natural_vegetation
plants 	natural_vegetation
machine_shop 	 	urban_fabric
civic 	 urban_fabric
street_lamp urban_fabric
paddling_pool 	 urban_fabric
cafe;bar 	 	urban_fabric
truck_rental;car_rental urban_fabric
hostel 	urban_fabric
plaza 	 urban_fabric
habour 	habour
sett 	 	urban_fabric
distillery 	urban_fabric
Wetlands 	 	wetlands
neighbourhood 	 urban_fabric
paper_mill 	urban_fabric
swimming_school urban_fabric
education 	 urban_fabric
convention_centre 	 urban_fabric
private_dam dam
research_institute 	urban_fabric
dance 	 urban_fabric
distributor urban_fabric
laundry urban_fabric
artificial 	urban_fabric
pet care 	 	urban_fabric
water_tank 	urban_fabric
grass_paver urban_fabric
driving_school 	urban_fabric
swimming_pool 	 urban_fabric
crematorium urban_fabric
living_seawall 	urban_fabric
recreation_centre 	 urban_fabric
beach_resort 	 	urban_fabric
road_transport 	road_corridor
living_street 	 urban_fabric
cinema 	urban_fabric
forestry 	 	forestry
hookah_lounge 	 urban_fabric
greenhouse_horticulture urban_fabric
fish 	 	urban_fabric
water_plant water plant
auction_house 	 urban_fabric
National Park 	 park
cafe;bicycle_rental urban_fabric
social_club urban_fabric
cryogenic_plant urban_fabric
car_wash;cafe 	 urban_fabric
riverbed 	 	riverbed
nature_reserve;park park
Timber 	urban_fabric
steelmaking urban_fabric
slaughterhouse 	urban_fabric
pier 	 	habour
fitness_centre 	urban_fabric
scrap_yard 	urban_fabric
college urban_fabric
government 	urban_fabric
grating urban_fabric
biergarten 	urban_fabric
waste_transfer_station 	urban_fabric
prison 	urban_fabric
city_block 	urban_fabric
fitness_station urban_fabric
coastline 	 coastline
vehicle_inspection 	urban_fabric
nature_reserve 	nature reserve
port 	 	port
townhall 	 	urban_fabric
waterfall 	 water_body
automotive_industry urban_fabric
dog_park 	 	park
schoolyard;pitch 	 	urban_fabric
post_office;cafe 	 	urban_fabric
ice_rink 	 	urban_fabric
grass_scrub natural_vegetation
sports_hall urban_fabric
day_care 	 	urban_fabric
causeway 	 	river_stream
Central Park 	 	park
courthouse 	urban_fabric
governmental 	 	urban_fabric
dentist urban_fabric
retirement_home urban_fabric
amusement_arcade 	 	urban_fabric
brickworks 	urban_fabric
village_green 	 natural_vegetation
driving_training 	 	urban_fabric
boarding_house 	urban_fabric
asphalt_plant 	 urban_fabric
dirt;mud 	 	urban_fabric
spring 	water_body
hill 	 	hill
water_storage 	 water_body
dancing_school 	urban_fabric
doctors urban_fabric
strait 	strait
driver_training urban_fabric
natural_reserve natural_reserve
dry_swamp 	 wetland
public_building urban_fabric
court 	 urban_fabric
cemetery 	 	urban_fabric
agriculture agriculture
civil 	 urban_fabric
cliff 	 bare_ground
gasfield 	 	urban_fabric
boat_storage 	 	urban_fabric
car_pooling urban_fabric
drystream 	 wetland
shoal 	 shoal
bowls_club 	urban_fabric
language_school urban_fabric
tailings_dam 	 	tailings dam
port;intermodal_freight_terminal 	 	port
public_hall urban_fabric
pharmacy 	 	urban_fabric
council urban_fabric
island 	island
brownfield 	brownfield
dirt/rocks 	brownfield
coworking_space urban_fabric
sluice_gate urban_fabric
swimming_area 	 water_body
concert_hall 	 	urban_fabric
jetty 	 jetty
student_accommodation 	 urban_fabric
yacht_club 	urban_fabric
hall 	 	urban_fabric
monastery 	 urban_fabric
boatyard 	 	urban_fabric
licenced_club 	 urban_fabric
golf_course natural_vegetation
cruise_terminal urban_fabric
private parking_space 	 urban_fabric
apartments 	urban_fabric
resort 	urban_fabric
spa urban_fabric
boat_rental urban_fabric
meditation_centre 	 urban_fabric
manufacturing 	 urban_fabric
veterinary 	urban_fabric
club 	 	urban_fabric
funeral_home 	 	urban_fabric
brickyard 	 urban_fabric
stream_pool water_body
flight_school 	 urban_fabric
motorcycle_parking 	urban_fabric
gas_plant 	 urban_fabric
canyon 	canyon
sanitary_dump_station 	 urban_fabric
overland_flow 	 river_stream
works_depot urban_fabric
greenfield 	greenfield
basin 	 basin
brothel urban_fabric
motor_registry 	urban_fabric
Earth_Tank_,_Off_Dtream_Flow_Dam 	 	dam
bicycle_parking urban_fabric
accommodation 	 urban_fabric
studio 	urban_fabric
rugby 	 urban_fabric
oil_tank 	 	urban_fabric
indoor_golf urban_fabric
grass;mud 	 natural_vegetation
telecommunication 	 urban_fabric
cooling_tower 	 urban_fabric
water_treatment urban_fabric
wharf 	 wharf
ferry_terminal 	port
parking_space 	 urban_fabric
motel 	 urban_fabric
The Crescent Centre urban_fabric
watermill 	 urban_fabric
pools 	 pool
conference_center 	 urban_fabric
Private_Dam dam
groyne 	dam
aged_care 	 urban_fabric
Private_dam dam
chain_bay 	 chain_bay
tower 	 urban_fabric
dog_parking park
natural natural_reserve
sewerage 	 	natural_reserve
sports_centre 	 urban_fabric
Tennis_Court 	 	urban_fabric
mineral_processing 	industry
brewery urban_fabric
salt_pool 	 pool
cattleyards cropland
summer_camp urban_fabric
pitch;playground 	 	urban_fabric
communiity_farm cropland
hamlet 	hamlet
trampoline_park urban_fabric
licensed_club 	 urban_fabric
adult_gaming_centre urban_fabric
local_government 	 	urban_fabric
water_park 	park
theatre urban_fabric
wastewater_plant 	 	water_plant
schoolyard 	urban_fabric
sports_centre;pitch urban_fabric
kindergarten 	 	urban_fabric
auditorium 	urban_fabric
beach 	 beach
Sand_and_rock 	 bare_ground
festival_grounds 	 	urban_fabric
isthmus isthmus
fast_food 	 urban_fabric
concrete_plant 	industry
embankment 	embankment
healthcare 	urban_fabric
salt_pond 	 salt pond
church 	urban_fabric
internet_cafe 	 urban_fabric
feeding_place 	 urban_fabric
exhibition_centre 	 urban_fabric
peak 	 	peak
park;swimming;fishing;boating;picnic 	 	park
coal_mine 	 open_pit_mine
police 	urban_fabric
canteen urban_fabric
concrete(07)_3010_6880 	urban_fabric
garden_centre 	 urban_fabric
post_office urban_fabric
lighthouse 	urban_fabric
industrlal 	industrial
gambling 	 	urban_fabric
training 	 	urban_fabric
aquaculture water_body
agricultural 	 	agricultural
bay bay
quay 	 	quay
community_centre 	 	urban_fabric
stockyards 	cropland
community_food_growing 	cropland
funeral_hall 	 	urban_fabric
dry_lake 	 	lake
dunes 	 bare_ground
ambulance_station 	 urban_fabric
pumping_station urban_fabric
winery 	urban_fabric
arts_centre urban_fabric
childcare 	 urban_fabric
school_camp urban_fabric
prep_school urban_fabric
mine_restoration 	 	open_pit_mine
bare_rock 	 bare_ground
community_hall 	urban_fabric
apiary 	cropland
cafe 	 	urban_fabric
caravan_park 	 	park
Warnbro Sports Ground 	 urban_fabric
mountain_range 	bare_ground
weir 	 	weir
livestock 	 cropland
reserve reserve
reservior 	 reservior
harbour harbour
peninsula 	 peninsula
social_centre 	 urban_fabric
woodchip_mulch 	urban_fabric
dock 	 	dock
conference_centre 	 urban_fabric
Sand_and_rocks 	bare_ground
museum 	urban_fabric
art_gallery urban_fabric
nightclub 	 urban_fabric
fuel_depot 	urban_fabric
fishpond 	 	pond
airport urban_fabric
courtyard 	 urban_fabric
churchyard 	urban_fabric
racetrack 	 urban_fabric
bar urban_fabric
library urban_fabric
dog_wash 	 	urban_fabric
toy_library urban_fabric
park 	 	park
retirement 	urban_fabric
stockyard 	 cropland
refinery 	 	industry
cafe;fast_food 	urban_fabric
levee 	 levee
container_terminal 	terminal
marina 	marina
residence 	 urban_fabric
food_court 	urban_fabric
irrigation 	cropland
""").strip()

# Parse into DataFrame (skipping header)
# Note: The original code splits by tab '\t', which may not match the
# whitespace in the MAPPING_RAW string above. This is preserved from the original.
pairs = [line.split("\t") for line in MAPPING_RAW.splitlines()[1:]]

pairs_dict = {}
for pair in pairs:
    # Ensure the pair has two elements to avoid index errors
    if len(pair) == 2:
        pairs_dict[pair[0]] = pair[1]

# Pre-compile regexes for speed
CLASS_REGEX = {cls: re.compile(pat, re.I) for cls, pat in KEYWORDS.items()}


def tag_to_pattern(tag: str) -> str:
    """
    Return the best-matching 30m pattern class for a raw OSM tag.

    Args:
        tag (str): The raw OSM tag string.

    Returns:
        str: The corresponding pattern class or 'unknown_or_too_small'.
    """
    tag = tag.strip().lower()
    for cls, rx in CLASS_REGEX.items():
        if rx.search(tag):
            return cls
    # Check manual mapping if regex fails
    if tag in pairs_dict:
        return pairs_dict[tag]
    return "unknown_or_too_small"


# --- File Paths ---
osm_file = "keep_4_landuse_metadata-categories-landuse.geojson"
osm_df = gpd.read_file(osm_file)

update_osm_keys = []
for osm_keys_json in osm_df["osm_keys"]:
    osm_key_dict = json.loads(osm_keys_json)
    updated_osm_key_list = []
    for tag_list in osm_key_dict.values():
        classified_tags = [
            tag_to_pattern(tag)
            for tag in tag_list
            if tag_to_pattern(tag) != "unknown_or_too_small"
        ]
        updated_osm_key_list.extend(classified_tags)
    update_osm_keys.append(", ".join(updated_osm_key_list))

osm_df["update_osm_keys"] = update_osm_keys