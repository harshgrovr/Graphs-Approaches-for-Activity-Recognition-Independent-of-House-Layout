
import os
import json
import pandas as pd
import numpy as np
import json

file_names = ['houseA', 'houseB', 'houseC', 'ordonezA', 'ordonezB']

rooms_across_all_house = {'Toilet': 0, 'Bathroom': 1, 'Kitchen': 2, 'Bedroom': 3, 'Hall': 4, 'Office': 5, 'LivingRoom': 6, 'OutsideRegion': 7}

total_rooms = 0
for file_name in file_names:

    json_file_name = file_name + '.json'
    csv_file_name = file_name + '.csv'

    json_file = os.path.join('../../data', file_name, json_file_name)
    csv_file = os.path.join('../../data', file_name, csv_file_name)

    with open(json_file) as f:
        json_file = json.load(f)

    csv_file = pd.read_csv(csv_file)

    locations = json_file['locations']

    sensors = json_file['sensors']
    def sensor_place_in_house_ID(col_name, location_id=1):
        # Find sensor dict and take the location key, whose name matches with col names
        location_id = [x['location'] for x in sensors if x['name'] == col_name][0]
        if len(location_id) == 1:
            location_type = [x['type'] for x in locations if x['id'] == location_id[0]][0]
            return [rooms_across_all_house[location_type]]
        else:
            items = []
            for item in location_id:
                location_type = [x['type'] for x in locations if x['id'] == item][0]
                items.append(rooms_across_all_house[location_type])
            return items


    # get rooms as nodes first, by giving them id and make a dictionary of their features
    nodes_dictionary_list = []
    d = {}

    # Sensor place in house will be taken from here, the index will represent place in house of sensor
    rooms = [name['type'] for name in locations]
    count = 0
    # convert it into a dictionary list
    for i, node in enumerate(rooms):
        d = {}
        d['Id'] = count
        d['Object'] = node

        d['Value'] = -1
        d['place_in_house'] = rooms_across_all_house[node]
        d['Type'] = 1

        # Room id will be taken from here
        count += 1
        nodes_dictionary_list.append(d)
        # if node not in rooms_across_all_house:
        #     rooms_across_all_house[node] = total_rooms
        #     total_rooms += 1

    # csv column names
    col = list(csv_file.columns[4:].values)

    for i, col_name in enumerate(col):
        Id = i
        Object = col_name.split('_')[0]
        Value = -1
        place_in_house = sensor_place_in_house_ID(Object)
        Type = 0

        # Append to nodes_dictionary_list

        for location_id in place_in_house:
            d = {}
            d['Id'] = len(nodes_dictionary_list)
            d['Object'] = Object
            d['Value'] = -1
            d['Type'] = 0
            d['place_in_house'] = location_id
            nodes_dictionary_list.append(d)

    # Adding time of the day as the last node

    d = {}
    d['Id'] = len(nodes_dictionary_list)
    d['Object'] = 'time_of_the_day'
    d['Value'] = -1
    d['Type'] = -1
    d['place_in_house'] = -1

    nodes_dictionary_list.append(d)

    # print(list(nodes_dictionary_list))

    import csv

    toCSV = nodes_dictionary_list

    # json_file = os.path.join('../../data', file_name)

    keys = toCSV[0].keys()
    with open(os.path.join('../../data', file_name, 'nodes.csv'), 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(toCSV)

    # Write Edge.csv

    # Make a list of src and destination and write it via pandas
    Src = []
    Dst = []
    for node_dict in nodes_dictionary_list:
        if node_dict['Type'] == 0:
            # Birectional Edges
            Src.append(node_dict['Id'])
            Dst.append(node_dict['place_in_house'])

            Src.append(node_dict['place_in_house'])
            Dst.append(node_dict['Id'])

    # Connect rooms

    for location_dict in locations:
        src_location_id = rooms.index(location_dict['type'])

        reaches_list = location_dict['reaches']
        for reach_id in reaches_list:
            Type = [x['type'] for x in locations if x['id'] == reach_id][0]
            dst_location_id = rooms.index(Type)
            Src.append(src_location_id)
            Dst.append(dst_location_id)

    Src.append(nodes_dictionary_list[-1]['Id'])
    Dst.append(nodes_dictionary_list[-1]['Id'])

    df = pd.DataFrame({'Src': Src, 'Dst': Dst})
    df.to_csv(os.path.join('../../data', file_name, 'bidrectional_edges.csv'), index=False)
# print(rooms_across_all_house)