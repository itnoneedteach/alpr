import sys
import json

plates = {"plates": []}

# parse cars from TG @dadfindboy
with open('plates_dadfindboy.json', 'r') as json_file:
    plates = json.load(json_file)

# parse cars from TG @dadfindcar
# parse government cars
with open('g_plates.txt', 'r') as txt_file:
    for line in txt_file:
        [plate, make, model, color, location, comment] = line.split("\t")
        plate = plate.strip().upper()
        make = make.strip().upper()
        model = model.strip().upper()
        location = location.strip()
        comment = comment[:-1]

        if plate:
            plates["plates"].append({"plate": plate, "make": make, "model": model, "color": color, "location": location, "comment": comment, "source": "TG: @dadfindcar", "category": "#G 政府車", "ID": ""})

# parse police private cars
with open('pp_plates.txt', 'r') as txt_file:
    for line in txt_file:
        [ID, plate, make, model, color, location, comment] = line.split("\t")
        plate = plate.strip().upper()
        make = make.strip().upper()
        model = model.strip().upper()
        location = location.strip()
        comment = comment[:-1]

        if plate:
            plates["plates"].append({"plate": plate, "make": make, "model": model, "color": color, "location": location, "comment": comment, "source": "TG: @dadfindcar", "category": "#A 差佬私人車", "ID": "A"+ID})

#plates["plates"].sort(key=lambda x: x["plate"])

# output to json file
with open('plates_all.json', 'w') as json_file:
    json.dump(plates, json_file, indent=4)
