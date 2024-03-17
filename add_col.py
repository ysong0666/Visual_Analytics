import pandas as pd

distance = pd.read_csv("distance.csv")
data = pd.read_csv("LekagulSensorData.csv")

data["speeded"] = False
data["skipped_sensor"] = False
data["is_night"] = False
data["trespassed"] = False

cars = data["car-id"].unique()


# odd_entrance = []
# odd_ranger_base = []

# for c in cars:
#     temp = data.loc[data["car-id"] == c]
#     if temp["gate-name"].iloc[0][:-1] != "entrance" or \
#         temp["gate-name"].iloc[-1][:-1] != "entrance":
#         if temp["car-type"].iloc[0] != "2P":
#             odd_entrance.append(c)
#         elif temp["gate-name"].iloc[0] != "ranger-base" or \
#                 temp["gate-name"].iloc[-1] != "ranger-base":
#             odd_ranger_base.append(c)
# print(odd_entrance)
# print(odd_ranger_base)
result = ['20155705025759-63', '20165520095518-683', '20161324021341-735', 
          '20164626054635-56', '20162027082019-218', '20161327091348-83', 
          '20161327021321-308', '20160827030854-705', '20161327031357-119', 
          '20163928063941-914', '20164928094952-954', '20165628115651-59', 
          '20161528021509-878', '20160928040927-911', '20163928043915-121', 
          '20161529081503-449', '20161229101224-445', '20161229031207-187', 
          '20165329035305-622', '20162729042749-245', '20160730070750-227', 
          '20164030094001-59', '20162630022643-828', '20161331071346-196', 
          '20160631080640-121', '20164531124512-886', '20164931124920-126', 
          '20161431011459-931', '20164731014734-842', '20160331030337-77', 
          '20160831030828-838', '20163831043800-36', '20165831105856-579', 
          '20161031111001-854']
# for c in result:
#     temp = data.loc[data["car-id"] == c]
#     #print(temp)
#     if temp["Timestamp"].iloc[0] < "2016-05-26":
#         print(temp)
# temp = data.loc[data["car-id"] == "20155705025759-63"]
# print(temp.to_string())

def get_miles(sensor1, sensor2):
    if sensor1 == sensor2:
        return 0
    mpp = 12/200
    temp = distance.loc[distance["sensor1"] == sensor1]
    if sensor2 in temp['sensor2'].unique():
        tt = temp.loc[distance["sensor2"] == sensor2]
        return tt["distance"].item() * mpp 
    temp = distance.loc[distance["sensor1"] == sensor2]
    return temp.loc[distance["sensor2"] == sensor1]["distance"].item() * mpp 

# #print(get_miles("entrance1", "gate2"))

# for c in cars:
#     temp = data.loc[data["car-id"] == c]
#     n = len(temp)
#     for i in range(1, n):
#         miles = get_miles(temp["gate-name"].iloc[i-1], temp["gate-name"].iloc[i])
#         #skipped_sensor
#         if miles < 0:
#             data.loc[data["Timestamp"] == temp["Timestamp"].iloc[i], "skipped_sensor"] = True
#             continue  
#         start = pd.to_datetime(temp["Timestamp"].iloc[i-1])
#         end = pd.to_datetime(temp["Timestamp"].iloc[i])
#         #is_night & trespassed
#         if i == 1:
#             if start.hour > 20 or start.hour < 6:
#                 data.loc[(data["Timestamp"] == temp["Timestamp"].iloc[i-1]) & (data["car-id"] == temp["car-id"].iloc[i]), "is_night"] = True
#             if temp["gate-name"].iloc[i-1][:-1] == "gate" and temp["car-type"].iloc[i-1] != "2P":
#                 data.loc[(data["Timestamp"] == temp["Timestamp"].iloc[i-1]) & (data["car-id"] == temp["car-id"].iloc[i]), "trespassed"] = True
#         if end.hour > 20 or end.hour < 6:
#             data.loc[(data["Timestamp"] == temp["Timestamp"].iloc[i]) & (data["car-id"] == temp["car-id"].iloc[i]), "is_night"] = True
#         if temp["gate-name"].iloc[i][:-1] == "gate" and temp["car-type"].iloc[i] != "2P":
#             data.loc[(data["Timestamp"] == temp["Timestamp"].iloc[i]) & (data["car-id"] == temp["car-id"].iloc[i]), "trespassed"] = True
#         hours = (end-start).total_seconds() / 360
#         #print("miles", miles, "hours", hours, "speed", miles / hours)
#         #speeded
#         if miles / hours > 25:
#             data.loc[(data["Timestamp"] == temp["Timestamp"].iloc[i]) & (data["car-id"] == temp["car-id"].iloc[i]), "speeded"] = True
#         #print(data.loc[data["Timestamp"] == temp["Timestamp"].iloc[i]]["speeded"])

# print(data)
# data.to_csv("add_col.csv")

# Suspicous skipped sensor

# added_data = pd.read_csv("add_col.csv")
# # #print(added_data.loc[added_data["skipped_sensor"] == True].to_string())
# cars_skipped = added_data.loc[added_data["skipped_sensor"] == True]["car-id"].unique()
# # #print(cars_skipped)
# print(added_data[added_data['car-id'].isin(cars_skipped)])
# #print(added_data.loc[added_data["car-id"] == "20152810102803-808"].to_string())
# #print(added_data.loc[(added_data["Timestamp"] < "2015-07-10 17") & (added_data["Timestamp"] > "2015-07-10 10")].to_string())

#Suspicious Speeding
# added_data = pd.read_csv("add_col.csv")
# print(added_data.loc[added_data["speeded"] == True].to_string())


# Suspicious Trespassing
added_data = pd.read_csv("add_col.csv")
print(added_data.loc[added_data["trespassed"] == True].to_string())

dur = pd.read_csv("car.csv")
print(dur.loc[dur["left-park"] == False].to_string())


#Duration
     
# duration = {}
# for c in cars:
#     temp = data.loc[data["car-id"] == c]
#     start = pd.to_datetime(temp["Timestamp"].iloc[0])
#     end = pd.to_datetime(temp["Timestamp"].iloc[-1])
#     if temp["gate-name"].iloc[0][:-1] != "entrance" or \
#        temp["gate-name"].iloc[-1][:-1] != "entrance":
#         if temp["car-type"].iloc[0] != "2P":
#             duration[c] = (end-start, False)
#             continue
#     duration[c] = (end-start, True)
    
# print(len(cars))
# print(len(duration))
# duration_df = pd.DataFrame.from_dict(duration, orient='index').reset_index()
# duration_df.columns = ["car-id", "duration", "left-park"]
# print(duration_df)
# duration_df.to_csv("car.csv")