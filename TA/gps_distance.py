import math

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # radius bumi dalam meter
    
    # 1. decimal degree ke radian
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    # 2. Ini haversine formula
    a = math.sin(delta_phi / 2)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # 3. Calculate the distance
    distance_meters = R * c
    return distance_meters

#-6.891935, 107.610573 second
# first -6.892191, 107.610710

dist = calculate_distance(-6.892025, 107.610583, -6.891935, 107.610573)
print(f"Distance: {dist:.2f} meters")