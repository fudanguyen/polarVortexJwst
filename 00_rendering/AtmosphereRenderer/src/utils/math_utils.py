def calculate_sine_wave(amplitude, frequency, phase, time):
    return amplitude * np.sin(2 * np.pi * frequency * time + phase)

def calculate_cosine_wave(amplitude, frequency, phase, time):
    return amplitude * np.cos(2 * np.pi * frequency * time + phase)

def gaussian(x, mean, std_dev):
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

def linear_interpolation(x0, x1, t):
    return (1 - t) * x0 + t * x1

def area_of_circle(radius):
    return np.pi * radius ** 2

def circumference_of_circle(radius):
    return 2 * np.pi * radius

def convert_latitude_to_pixel(lat, image_height):
    return int((90 - lat) / 180 * image_height)

def convert_longitude_to_pixel(lon, image_width):
    return int((lon + 180) / 360 * image_width)