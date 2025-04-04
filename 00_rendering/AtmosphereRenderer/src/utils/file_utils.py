def save_image_with_cmap(data, filename, cmap='inferno', dpi=300):
    normalized = (np.clip(data, data.min(), data.max()) - data.min()) / (data.max() - data.min())
    cmap = plt.cm.get_cmap(cmap)
    rgba = cmap(normalized)
    img = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))
    img.save(filename, dpi=(dpi, dpi), format='PNG')

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")

def load_data_from_file(filepath):
    with open(filepath, 'r') as file:
        data = file.readlines()
    return [line.strip() for line in data]

def save_data_to_file(data, filepath):
    with open(filepath, 'w') as file:
        for line in data:
            file.write(f"{line}\n")

def load_npy_data(filepath):
    return np.load(filepath)

def save_npy_data(data, filepath):
    np.save(filepath, data)