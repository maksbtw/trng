import numpy as np
from PIL import Image
import hashlib
import yt_dlp
import subprocess
import time
import os

def logistic_map(x, lambd):
    return lambd * x * (1 - x)

def generate_chaotic_sequence(x0, lambd, length, discard):
    sequence = []
    x = x0
    for _ in range(length + discard):
        x = logistic_map(x, lambd)
        if discard <= 0:
            sequence.append(x)
        else:
            discard -= 1
    return sequence

def threshold_function(sequence, threshold):
    return [1 if x >= threshold else 0 for x in sequence]

def permute_image(image_array, index_sequence):
    rows, cols = image_array.shape
    new_image_array = np.zeros_like(image_array)
    for m in range(rows):
        for n in range(cols):
            new_image_array[m, n] = image_array[index_sequence[m * cols + n] // cols, index_sequence[m * cols + n] % cols]
    return new_image_array

def image_to_bit_planes(image_array):
    bit_planes = []
    for i in range(8):
        bit_plane = ((image_array >> i) & 1).flatten().tolist()
        bit_planes.append(bit_plane)
    return bit_planes

def generate_random_sequence(bit_sequences, thresholded_sequences):
    random_sequence = []
    for i in range(len(bit_sequences)):
        if len(bit_sequences[i]) != len(thresholded_sequences[i]):
            raise ValueError(f"Długość sekwencji bitowej nie zgadza się")
        random_sequence.append([(bit_sequences[i][k] ^ thresholded_sequences[i][k]) for k in range(len(bit_sequences[i]))])
    return random_sequence

def sha3_256(data):
    sha3 = hashlib.sha3_256()
    sha3.update(data)
    return sha3.digest()

def process_image(image_path):
    color_image = Image.open(image_path).convert('L')
    image_array = np.array(color_image)
    M, N = image_array.shape

    with open("source.bin", "wb") as file:
        for pixel_value in image_array.flatten():
            file.write(np.uint8(pixel_value).tobytes())

    # Generowanie 9 sekwencji chaotycznych
    lambd = 4.0
    discard = 250
    initial_values = [0.361, 0.362, 0.363, 0.364, 0.365, 0.366, 0.367, 0.368, 0.369]
    chaotic_sequences = []
    for x0 in initial_values:
        chaotic_sequences.append(generate_chaotic_sequence(x0, lambd, M * N, discard))

    # Permutacja pikseli obrazu
    index_sequence = np.argsort(chaotic_sequences[0]).tolist()
    permuted_image_array = permute_image(image_array, index_sequence)

    # progowanie
    threshold = 0.5
    thresholded_sequences = [threshold_function(seq, threshold) for seq in chaotic_sequences[1:]]

    # Podział permutowanego obrazu na płaszczyzny bitowe
    bit_planes = image_to_bit_planes(permuted_image_array)

    # Generowanie binarnej sekwencji losowej
    random_sequences = generate_random_sequence(bit_planes, thresholded_sequences)

    with open("post.bin", "wb") as file:
        for seq in random_sequences:
            while len(seq) % 8 != 0:  # Dopychamy ciąg do długości podzielnej przez 8
                seq.append(0)

            string_of_bits = ''.join(list(map(str, seq)))
            packed_bytes = int(string_of_bits, 2).to_bytes(len(string_of_bits) // 8, byteorder='big')  # Pakujemy bity w bajty

            file.write(packed_bytes)

    with open("post.bin", "rb") as post_file, open("sha.bin", "wb") as sha_file:
        while True:
            block = post_file.read(32)
            if not block:
                break

            hash_value = sha3_256(block)

            sha_file.write(hash_value)

    print("Zakończono generowanie sha")
    
def get_stream_url(youtube_url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'force_generic_extractor': False,
        'format': 'best[ext=mp4]/best',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

def capture_single_frame(stream_url, output_path):
    ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin", "ffmpeg.exe")
    command = [
        ffmpeg_path,
        '-y',
        '-i', stream_url,
        '-frames:v', '1',
        '-q:v', '2',
        output_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg error:", result.stderr.strip())
        return False
    return os.path.exists(output_path)

def capture_multiple_frames(stream_url, output_dir, count=20, interval=1):
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    for i in range(count):
        filename = os.path.join(output_dir, f"frame_{i+1:02d}.jpg")
        print(f"Zrzut klatki {i+1}/{count}...")
        if capture_single_frame(stream_url, filename):
            frames.append(filename)
        else:
            print(f"Pominieto klatke {i+1}")
        time.sleep(interval)
    return frames

def combine_frames_horizontally(frame_paths, output_path):
    images = [Image.open(p) for p in frame_paths]
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)

    collage = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        collage.paste(img, (x_offset, 0))
        x_offset += img.width

    collage.save(output_path)
    print(f"Zapisano obraz jako {output_path}")

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=WKGK_hYnlGE"
    print("Pobieranie adresu streamu z YouTube...")
    stream_url = get_stream_url(youtube_url)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(project_dir, "frames")
    output_file = os.path.join(project_dir, "frame.jpg")

    print("Start przechwytywania klatek...")
    frame_paths = capture_multiple_frames(stream_url, frames_dir, count=20, interval=1)

    if frame_paths:
        print("Skladanie klatek w jeden obraz...")
        combine_frames_horizontally(frame_paths, output_file)

        process_image(output_file)
    else:
        print("Nie udalo sie przechwycic")