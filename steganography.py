import cv2
from PIL import Image
import numpy as np
from bitstring import BitStream

def encode_text(carrier_img, message_text, start_pos, skip_bits):
    # Convert text message to bits
    message_bits = BitStream(message_text.encode('utf-8'))
    
    # Flatten carrier image
    carrier_pixels = np.array(carrier_img)
    carrier_flat = carrier_pixels.flatten()
    
    # Check the maximum index to ensure we do not go out of bounds
    max_index = min(len(carrier_flat), start_pos + len(message_bits) * skip_bits)
    
    # Encoding message by modifying the least significant bit of the carrier image
    for i in range((max_index - start_pos) // skip_bits):
        index = start_pos + i * skip_bits
        if index < len(carrier_flat):  # Ensure index is within the range
            bit = message_bits.read('bin:1')
            carrier_flat[index] = (carrier_flat[index] & ~1) | int(bit, 2)
    
    # Reshape flattened array
    encoded_pixels = carrier_flat.reshape(carrier_pixels.shape)
    
    # Create and return encoded image
    encoded_image = Image.fromarray(encoded_pixels.astype(np.uint8))
    return encoded_image

def decode_text(encoded_img, start_pos, skip_bits, message_length):
    # Flatten encoded image
    encoded_pixels = np.array(encoded_img)
    encoded_flat = encoded_pixels.flatten()
 
    # Initialize bitstream to store decoded bits
    decoded_bits = BitStream()
 
    # Extracting bits from the flattened array
    for i in range((message_length - start_pos) // skip_bits):
        index = start_pos + i * skip_bits
        if index < len(encoded_flat):  # Ensure we don't go out of index bounds
            decoded_bits.append(BitStream(bin=str(encoded_flat[index] & 1)))
 
    # Convert decoded bits to text
    decoded_text = decoded_bits.bytes.decode('utf-8', errors='replace')
    return decoded_text

def encode_image(carrier_file, message_file, start_pos, skip_bits):
    carrier_img = Image.open(carrier_file)
    msg_img = Image.open(message_file)
    
    # Resize message image to match carrier size
    msg_resized = msg_img.resize(carrier_img.size, Image.Resampling.LANCZOS)
    
    # Convert images to numpy arrays
    carrier_pixels = np.array(carrier_img)
    msg_pixels = np.array(msg_resized)
    
    # Flatten arrays
    carrier_flat = carrier_pixels.flatten()
    msg_flat = msg_pixels.flatten()
    
    # Check the maximum index to ensure we do not go out of bounds
    max_index = min(len(carrier_flat), start_pos + len(msg_flat) * skip_bits)
    
    # Encoding message by modifying the least significant bit of the carrier image
    for i in range((max_index - start_pos) // skip_bits):
        index = start_pos + i * skip_bits
        if index < len(carrier_flat):  # Ensure index is within the range
            carrier_flat[index] = (carrier_flat[index] & ~1) | ((msg_flat[i] >> 7) & 1)
    
    # Reshape flattened array
    encoded_pixels = carrier_flat.reshape(carrier_pixels.shape)
    
    # Create and return encoded image
    encoded_image = Image.fromarray(encoded_pixels.astype(np.uint8))
    return encoded_image

def decode_image(encoded_img, start_pos, skip_bits, message_size):
    # Assumption: the carrier is an RGB image
    width, height = encoded_img.size
    encoded_pixels = np.array(encoded_img)
    encoded_flat = encoded_pixels.flatten()
 
    # Assume the message is embedded in the RGB channels, must calculate total bits accordingly
    expected_size = width * height * 3  # Assuming an RGB image
 
    decoded_flat = np.zeros(expected_size)
 
    # Extracting bits from the flattened array
    for i in range((expected_size - start_pos) // skip_bits):  # Dividing by 3 since we consider RGB components
        index = start_pos + i * skip_bits
        if index < len(encoded_flat):  # Ensure we don't go out of index bounds
            decoded_flat[index] = (encoded_flat[index] & 1) << 7  # Shift LSB to MSB position
 
    # Reshape according to the original image dimensions and the number of color channels
    decoded_pixels = decoded_flat.reshape((height, width, 3))
    decoded_image = Image.fromarray(decoded_pixels.astype(np.uint8))
    return decoded_image

def encode_video_single_frame(carrier_file, message_file, start_pos, skip_bits, output_path, frame_number):
    cap = cv2.VideoCapture(carrier_file)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Use the 'MJPG' codec for near-lossless compression
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height), True)
    
    frame_count = 0
    msg_img = Image.open(message_file)
    msg_resized = msg_img.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
    msg_pixels = np.array(msg_resized).flatten()
    
    encoded = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # If it's the specific frame we want to encode
        if frame_count == frame_number and not encoded:
            flat_frame = frame.flatten()
            for i in range(len(msg_pixels)):
                index = start_pos + i * skip_bits
                if index < len(flat_frame):
                    flat_frame[index] = (flat_frame[index] & ~1) | ((msg_pixels[i] >> 7) & 1)
            
            frame = flat_frame.reshape((frame_height, frame_width, 3))
            encoded = True  # Mark as encoded to avoid re-encoding if loop continues

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

def encode_image_on_image(carrier_img, message_img, start_pos, skip_bits):
    # Resize the message image to match the carrier size if they're not the same
    msg_resized = message_img.resize(carrier_img.size, Image.Resampling.LANCZOS)
    carrier_pixels = np.array(carrier_img)
    msg_pixels = np.array(msg_resized)
 
    carrier_flat = carrier_pixels.flatten()
    msg_flat = msg_pixels.flatten()
 
    # Check the maximum index to ensure we do not go out of bounds
    max_index = min(len(carrier_flat), start_pos + len(msg_flat) * skip_bits)
 
    # Encoding message by modifying the least significant bit of the carrier image
    for i in range((max_index - start_pos) // skip_bits):
        index = start_pos + i * skip_bits
        if index < len(carrier_flat):  # Ensure index is within the range
            carrier_flat[index] = (carrier_flat[index] & ~1) | ((msg_flat[i] >> 7) & 1)
 
    # Reshape flattened array
    encoded_pixels = carrier_flat.reshape(carrier_pixels.shape)
 
    # Create and return encoded image
    encoded_image = Image.fromarray(encoded_pixels.astype(np.uint8))
    return encoded_image

def decode_video_single_frame(video_path, start_pos, skip_bits, frame_number, message_size):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    bits = []
   
    while True:
        ret, frame = cap.read()
        if not ret or frame_count > frame_number:
            break
       
        # If it's the specific frame we want to decode
        if frame_count == frame_number:
            flat_frame = frame.flatten()
            for i in range((message_size - start_pos) // skip_bits):
                index = start_pos + i * skip_bits
                if index < len(flat_frame):
                    bits.append(str(flat_frame[index] & 1))
       
        frame_count += 1
   
    cap.release()
    bit_string = ''.join(bits)
    decode_msg = BitStream(bin=bit_string)
    return decode_msg.bytes.decode('utf-8', errors='replace')
