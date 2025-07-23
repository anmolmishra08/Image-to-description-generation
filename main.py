import os
import string
import random
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import TextVectorization
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# --- 1. PATHS ---
images_dir = r'C:/Users/asus/Desktop/Test/dataset/flickr8k/images'
captions_file = r'C:/Users/asus/Desktop/Test/dataset/flickr8k/captions.txt'
features_path = 'image_features.pkl'
model_path = 'image_caption_model.keras'  # Use .keras format for saving/loading

def read_file(path):
    df = pd.read_csv(path, sep=',')
    df.columns = [col.strip() for col in df.columns]
    print("Columns in DataFrame:", df.columns.tolist())
    df.dropna(subset=['caption', 'image'], inplace=True)
    return df

# --- 2. LOAD CAPTIONS FROM CSV ---
df = read_file(captions_file)

IMAGE_COL = 'image'
CAPTION_COL = 'caption'

captions_dict = {}
for idx, row in df.iterrows():
    img_name = row[IMAGE_COL]
    caption = row[CAPTION_COL]
    img_path = os.path.normpath(os.path.join(images_dir, img_name))
    if os.path.exists(img_path):
        if img_path in captions_dict:
            captions_dict[img_path].append(caption)
        else:
            captions_dict[img_path] = [caption]

image_paths = list(captions_dict.keys())
print(f"Total images found: {len(image_paths)}")

# --- 3. CAPTION CLEANING ---
def clean_caption(caption):
    # Keep only words, remove punctuation, but keep numbers
    caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
    caption = ' '.join([word for word in caption.split() if word.isalpha() or word.isdigit()])
    return '<start> ' + caption + ' <end>'

all_captions = []
for caps in captions_dict.values():
    for cap in caps:
        all_captions.append(clean_caption(cap))

print(f"Total cleaned captions: {len(all_captions)}")

# --- 4. TEXT VECTORIZATION ---
max_vocab_size = 10000
max_seq_length = 40

vectorizer = TextVectorization(
    max_tokens=max_vocab_size,
    output_sequence_length=max_seq_length,
    standardize=None
)
vectorizer.adapt(all_captions)
print("Text vectorization adapted.")

# --- 5. IMAGE FEATURE EXTRACTION (VGG16) ---
if os.path.exists(features_path):
    print("Loading precomputed image features...")
    with open(features_path, 'rb') as f:
        image_features = pickle.load(f)
else:
    vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    def extract_features(img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.vgg16.preprocess_input(x)
        features = vgg_model.predict(x, verbose=0)
        return features[0]
    image_features = {}
    print("Extracting image features...")
    for i, img_path in enumerate(image_paths):
        norm_img_path = os.path.normpath(img_path)
        try:
            image_features[norm_img_path] = extract_features(norm_img_path)
            if (i+1) % 100 == 0:
                print(f"Processed {i+1}/{len(image_paths)} images...")
        except Exception as e:
            print(f"Error processing {norm_img_path}: {e}")
    print(f"Extracted features for {len(image_features)} images.")
    with open(features_path, 'wb') as f:
        pickle.dump(image_features, f)
    print(f"Image features saved to {features_path}")

# --- 6. DATA GENERATOR ---
def data_generator(captions_dict, image_features, vectorizer, batch_size):
    input_img, input_seq, output_word = [], [], []
    img_items = list(captions_dict.items())
    while True:
        random.shuffle(img_items)
        for img_path, caps in img_items:
            random.shuffle(caps)
            norm_img_path = os.path.normpath(img_path)
            if norm_img_path not in image_features:
                continue
            for cap in caps:
                seq = vectorizer([clean_caption(cap)]).numpy()[0]
                for i in range(1, len(seq)):
                    in_seq, out_word = seq[:i], seq[i]
                    in_seq = np.pad(in_seq, (max_seq_length - len(in_seq), 0), 'constant')
                    input_img.append(image_features[norm_img_path])
                    input_seq.append(in_seq)
                    output_word.append(out_word)
                    if len(input_img) == batch_size:
                        yield (
                            (np.array(input_img, dtype=np.float32), np.array(input_seq, dtype=np.int64)),
                            np.array(tf.keras.utils.to_categorical(output_word, num_classes=max_vocab_size), dtype=np.float32)
                        )
                        input_img, input_seq, output_word = [], [], []

# --- 7. MODEL ARCHITECTURE & TRAINING ---
if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
else:
    img_input = Input(shape=(512,))
    img_dense = Dense(256, activation='relu')(img_input)

    cap_input = Input(shape=(max_seq_length,))
    cap_emb = Embedding(max_vocab_size, 256, mask_zero=True)(cap_input)
    cap_lstm = LSTM(256)(cap_emb)

    decoder = Add()([img_dense, cap_lstm])
    output = Dense(max_vocab_size, activation='softmax')(decoder)

    model = Model(inputs=[img_input, cap_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
    print(model.summary())

    # --- 8. TRAINING ---
    batch_size = 32
    steps_per_epoch = len(all_captions) // batch_size

    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(captions_dict, image_features, vectorizer, batch_size),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 512), dtype=tf.float32),      # image features
                tf.TensorSpec(shape=(None, max_seq_length), dtype=tf.int64),  # input sequence
            ),
            tf.TensorSpec(shape=(None, max_vocab_size), dtype=tf.float32)     # output word (one-hot)
        )
    )

    print("Starting model training...")
    history = model.fit(
        train_dataset,
        epochs=20,  # Try more epochs for better results
        steps_per_epoch=steps_per_epoch,
        verbose=1
    )
    print("Training complete.")
    model.save(model_path)
    print(f"Model saved as '{model_path}'.")

# --- 9. CAPTION GENERATION WITH BEAM SEARCH ---
def generate_caption_beam_search(img_path, model, vectorizer, image_features, beam_size=3):
    start_token = vectorizer(['<start>']).numpy()[0][0]
    end_token = vectorizer(['<end>']).numpy()[0][0]
    sequences = [[list([start_token]), 0.0]]
    img_feat = image_features[os.path.normpath(img_path)].reshape(1, -1)
    for _ in range(max_seq_length):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == end_token:
                all_candidates.append((seq, score))
                continue
            padded_seq = np.pad(seq, (max_seq_length - len(seq), 0), 'constant')
            preds = model.predict([img_feat, padded_seq.reshape(1, -1)], verbose=0)
            top_k = np.argsort(preds[0])[-beam_size:]
            for idx in top_k:
                candidate = [seq + [idx], score - np.log(preds[0][idx] + 1e-10)]
                all_candidates.append(candidate)
        sequences = sorted(all_candidates, key=lambda tup: tup[1])[:beam_size]
    best_seq = sequences[0][0]
    inv_vocab = dict((i, w) for i, w in enumerate(vectorizer.get_vocabulary()))
    caption = ' '.join([inv_vocab.get(idx, '') for idx in best_seq if idx != 0 and inv_vocab.get(idx, '') not in ['<start>', '<end>']])
    return caption

# --- 10. EVALUATION (BLEU SCORE) ---
print("\nEvaluating model on first 5 images...")
smoothie = SmoothingFunction().method4
for img_path in list(captions_dict.keys())[:5]:
    generated = generate_caption_beam_search(img_path, model, vectorizer, image_features)
    references = [clean_caption(cap).split() for cap in captions_dict[img_path]]
    print("Generated:", generated)
    print("References:", references)
    bleu_score = sentence_bleu(references, generated.split(), smoothing_function=smoothie)
    print(f"Image: {os.path.basename(img_path)}\nBLEU: {bleu_score:.3f}\n")

def predict_caption_for_user_image(model, vectorizer, image_features, vgg_model=None, max_seq_length=40):
    """
    Prompts the user for an image path, processes the image, and prints the generated caption.
    """
    img_path = input("Enter the full path to your image: ").strip()
    if not os.path.exists(img_path):
        print("Image file does not exist.")
        return

    # Extract features for the new image
    try:
        if vgg_model is None:
            vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.vgg16.preprocess_input(x)
        features = vgg_model.predict(x, verbose=0)[0]
    except Exception as e:
        print(f"Error processing the image: {e}")
        return

    # Generate caption using beam search
    start_token = vectorizer(['<start>']).numpy()[0][0]
    end_token = vectorizer(['<end>']).numpy()[0][0]
    sequences = [[list([start_token]), 0.0]]
    img_feat = features.reshape(1, -1)
    for _ in range(max_seq_length):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == end_token:
                all_candidates.append((seq, score))
                continue
            padded_seq = np.pad(seq, (max_seq_length - len(seq), 0), 'constant')
            preds = model.predict([img_feat, padded_seq.reshape(1, -1)], verbose=0)
            top_k = np.argsort(preds[0])[-3:]  # beam size 3
            for idx in top_k:
                candidate = [seq + [idx], score - np.log(preds[0][idx] + 1e-10)]
                all_candidates.append(candidate)
        sequences = sorted(all_candidates, key=lambda tup: tup[1])[:3]
    best_seq = sequences[0][0]
    inv_vocab = dict((i, w) for i, w in enumerate(vectorizer.get_vocabulary()))
    caption = ' '.join([inv_vocab.get(idx, '') for idx in best_seq if idx != 0 and inv_vocab.get(idx, '') not in ['<start>', '<end>']])
    print(f"\nPredicted Caption: {caption}")

# Example usage after training:
predict_caption_for_user_image(model, vectorizer, image_features, max_seq_length=max_seq_length)