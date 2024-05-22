from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = load_model('Animal_model.h5')

# Define the class labels using the provided dictionary
class_labels = {
    0: 'antelope', 1: 'badger', 2: 'bat', 3: 'bear', 4: 'bee', 5: 'beetle', 6: 'bison', 7: 'boar', 8: 'butterfly',
    9: 'cat', 10: 'caterpillar', 11: 'chimpanzee', 12: 'cockroach', 13: 'cow', 14: 'coyote', 15: 'crab', 16: 'crow',
    17: 'deer', 18: 'dog', 19: 'dolphin', 20: 'donkey', 21: 'dragonfly', 22: 'duck', 23: 'eagle', 24: 'elephant',
    25: 'flamingo', 26: 'fly', 27: 'fox', 28: 'goat', 29: 'goldfish', 30: 'goose', 31: 'gorilla', 32: 'grasshopper',
    33: 'hamster', 34: 'hare', 35: 'hedgehog', 36: 'hippopotamus', 37: 'hornbill', 38: 'horse', 39: 'hummingbird',
    40: 'hyena', 41: 'jellyfish', 42: 'kangaroo', 43: 'koala', 44: 'ladybugs', 45: 'leopard', 46: 'lion',
    47: 'lizard', 48: 'lobster', 49: 'mosquito', 50: 'moth', 51: 'mouse', 52: 'octopus', 53: 'okapi', 54: 'orangutan',
    55: 'otter', 56: 'owl', 57: 'ox', 58: 'oyster', 59: 'panda', 60: 'parrot', 61: 'pelecaniformes', 62: 'penguin',
    63: 'pig', 64: 'pigeon', 65: 'porcupine', 66: 'possum', 67: 'raccoon', 68: 'rat', 69: 'reindeer', 70: 'rhinoceros',
    71: 'sandpiper', 72: 'seahorse', 73: 'seal', 74: 'shark', 75: 'sheep', 76: 'snake', 77: 'sparrow', 78: 'squid',
    79: 'squirrel', 80: 'starfish', 81: 'swan', 82: 'tiger', 83: 'turkey', 84: 'turtle', 85: 'whale', 86: 'wolf',
    87: 'wombat', 88: 'woodpecker', 89: 'zebra'
}

# Additional information about each species (you can expand this dictionary)
species_info = {
    'antelope': 'A fast-running mammal with long horns, native to Africa and parts of Asia.',
    'badger': 'A burrowing mammal with distinctive black and white markings on its face, known for its aggressive behavior.',
    'bat': 'A nocturnal flying mammal capable of echolocation.',
    'bear': 'A large mammal with a stocky build and shaggy fur, known for its strength and hibernation habits.',
    'bee': 'A flying insect known for its role in pollination and honey production.',
    'beetle': 'A type of insect with a hard shell covering its wings, often found in gardens and forests.',
    'bison': 'A large herbivorous mammal with a humped back and shaggy fur, native to North America.',
    'boar': 'A wild pig, often characterized by its tusks and aggressive behavior.',
    'butterfly': 'A colorful insect with delicate wings, known for its fluttering flight.',
    'cat': 'A domesticated carnivorous mammal known for its agility, grace, and independent nature.',
    'caterpillar': 'The larval stage of a butterfly or moth, typically characterized by its segmented body and voracious appetite.',
    'chimpanzee': 'A highly intelligent and social primate native to Africa, closely related to humans.',
    'cockroach': 'A hardy insect known for its ability to survive in diverse environments, often considered a pest.',
    'cow': 'A domesticated ruminant mammal kept for its milk, meat, and other products.',
    'coyote': 'A wild canine native to North and Central America, known for its adaptable nature and distinctive howl.',
    'crab': 'A crustacean with a broad, flat body and large pincers, found in oceans and freshwater habitats.',
    'crow': 'A large, black bird known for its intelligence and loud cawing calls.',
    'deer': 'A hoofed mammal with branched antlers (in males) or no antlers (in females), found in forests and grasslands.',
    'dog': 'A domesticated carnivorous mammal known for its loyalty, trainability, and diverse breeds.',
    'dolphin': 'A highly intelligent marine mammal known for its playful behavior and echolocation abilities.',
    'donkey': 'A domesticated mammal related to horses, known for its long ears and braying sound.',
    'dragonfly': 'An insect with a long body and two pairs of transparent wings, known for its agile flight.',
    'duck': 'A waterfowl with a broad, flat bill and webbed feet, found in ponds, rivers, and lakes.',
    'eagle': 'A large bird of prey with keen eyesight and powerful talons, known for its soaring flight.',
    'elephant': 'A large mammal with a long trunk, large ears, and tusks, known for its intelligence and social behavior.',
    'flamingo': 'A tall wading bird with pink plumage and a distinctive curved bill, often found in large flocks near water.',
    'fly': 'A small flying insect known for its ability to hover and dart quickly through the air.',
    'fox': 'A carnivorous mammal with a pointed muzzle, bushy tail, and keen sense of smell.',
    'goat': 'A domesticated ruminant mammal known for its agility, surefootedness, and ability to climb.',
    'goldfish': 'A small freshwater fish with bright orange or gold coloring, often kept as a pet in ponds or aquariums.',
    'goose': 'A waterfowl with a long neck and loud honking call, often found in migratory flocks.',
    'gorilla': 'A large, powerful primate native to Africa, known for its strength and peaceful nature.',
    'grasshopper': 'An insect with long hind legs adapted for jumping, often found in grassy habitats.',
    'hamster': 'A small rodent often kept as a pet, known for its round body and furry cheeks.',
    'hare': 'A fast-running mammal similar to a rabbit, often found in grasslands and open fields.',
    'hedgehog': 'A small spiny mammal with a pointed snout, known for its ability to roll into a ball for defense.',
    'hippopotamus': 'A large herbivorous mammal with a barrel-shaped body and large mouth, native to Africa.',
    'hornbill': 'A large bird with a distinctive long, curved bill and brightly colored casque on its head.',
    'horse': 'A large domesticated mammal used for riding, racing, and work, known for its speed and strength.',
    'hummingbird': 'A small, brightly colored bird capable of hovering in mid-air while feeding on nectar.',
    'hyena': 'A carnivorous mammal with a sloping back and powerful jaws, known for its scavenging habits.',
    'jellyfish': 'A gelatinous marine creature with tentacles equipped with stinging cells, found in oceans worldwide.',
    'kangaroo': 'A marsupial mammal native to Australia, known for its powerful hind legs and ability to hop.',
    'koala': 'A marsupial mammal native to Australia, known for its eucalyptus diet and tree-dwelling habits.',
    'ladybugs': 'Small beetles with bright red or orange bodies and black spots, often considered beneficial insects.',
    'leopard': 'A large, spotted carnivorous cat known for its agility and stealthy hunting skills.',
    'lion': 'A large, muscular carnivorous cat known for its majestic mane and social behavior in prides.',
    'lizard': 'A reptile with a long body, scaly skin, and four legs, often found in diverse habitats worldwide.',
    'lobster': 'A large marine crustacean with a hard exoskeleton and long, jointed limbs, often caught for food.',
    'mosquito': 'A small flying insect known for its blood-feeding habits and ability to transmit diseases to humans.',
    'moth': 'A nocturnal flying insect with feathery antennae and typically drab-colored wings.',
    'mouse': 'A small rodent with a pointed snout and long tail, often considered a pest in human habitats.',
    'octopus': 'A marine mollusk with eight long tentacles and a soft, bulbous body, known for its intelligence and camouflage abilities.',
    'okapi': 'A giraffe-like mammal native to the forests of central Africa, known for its striped hindquarters and long neck.',
    'orangutan': 'A large, long-haired ape native to Indonesia and Malaysia, known for its intelligence and arboreal lifestyle.',
    'otter': 'A semiaquatic mammal with a streamlined body, webbed feet, and dense fur, often found near rivers and lakes.',
    'owl': 'A nocturnal bird of prey with large, forward-facing eyes and silent flight.',
    'ox': 'A domesticated bovine mammal used for pulling carts, plowing fields, and providing milk and meat.',
    'oyster': 'A bivalve mollusk with a rough shell, often found attached to rocks or other substrates in marine habitats.',
    'panda': 'A large bear-like mammal native to China, known for its distinctive black and white fur.',
    'parrot': 'A brightly colored bird with a strong, curved bill, known for its ability to mimic human speech.',
    'pelecaniformes': 'A group of water birds including pelicans, cormorants, and gannets, known for their long bills and excellent fishing skills.',
    'penguin': 'A flightless bird with flipper-like wings, found primarily in the Southern Hemisphere.',
    'pig': 'A domesticated mammal with a stout body, snout, and curly tail, often raised for its meat (pork).',
    'pigeon': 'A common bird found in urban areas worldwide, often known for its cooing call and iridescent plumage.',
    'porcupine': 'A rodent covered in sharp quills, known for its defensive behavior of raising its quills when threatened.',
    'possum': 'A nocturnal marsupial native to Australia and surrounding islands, known for its prehensile tail and ability to play dead.',
    'raccoon': 'A mammal with distinctive facial markings and a ringed tail, native to North and Central America.',
    'rat': 'A small rodent often considered a pest, known for its rapid reproduction and ability to spread disease.',
    'reindeer': 'A deer species native to the Arctic regions, known for its large antlers and association with Santa Claus.',
    'rhinoceros': 'A large mammal with a thick, protective skin and one or two horns on its nose, native to Africa and Asia.',
    'sandpiper': 'A small wading bird with long legs and a slender bill, often found on sandy beaches and mudflats.',
    'seahorse': 'A small marine fish with a distinctive horse-like head and curled tail, found in shallow coastal waters.',
    'seal': 'A marine mammal with a streamlined body and flippers, known for its playful behavior and whiskered face.',
    'shark': 'A large predatory fish with cartilaginous skeletons and sharp teeth, found in oceans worldwide.',
    'sheep': 'A domesticated ruminant mammal raised for its wool, meat (mutton), and milk.',
    'snake': 'A legless reptile with a long, flexible body and forked tongue, known for its predatory behavior.',
    'sparrow': 'A small bird with brown or gray plumage and a short, thick bill, often found in urban and suburban areas.',
    'squid': 'A marine mollusk with a soft body, eight arms, and two long tentacles, often used as food.',
    'squirrel': 'A small rodent with a bushy tail, known for its agility and habit of storing food in caches.',
    'starfish': 'A marine invertebrate with a star-shaped body and multiple arms, known for its ability to regenerate lost limbs.',
    'swan': 'A large waterfowl with a long neck and gracefully curved body, often associated with romance and beauty.',
    'tiger': 'A large carnivorous cat with orange fur and black stripes, known for its strength and agility.',
    'turkey': 'A large bird native to North America, often raised for its meat (turkey) and feathers.',
    'turtle': 'A reptile with a bony or cartilaginous shell covering its body, often found in aquatic habitats.',
    'whale': 'A large marine mammal with a streamlined body and blowhole for breathing, found in oceans worldwide.',
    'wolf': 'A carnivorous mammal related to domestic dogs, known for its intelligence and hunting prowess.',
    'wombat': 'A marsupial native to Australia, known for its burrowing behavior and stocky build.',
    'woodpecker': 'A bird with a specially adapted bill for drilling into trees, often found in forests and woodlands.',
    'zebra': 'A large herbivorous mammal with distinctive black and white stripes, native to Africa.',
}


@app.route('/')
def index():
    return render_template('test.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'file' in request.files:
        # Get the image file from the request
        file = request.files['file']
        img = Image.open(file)

        # Preprocess the image
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = img_array / 255.0

        # Make predictions
        predicted_probs = model.predict(np.expand_dims(img_array, axis=0))[0]
        predicted_class_idx = np.argmax(predicted_probs)
        confidence = float(predicted_probs[predicted_class_idx])
        predicted_class = class_labels[predicted_class_idx]

        # Retrieve additional information about the species
        species_description = species_info.get(predicted_class, 'No information available.')

        # Save the image temporarily in the static folder
        image_file_path = os.path.join(app.static_folder, 'uploaded_image.jpg')
        img.save(image_file_path)

        # Return the prediction result along with the image and species information
        return render_template('result.html', predicted_class=predicted_class, confidence=confidence,
                               species_description=species_description, image_file_path='uploaded_image.jpg')

    # If it's a GET request or no file was uploaded, render the form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
