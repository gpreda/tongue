"""Static vocabulary challenges for common categories."""

# Spanish to English vocabulary by category
VOCABULARY_CHALLENGES = {
    'month': {
        'name': 'Month',
        'items': {
            'enero': 'january',
            'febrero': 'february',
            'marzo': 'march',
            'abril': 'april',
            'mayo': 'may',
            'junio': 'june',
            'julio': 'july',
            'agosto': 'august',
            'septiembre': 'september',
            'octubre': 'october',
            'noviembre': 'november',
            'diciembre': 'december'
        }
    },
    'season': {
        'name': 'Season',
        'items': {
            'primavera': 'spring',
            'verano': 'summer',
            'otoño': 'autumn,fall',
            'invierno': 'winter'
        }
    },
    'day': {
        'name': 'Day of Week',
        'items': {
            'lunes': 'monday',
            'martes': 'tuesday',
            'miércoles': 'wednesday',
            'jueves': 'thursday',
            'viernes': 'friday',
            'sábado': 'saturday',
            'domingo': 'sunday'
        }
    },
    'number': {
        'name': 'Number',
        'items': {
            'cero': '0,zero',
            'uno': '1,one',
            'dos': '2,two',
            'tres': '3,three',
            'cuatro': '4,four',
            'cinco': '5,five',
            'seis': '6,six',
            'siete': '7,seven',
            'ocho': '8,eight',
            'nueve': '9,nine',
            'diez': '10,ten',
            'once': '11,eleven',
            'doce': '12,twelve',
            'trece': '13,thirteen',
            'catorce': '14,fourteen',
            'quince': '15,fifteen',
            'dieciséis': '16,sixteen',
            'diecisiete': '17,seventeen',
            'dieciocho': '18,eighteen',
            'diecinueve': '19,nineteen',
            'veinte': '20,twenty',
            'veintiuno': '21,twenty-one',
            'veintidós': '22,twenty-two',
            'veintitrés': '23,twenty-three',
            'veinticuatro': '24,twenty-four',
            'veinticinco': '25,twenty-five',
            'veintiséis': '26,twenty-six',
            'veintisiete': '27,twenty-seven',
            'veintiocho': '28,twenty-eight',
            'veintinueve': '29,twenty-nine',
            'treinta': '30,thirty'
        }
    },
    'color': {
        'name': 'Color',
        'items': {
            'rojo': 'red',
            'azul': 'blue',
            'verde': 'green',
            'amarillo': 'yellow',
            'naranja': 'orange',
            'morado': 'purple',
            'rosa': 'pink',
            'negro': 'black',
            'blanco': 'white',
            'gris': 'gray,grey',
            'marrón': 'brown',
            'dorado': 'gold,golden'
        }
    },
    'family': {
        'name': 'Family',
        'items': {
            'madre': 'mother,mom',
            'padre': 'father,dad',
            'hermano': 'brother',
            'hermana': 'sister',
            'hijo': 'son',
            'hija': 'daughter',
            'abuelo': 'grandfather,grandpa',
            'abuela': 'grandmother,grandma',
            'tío': 'uncle',
            'tía': 'aunt',
            'primo': 'cousin',
            'esposo': 'husband',
            'esposa': 'wife'
        }
    },
    'animal': {
        'name': 'Animal',
        'items': {
            'perro': 'dog',
            'gato': 'cat',
            'pájaro': 'bird',
            'pez': 'fish',
            'caballo': 'horse',
            'vaca': 'cow',
            'cerdo': 'pig',
            'oveja': 'sheep',
            'pollo': 'chicken',
            'ratón': 'mouse',
            'conejo': 'rabbit',
            'oso': 'bear',
            'león': 'lion',
            'elefante': 'elephant',
            'mono': 'monkey'
        }
    },
    'body': {
        'name': 'Body Part',
        'items': {
            'cabeza': 'head',
            'ojo': 'eye',
            'oreja': 'ear',
            'nariz': 'nose',
            'boca': 'mouth',
            'mano': 'hand',
            'pie': 'foot',
            'brazo': 'arm',
            'pierna': 'leg',
            'dedo': 'finger,toe',
            'corazón': 'heart',
            'espalda': 'back',
            'cuello': 'neck'
        }
    },
    'food': {
        'name': 'Food',
        'items': {
            'pan': 'bread',
            'leche': 'milk',
            'agua': 'water',
            'carne': 'meat',
            'pollo': 'chicken',
            'arroz': 'rice',
            'huevo': 'egg',
            'queso': 'cheese',
            'fruta': 'fruit',
            'manzana': 'apple',
            'naranja': 'orange',
            'plátano': 'banana',
            'verdura': 'vegetable',
            'ensalada': 'salad'
        }
    },
    'clothing': {
        'name': 'Clothing',
        'items': {
            'camisa': 'shirt',
            'pantalón': 'pants,trousers',
            'zapato': 'shoe',
            'vestido': 'dress',
            'falda': 'skirt',
            'chaqueta': 'jacket',
            'abrigo': 'coat',
            'sombrero': 'hat',
            'calcetín': 'sock',
            'guante': 'glove',
            'bufanda': 'scarf'
        }
    }
}


MULTI_WORD_CATEGORIES = {'day', 'month', 'season', 'number'}

# Numbers in the 10-30 range for multi-word challenges
MULTI_WORD_NUMBER_RANGE = {
    'diez', 'once', 'doce', 'trece', 'catorce', 'quince',
    'dieciséis', 'diecisiete', 'dieciocho', 'diecinueve', 'veinte',
    'veintiuno', 'veintidós', 'veintitrés', 'veinticuatro', 'veinticinco',
    'veintiséis', 'veintisiete', 'veintiocho', 'veintinueve', 'treinta'
}


def get_all_categories() -> list[str]:
    """Get list of all category keys."""
    return list(VOCABULARY_CHALLENGES.keys())


def get_category_name(category: str) -> str:
    """Get display name for a category."""
    return VOCABULARY_CHALLENGES.get(category, {}).get('name', category)


def get_category_items(category: str) -> dict:
    """Get items for a category."""
    return VOCABULARY_CHALLENGES.get(category, {}).get('items', {})


def get_random_challenge(category: str, exclude_words: list[str] = None) -> dict | None:
    """Get a random word from a category.

    Returns dict with: word, translation, category, category_name
    """
    import random

    items = get_category_items(category)
    if not items:
        return None

    exclude_words = exclude_words or []
    available = [w for w in items.keys() if w not in exclude_words]

    if not available:
        return None

    word = random.choice(available)
    return {
        'word': word,
        'translation': items[word],
        'category': category,
        'category_name': get_category_name(category)
    }


def get_multi_word_number_items() -> dict:
    """Get number items filtered to 10-30 range for multi-word challenges."""
    items = get_category_items('number')
    return {w: t for w, t in items.items() if w in MULTI_WORD_NUMBER_RANGE}


def get_multi_word_challenge(category: str, exclude_words: list[str] = None, reverse: bool = False) -> dict | None:
    """Get a multi-word challenge with 4 random words from a category.

    Returns dict with: words (list of {word, translation}), category, category_name,
                       is_multi (True), is_reverse (bool)
    """
    import random

    if category == 'number':
        items = get_multi_word_number_items()
    else:
        items = get_category_items(category)

    if not items:
        return None

    exclude_words = exclude_words or []
    available = [w for w in items.keys() if w not in exclude_words]

    # For seasons (only 4 words), use all of them even if mastered
    if category == 'season' and len(available) < 4:
        available = list(items.keys())

    if len(available) < 4:
        # Not enough unmastered words; fill from all items
        remaining = [w for w in items.keys() if w not in available]
        random.shuffle(remaining)
        available.extend(remaining[:4 - len(available)])

    if len(available) < 4:
        return None

    selected = random.sample(available, 4)
    words = [{'word': w, 'translation': items[w]} for w in selected]

    return {
        'words': words,
        'category': category,
        'category_name': get_category_name(category),
        'is_multi': True,
        'is_reverse': reverse
    }
