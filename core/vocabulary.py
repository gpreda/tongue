"""Static vocabulary challenges for common categories."""

import random

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

# Category display names (challenge behavior config, not vocabulary data)
CATEGORY_DISPLAY_NAMES = {cat: data['name'] for cat, data in VOCABULARY_CHALLENGES.items()}

MULTI_WORD_CATEGORIES = {'day', 'month', 'season', 'number'}

# English numbers in the 10-30 range for multi-word challenges (language-agnostic)
MULTI_WORD_NUMBER_ENGLISH = {
    '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'
}

# Module-level storage for DB-backed vocabulary
_storage = None


def _english_key(alternatives: str) -> str:
    """Extract the canonical English key (first comma-separated value)."""
    return alternatives.split(',')[0].strip()


def _get_vocab_dict(language: str) -> dict:
    """Get the vocabulary challenges dict for a given language."""
    if language == 'es':
        return VOCABULARY_CHALLENGES
    if language == 'sr-latn':
        from scripts.seed_serbian import get_seed_data_sr_latn
        return get_seed_data_sr_latn()
    if language == 'sr-cyrl':
        from scripts.seed_serbian import get_seed_data_sr_cyrl
        return get_seed_data_sr_cyrl()
    return {}


def get_seed_data(language: str = 'es') -> list[dict]:
    """Generate seed data from vocabulary dict for a given language.

    Returns list of {category, english, word, language, alternatives} dicts.
    English key = first comma-separated value from the alternatives string.
    """
    vocab = _get_vocab_dict(language)
    items = []
    for category, data in vocab.items():
        for word, alternatives in data['items'].items():
            english = _english_key(alternatives)
            items.append({
                'category': category,
                'english': english,
                'word': word,
                'language': language,
                'alternatives': alternatives
            })
    return items


def init_storage(storage) -> None:
    """Set the storage backend for DB-backed vocabulary lookups."""
    global _storage
    _storage = storage


def get_all_categories(language: str = 'es') -> list[str]:
    """Get list of all category keys."""
    if _storage:
        try:
            cats = _storage.get_vocab_categories(language)
            if cats:
                return cats
        except Exception:
            pass
    vocab = _get_vocab_dict(language)
    return list(vocab.keys()) if vocab else list(VOCABULARY_CHALLENGES.keys())


def get_category_name(category: str) -> str:
    """Get display name for a category."""
    return CATEGORY_DISPLAY_NAMES.get(category, category)


def get_category_items(category: str, language: str = 'es') -> dict:
    """Get items for a category.

    Returns {target_word: english_alternatives} dict.
    When storage is set, queries DB; otherwise falls back to static dict.
    """
    if _storage:
        try:
            db_items = _storage.get_vocab_category_items(category, language)
            if db_items:
                return {item['word']: item['alternatives'] for item in db_items}
        except Exception:
            pass
    vocab = _get_vocab_dict(language)
    return vocab.get(category, {}).get('items', {})


def get_category_items_with_english(category: str, language: str = 'es') -> list[dict]:
    """Get items for a category with english keys included.

    Returns list of {english, word, alternatives} dicts.
    """
    if _storage:
        try:
            db_items = _storage.get_vocab_category_items(category, language)
            if db_items:
                return db_items
        except Exception:
            pass
    # Fallback to static dict
    vocab = _get_vocab_dict(language)
    data = vocab.get(category, {}).get('items', {})
    return [
        {'english': _english_key(alt), 'word': word, 'alternatives': alt}
        for word, alt in data.items()
    ]


def get_random_challenge(category: str, reverse: bool = False, language: str = 'es') -> dict | None:
    """Get a random word from a category.

    Returns dict with: word, translation, english, category, category_name, is_reverse
    """
    items_list = get_category_items_with_english(category, language)
    if not items_list:
        return None

    item = random.choice(items_list)
    return {
        'word': item['word'],
        'translation': item['alternatives'],
        'english': item['english'],
        'category': category,
        'category_name': get_category_name(category),
        'is_reverse': reverse
    }


def get_multi_word_number_items(language: str = 'es') -> list[dict]:
    """Get number items filtered to 10-30 range for multi-word challenges.

    Returns list of {english, word, alternatives} dicts.
    """
    items = get_category_items_with_english('number', language)
    return [item for item in items if item['english'] in MULTI_WORD_NUMBER_ENGLISH]


def get_multi_word_challenge(category: str, reverse: bool = False, language: str = 'es') -> dict | None:
    """Get a multi-word challenge with 4 random words from a category.

    Returns dict with: words (list of {word, translation, english}), category, category_name,
                       is_multi (True), is_reverse (bool)
    """
    if category == 'number':
        items_list = get_multi_word_number_items(language)
    else:
        items_list = get_category_items_with_english(category, language)

    if not items_list or len(items_list) < 4:
        return None

    selected = random.sample(items_list, 4)
    words = [{'word': item['word'], 'translation': item['alternatives'], 'english': item['english']}
             for item in selected]

    return {
        'words': words,
        'category': category,
        'category_name': get_category_name(category),
        'is_multi': True,
        'is_reverse': reverse
    }
