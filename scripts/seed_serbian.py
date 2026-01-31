"""Serbian vocabulary seed data for Latin and Cyrillic scripts."""


def get_seed_data_sr_latn():
    """Serbian Latin vocabulary by category.

    Returns dict in same format as VOCABULARY_CHALLENGES:
    {category: {name: str, items: {serbian_word: english_alternatives}}}
    """
    return {
        'month': {
            'name': 'Mesec',
            'items': {
                'januar': 'january',
                'februar': 'february',
                'mart': 'march',
                'april': 'april',
                'maj': 'may',
                'jun': 'june',
                'jul': 'july',
                'avgust': 'august',
                'septembar': 'september',
                'oktobar': 'october',
                'novembar': 'november',
                'decembar': 'december'
            }
        },
        'season': {
            'name': 'Godišnje doba',
            'items': {
                'proleće': 'spring',
                'leto': 'summer',
                'jesen': 'autumn,fall',
                'zima': 'winter'
            }
        },
        'day': {
            'name': 'Dan u nedelji',
            'items': {
                'ponedeljak': 'monday',
                'utorak': 'tuesday',
                'sreda': 'wednesday',
                'četvrtak': 'thursday',
                'petak': 'friday',
                'subota': 'saturday',
                'nedelja': 'sunday'
            }
        },
        'number': {
            'name': 'Broj',
            'items': {
                'nula': '0,zero',
                'jedan': '1,one',
                'dva': '2,two',
                'tri': '3,three',
                'četiri': '4,four',
                'pet': '5,five',
                'šest': '6,six',
                'sedam': '7,seven',
                'osam': '8,eight',
                'devet': '9,nine',
                'deset': '10,ten',
                'jedanaest': '11,eleven',
                'dvanaest': '12,twelve',
                'trinaest': '13,thirteen',
                'četrnaest': '14,fourteen',
                'petnaest': '15,fifteen',
                'šesnaest': '16,sixteen',
                'sedamnaest': '17,seventeen',
                'osamnaest': '18,eighteen',
                'devetnaest': '19,nineteen',
                'dvadeset': '20,twenty',
                'dvadeset jedan': '21,twenty-one',
                'dvadeset dva': '22,twenty-two',
                'dvadeset tri': '23,twenty-three',
                'dvadeset četiri': '24,twenty-four',
                'dvadeset pet': '25,twenty-five',
                'dvadeset šest': '26,twenty-six',
                'dvadeset sedam': '27,twenty-seven',
                'dvadeset osam': '28,twenty-eight',
                'dvadeset devet': '29,twenty-nine',
                'trideset': '30,thirty'
            }
        },
        'color': {
            'name': 'Boja',
            'items': {
                'crvena': 'red',
                'plava': 'blue',
                'zelena': 'green',
                'žuta': 'yellow',
                'narandžasta': 'orange',
                'ljubičasta': 'purple',
                'roze': 'pink',
                'crna': 'black',
                'bela': 'white',
                'siva': 'gray,grey',
                'braon': 'brown',
                'zlatna': 'gold,golden'
            }
        },
        'family': {
            'name': 'Porodica',
            'items': {
                'majka': 'mother,mom',
                'otac': 'father,dad',
                'brat': 'brother',
                'sestra': 'sister',
                'sin': 'son',
                'ćerka': 'daughter',
                'deda': 'grandfather,grandpa',
                'baba': 'grandmother,grandma',
                'ujak': 'uncle',
                'tetka': 'aunt',
                'rođak': 'cousin',
                'muž': 'husband',
                'žena': 'wife'
            }
        },
        'animal': {
            'name': 'Životinja',
            'items': {
                'pas': 'dog',
                'mačka': 'cat',
                'ptica': 'bird',
                'riba': 'fish',
                'konj': 'horse',
                'krava': 'cow',
                'svinja': 'pig',
                'ovca': 'sheep',
                'piletina': 'chicken',
                'miš': 'mouse',
                'zec': 'rabbit',
                'medved': 'bear',
                'lav': 'lion',
                'slon': 'elephant',
                'majmun': 'monkey'
            }
        },
        'body': {
            'name': 'Deo tela',
            'items': {
                'glava': 'head',
                'oko': 'eye',
                'uvo': 'ear',
                'nos': 'nose',
                'usta': 'mouth',
                'ruka': 'hand',
                'stopalo': 'foot',
                'nadlaktica': 'arm',
                'noga': 'leg',
                'prst': 'finger,toe',
                'srce': 'heart',
                'leđa': 'back',
                'vrat': 'neck'
            }
        },
        'food': {
            'name': 'Hrana',
            'items': {
                'hleb': 'bread',
                'mleko': 'milk',
                'voda': 'water',
                'meso': 'meat',
                'piletina': 'chicken',
                'pirinač': 'rice',
                'jaje': 'egg',
                'sir': 'cheese',
                'voće': 'fruit',
                'jabuka': 'apple',
                'pomorandža': 'orange',
                'banana': 'banana',
                'povrće': 'vegetable',
                'salata': 'salad'
            }
        },
        'clothing': {
            'name': 'Odeća',
            'items': {
                'košulja': 'shirt',
                'pantalone': 'pants,trousers',
                'cipela': 'shoe',
                'haljina': 'dress',
                'suknja': 'skirt',
                'jakna': 'jacket',
                'kaput': 'coat',
                'šešir': 'hat',
                'čarapa': 'sock',
                'rukavica': 'glove',
                'šal': 'scarf'
            }
        }
    }


def get_seed_data_sr_cyrl():
    """Serbian Cyrillic vocabulary by category.

    Same words as Serbian Latin but written in Cyrillic script.
    """
    return {
        'month': {
            'name': 'Месец',
            'items': {
                'јануар': 'january',
                'фебруар': 'february',
                'март': 'march',
                'април': 'april',
                'мај': 'may',
                'јун': 'june',
                'јул': 'july',
                'август': 'august',
                'септембар': 'september',
                'октобар': 'october',
                'новембар': 'november',
                'децембар': 'december'
            }
        },
        'season': {
            'name': 'Годишње доба',
            'items': {
                'пролеће': 'spring',
                'лето': 'summer',
                'јесен': 'autumn,fall',
                'зима': 'winter'
            }
        },
        'day': {
            'name': 'Дан у недељи',
            'items': {
                'понедељак': 'monday',
                'уторак': 'tuesday',
                'среда': 'wednesday',
                'четвртак': 'thursday',
                'петак': 'friday',
                'субота': 'saturday',
                'недеља': 'sunday'
            }
        },
        'number': {
            'name': 'Број',
            'items': {
                'нула': '0,zero',
                'један': '1,one',
                'два': '2,two',
                'три': '3,three',
                'четири': '4,four',
                'пет': '5,five',
                'шест': '6,six',
                'седам': '7,seven',
                'осам': '8,eight',
                'девет': '9,nine',
                'десет': '10,ten',
                'једанаест': '11,eleven',
                'дванаест': '12,twelve',
                'тринаест': '13,thirteen',
                'четрнаест': '14,fourteen',
                'петнаест': '15,fifteen',
                'шеснаест': '16,sixteen',
                'седамнаест': '17,seventeen',
                'осамнаест': '18,eighteen',
                'деветнаест': '19,nineteen',
                'двадесет': '20,twenty',
                'двадесет један': '21,twenty-one',
                'двадесет два': '22,twenty-two',
                'двадесет три': '23,twenty-three',
                'двадесет четири': '24,twenty-four',
                'двадесет пет': '25,twenty-five',
                'двадесет шест': '26,twenty-six',
                'двадесет седам': '27,twenty-seven',
                'двадесет осам': '28,twenty-eight',
                'двадесет девет': '29,twenty-nine',
                'тридесет': '30,thirty'
            }
        },
        'color': {
            'name': 'Боја',
            'items': {
                'црвена': 'red',
                'плава': 'blue',
                'зелена': 'green',
                'жута': 'yellow',
                'наранџаста': 'orange',
                'љубичаста': 'purple',
                'розе': 'pink',
                'црна': 'black',
                'бела': 'white',
                'сива': 'gray,grey',
                'браон': 'brown',
                'златна': 'gold,golden'
            }
        },
        'family': {
            'name': 'Породица',
            'items': {
                'мајка': 'mother,mom',
                'отац': 'father,dad',
                'брат': 'brother',
                'сестра': 'sister',
                'син': 'son',
                'ћерка': 'daughter',
                'деда': 'grandfather,grandpa',
                'баба': 'grandmother,grandma',
                'ујак': 'uncle',
                'тетка': 'aunt',
                'рођак': 'cousin',
                'муж': 'husband',
                'жена': 'wife'
            }
        },
        'animal': {
            'name': 'Животиња',
            'items': {
                'пас': 'dog',
                'мачка': 'cat',
                'птица': 'bird',
                'риба': 'fish',
                'коњ': 'horse',
                'крава': 'cow',
                'свиња': 'pig',
                'овца': 'sheep',
                'пилетина': 'chicken',
                'миш': 'mouse',
                'зец': 'rabbit',
                'медвед': 'bear',
                'лав': 'lion',
                'слон': 'elephant',
                'мајмун': 'monkey'
            }
        },
        'body': {
            'name': 'Део тела',
            'items': {
                'глава': 'head',
                'око': 'eye',
                'уво': 'ear',
                'нос': 'nose',
                'уста': 'mouth',
                'рука': 'hand',
                'стопало': 'foot',
                'надлактица': 'arm',
                'нога': 'leg',
                'прст': 'finger,toe',
                'срце': 'heart',
                'леђа': 'back',
                'врат': 'neck'
            }
        },
        'food': {
            'name': 'Храна',
            'items': {
                'хлеб': 'bread',
                'млеко': 'milk',
                'вода': 'water',
                'месо': 'meat',
                'пилетина': 'chicken',
                'пиринач': 'rice',
                'јаје': 'egg',
                'сир': 'cheese',
                'воће': 'fruit',
                'јабука': 'apple',
                'поморанџа': 'orange',
                'банана': 'banana',
                'поврће': 'vegetable',
                'салата': 'salad'
            }
        },
        'clothing': {
            'name': 'Одећа',
            'items': {
                'кошуља': 'shirt',
                'панталоне': 'pants,trousers',
                'ципела': 'shoe',
                'хаљина': 'dress',
                'сукња': 'skirt',
                'јакна': 'jacket',
                'капут': 'coat',
                'шешир': 'hat',
                'чарапа': 'sock',
                'рукавица': 'glove',
                'шал': 'scarf'
            }
        }
    }
