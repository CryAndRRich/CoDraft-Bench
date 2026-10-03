class CONFIG_DATA:

    TARGET_SAMPLES = 3000
    MAX_SAMPLES_CLASS_0 = 4500
    MAX_LEN = 256

    NICE_CLASS_MAP = {
        1: "Chemicals, Industry, Science, Photography",
        2: "Paints, Varnishes, Lacquers, Preservatives",
        3: "Cosmetics, Soaps, Perfumery, Cleaning preparations",
        4: "Industrial oils, Greases, Fuels, Candles",
        5: "Pharmaceuticals, Medical, Hygiene, Dietetic food",
        6: "Common metals, Hardware, Metal building materials",
        7: "Machines, Motors, Engines, Tools",
        8: "Hand tools, Cutlery, Side arms, Razors",
        9: "Electronics, Computers, Software, Scientific instruments, Mobile phones",
        10: "Medical devices, Surgical instruments, Orthopedic",
        11: "Lighting, Heating, Cooling, Cooking appliances",
        12: "Vehicles, Transport apparatus, Cars, Bikes",
        13: "Firearms, Ammunition, Explosives",
        14: "Jewelry, Watches, Precious metals, Gemstones",
        15: "Musical instruments",
        16: "Paper, Printed matter, Stationery, Office requisites",
        17: "Rubber, Plastics, Insulation materials",
        18: "Leather, Bags, Wallets, Umbrellas",
        19: "Building materials (Non-metallic), Asphalt, Cement",
        20: "Furniture, Mirrors, Picture frames",
        21: "Household utensils, Kitchenware, Glassware, Combs",
        22: "Ropes, Strings, Nets, Tents, Sacks",
        23: "Yarns, Threads for textile",
        24: "Textiles, Bed covers, Table covers, Fabrics",
        25: "Clothing, Footwear, Headgear, Fashion",
        26: "Lace, Embroidery, Ribbons, Buttons",
        27: "Carpets, Rugs, Mats, Floor coverings",
        28: "Games, Toys, Sports equipment",
        29: "Meat, Fish, Poultry, Processed fruits & vegetables",
        30: "Coffee, Tea, Bread, Rice, Sweets, Spices",
        31: "Agricultural products, Fresh fruits, Vegetables, Seeds",
        32: "Beers, Mineral waters, Non-alcoholic drinks",
        33: "Alcoholic beverages (except beer), Wines, Spirits",
        34: "Tobacco, Smokers' articles, Matches",
        35: "Advertising, Business management, Retail services, Marketing",
        36: "Insurance, Financial affairs, Real estate, Banking",
        37: "Building construction, Repair, Installation services",
        38: "Telecommunications, Broadcasting",
        39: "Transport, Packaging, Storage of goods, Travel",
        40: "Treatment of materials, Recycling, Manufacturing services",
        41: "Education, Training, Entertainment, Sporting activities",
        42: "Technology services, Software development, IT consulting",
        43: "Food and drink services, Restaurants, Hotels",
        44: "Medical services, Hygiene, Beauty care (Spa/Salon)",
        45: "Legal services, Security services, Social services"
    }

    # One dedicated token per NICE class. create_patterns prepends the class token of the
    # unmasked side and [MASK] on the other, and the auxiliary head predicts the masked
    # side's class from the [MASK] hidden state. These must be registered with the
    # tokenizer (get_tokenizer(add_class_tokens=True)); otherwise SentencePiece splits
    # "[CLASS_11]" into ordinary subwords and the resize below it does nothing.
    CLASS_TOKENS = [f"[CLASS_{i}]" for i in range(1, 46)]

    CODRAFT_CONFIG = {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "batch_size": 30,
    }