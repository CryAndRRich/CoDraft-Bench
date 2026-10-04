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

    # The official class headings (Nice Classification, 12th edition), which the enrichment
    # prompt receives as each term's "Description". NICE_CLASS_MAP above is a short paraphrase
    # used in the model input ("Category: ..."); the two are not interchangeable. Gemini's
    # traces in term_attributes.csv quote these headings ("Clothing, footwear, headwear").
    NICE_CLASS_HEADINGS = {
        1: "Chemicals for use in industry, science and photography, as well as in agriculture, horticulture and forestry; unprocessed artificial resins, unprocessed plastics; fire extinguishing and fire prevention compositions; tempering and soldering preparations; substances for tanning animal skins and hides; adhesives for use in industry; putties and other paste fillers; compost, manures, fertilizers; biological preparations for use in industry and science.",
        2: "Paints, varnishes, lacquers; preservatives against rust and against deterioration of wood; colorants, dyes; inks for printing, marking and engraving; raw natural resins; metals in foil and powder form for use in painting, decorating, printing and art.",
        3: "Non-medicated cosmetics and toiletry preparations; non-medicated dentifrices; perfumery, essential oils; bleaching preparations and other substances for laundry use; cleaning, polishing, scouring and abrasive preparations.",
        4: "Industrial oils and greases, wax; lubricants; dust absorbing, wetting and binding compositions; fuels and illuminants; candles and wicks for lighting.",
        5: "Pharmaceuticals, medical and veterinary preparations; sanitary preparations for medical purposes; dietetic food and substances adapted for medical or veterinary use, food for babies; dietary supplements for human beings and animals; plasters, materials for dressings; material for stopping teeth, dental wax; disinfectants; preparations for destroying vermin; fungicides, herbicides.",
        6: "Common metals and their alloys, ores; metal materials for building and construction; transportable buildings of metal; non-electric cables and wires of common metal; small items of metal hardware; metal containers for storage or transport; safes.",
        7: "Machines, machine tools, power-operated tools; motors and engines, except for land vehicles; machine coupling and transmission components, except for land vehicles; agricultural implements, other than hand-operated hand tools; incubators for eggs; automatic vending machines.",
        8: "Hand tools and implements, hand-operated; cutlery; side arms, except firearms; razors.",
        9: "Scientific, research, navigation, surveying, photographic, cinematographic, audiovisual, optical, weighing, measuring, signalling, detecting, testing, inspecting, life-saving and teaching apparatus and instruments; apparatus and instruments for conducting, switching, transforming, accumulating, regulating or controlling the distribution or use of electricity; apparatus and instruments for recording, transmitting, reproducing or processing sound, images or data; recorded and downloadable media, computer software, blank digital or analogue recording and storage media; mechanisms for coin-operated apparatus; cash registers, calculating devices; computers and computer peripheral devices; diving suits, divers' masks, ear plugs for divers, nose clips for divers and swimmers, gloves for divers, breathing apparatus for underwater swimming; fire-extinguishing apparatus.",
        10: "Surgical, medical, dental and veterinary apparatus and instruments; artificial limbs, eyes and teeth; orthopaedic articles; suture materials; therapeutic and assistive devices adapted for persons with disabilities; massage apparatus; apparatus, devices and articles for nursing infants; sexual activity apparatus, devices and articles.",
        11: "Apparatus and installations for lighting, heating, cooling, steam generating, cooking, drying, ventilating, water supply and sanitary purposes.",
        12: "Vehicles; apparatus for locomotion by land, air or water.",
        13: "Firearms; ammunition and projectiles; explosives; fireworks.",
        14: "Precious metals and their alloys; jewellery, precious and semi-precious stones; horological and chronometric instruments.",
        15: "Musical instruments; music stands and stands for musical instruments; conductors' batons.",
        16: "Paper and cardboard; printed matter; bookbinding material; photographs; stationery and office requisites, except furniture; adhesives for stationery or household purposes; drawing materials and materials for artists; paintbrushes; instructional and teaching materials; plastic sheets, films and bags for wrapping and packaging; printers' type, printing blocks.",
        17: "Unprocessed and semi-processed rubber, gutta-percha, gum, asbestos, mica and substitutes for all these materials; plastics and resins in extruded form for use in manufacture; packing, stopping and insulating materials; flexible pipes, tubes and hoses, not of metal.",
        18: "Leather and imitations of leather; animal skins and hides; luggage and carrying bags; umbrellas and parasols; walking sticks; whips, harness and saddlery; collars, leashes and clothing for animals.",
        19: "Materials, not of metal, for building and construction; rigid pipes, not of metal, for building; asphalt, pitch, tar and bitumen; transportable buildings, not of metal; monuments, not of metal.",
        20: "Furniture, mirrors, picture frames; containers, not of metal, for storage or transport; unworked or semi-worked bone, horn, whalebone or mother-of-pearl; shells; meerschaum; yellow amber.",
        21: "Household or kitchen utensils and containers; cookware and tableware, except forks, knives and spoons; combs and sponges; brushes, except paintbrushes; brush-making materials; articles for cleaning purposes; unworked or semi-worked glass, except building glass; glassware, porcelain and earthenware.",
        22: "Ropes and string; nets; tents and tarpaulins; awnings of textile or synthetic materials; sails; sacks for the transport and storage of materials in bulk; padding, cushioning and stuffing materials, except of paper, cardboard, rubber or plastics; raw fibrous textile materials and substitutes therefor.",
        23: "Yarns and threads for textile use.",
        24: "Textiles and substitutes for textiles; household linen; curtains of textile or plastic.",
        25: "Clothing, footwear, headwear.",
        26: "Lace, braid and embroidery, and haberdashery ribbons and bows; buttons, hooks and eyes, pins and needles; artificial flowers; hair decorations; false hair.",
        27: "Carpets, rugs, mats and matting, linoleum and other materials for covering existing floors; wall hangings, not of textile.",
        28: "Games, toys and playthings; video game apparatus; gymnastic and sporting articles; decorations for Christmas trees.",
        29: "Meat, fish, poultry and game; meat extracts; preserved, frozen, dried and cooked fruits and vegetables; jellies, jams, compotes; eggs; milk, cheese, butter, yogurt and other milk products; oils and fats for food.",
        30: "Coffee, tea, cocoa and substitutes therefor; rice, pasta and noodles; tapioca and sago; flour and preparations made from cereals; bread, pastries and confectionery; chocolate; ice cream, sorbets and other edible ices; sugar, honey, treacle; yeast, baking-powder; salt, seasonings, spices, preserved herbs; vinegar, sauces and other condiments; ice (frozen water).",
        31: "Raw and unprocessed agricultural, aquacultural, horticultural and forestry products; raw and unprocessed grains and seeds; fresh fruits and vegetables, fresh herbs; natural plants and flowers; bulbs, seedlings and seeds for planting; live animals; foodstuffs and beverages for animals; malt.",
        32: "Beers; non-alcoholic beverages; mineral and aerated waters; fruit beverages and fruit juices; syrups and other non-alcoholic preparations for making beverages.",
        33: "Alcoholic beverages, except beers; alcoholic preparations for making beverages.",
        34: "Tobacco and tobacco substitutes; cigarettes and cigars; electronic cigarettes and oral vaporizers for smokers; smokers' articles; matches.",
        35: "Advertising; business management, organization and administration; office functions.",
        36: "Financial, monetary and banking services; insurance services; real estate services.",
        37: "Construction services; installation and repair services; mining extraction, oil and gas drilling.",
        38: "Telecommunications services.",
        39: "Transport; packaging and storage of goods; travel arrangement.",
        40: "Treatment of materials; recycling of waste and trash; air purification and treatment of water; printing services; food and drink preservation.",
        41: "Education; providing of training; entertainment; sporting and cultural activities.",
        42: "Scientific and technological services and research and design relating thereto; industrial analysis, industrial research and industrial design services; quality control and authentication services; design and development of computer hardware and software.",
        43: "Services for providing food and drink; temporary accommodation.",
        44: "Medical services; veterinary services; hygienic and beauty care for human beings or animals; agriculture, aquaculture, horticulture and forestry services.",
        45: "Legal services; security services for the physical protection of tangible property and individuals; dating services, online social networking services; funeral services; babysitting.",
    }

    CODRAFT_CONFIG = {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "batch_size": 30,
    }