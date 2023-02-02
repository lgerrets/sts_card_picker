"""
Deprecated script

For .run files that come from mod Run History Plus
"""

print("Deprecated")

import json

from sts_ml.deck_history import add_card, upgrade_card, remove_card, card_name_in_deck, format_string, card_to_name, ALL_CARDS_FORMATTED

def rebuild_deck(data : dict, draft_dataset : list):
    deck = ["Defend_R"]*4 + ["Strike_R"]*5 + ["Bash", "AscendersBane"]
    modifiers = {}

    for card in data["neow_bonus_log"]["cardsObtained"]:
        add_card(deck, card)
    for card in data["neow_bonus_log"]["cardsUpgraded"]:
        upgrade_card(deck, card)
    for card in data["neow_bonus_log"]["cardsRemoved"]:
        remove_card(deck, card)
    for card in data["neow_bonus_log"]["cardsTransformed"]:
        remove_card(deck, card)

    floor = -1
    for node in ["NEOW"] + data["path_per_floor"]:
        floor += 1
        draft_data = {}

        relics_obtained = []
        for relic, relic_floor in data["relic_stats"]["obtain_stats"][0].items():
            if relic_floor == floor:
                relics_obtained.append(relic)
                if relic == "Omamori":
                    modifiers["omamori_counter"] = 2
                elif relic == "Pandora's Box":
                    for card in data["relic_stats"]["Pandora's Box"]:
                        add_card(deck, card)
                    for card_to_remove in ["Strike_R", "Defend_R"]:
                        while 1:
                            if not card_name_in_deck(deck, card_to_remove):
                                break
                            remove_card(deck, card_to_remove)
                elif relic == "Astrolabe":
                    for card in data["relic_stats"]["Astrolabe"]: # NOTE: Astrolabe does not record which cards were removed
                        add_card(deck, card)
                        upgrade_card(deck, card)
                # NOTE: tiny house does not record which card was upgraded

        draft_data["deck"] = [format_string(_) for _ in deck]
        cards_skipped = []
        cards_picked = []
        if floor not in [50, 51, 56]:
            got_at_least_one_reward = False
            for card_choice in data["card_choices"]:
                if card_choice["floor"] == floor:
                    got_at_least_one_reward = True
                    if card_choice["picked"] not in ["SKIP", "Singing Bowl"]:
                        add_card(deck, card_choice["picked"])
                        cards_picked.append(card_choice["picked"])
                    cards_skipped += card_choice["not_picked"]
        if len(cards_picked) or len(cards_skipped):
            draft_data["cards_picked"] = [format_string(_) for _ in cards_picked]
            draft_data["cards_skipped"] = [format_string(_) for _ in cards_skipped]
            draft_dataset.append(draft_data)
        
        if (node == "M") or (node == "E") or (node == "B") or (node == "NEOW"):
            pass
        elif node == "?":
            for event_choice in data["event_choices"]:
                if event_choice["floor"] == floor:
                    if "cards_obtained" in event_choice:
                        for card in event_choice["cards_obtained"]:
                            add_card(deck, card, modifiers)
                    if "cards_upgraded" in event_choice:
                        for card in event_choice["cards_upgraded"]:
                            upgrade_card(deck, card)
                    if "cards_removed" in event_choice:
                        for card in event_choice["cards_removed"]:
                            remove_card(deck, card)
                    if "cards_transformed" in event_choice:
                        for card in event_choice["cards_transformed"]:
                            remove_card(deck, card)
        elif node == "R":
            for campfire_choice in data["campfire_choices"]:
                if campfire_choice["floor"] == floor:
                    assert campfire_choice["key"] in ["REST", "SMITH", "RECALL", "DIG", "LIFT", "PURGE"], campfire_choice["key"]
                    if campfire_choice["key"] == "SMITH":
                        upgrade_card(deck, campfire_choice["data"])
                    elif campfire_choice["key"] == "PURGE":
                        remove_card(deck, campfire_choice["data"])
        elif node == "T":
            pass
        elif node == "$":
            for purged_card, purged_floor in zip(data["items_purged"], data["items_purged_floors"]):
                if purged_floor == floor:
                    remove_card(deck, purged_card)
            for purchase_card, purchase_floor in zip(data["items_purchased"], data["item_purchase_floors"]):
                if purchase_floor == floor:
                    add_card(deck, purchase_card)
        elif node is None:
            pass
        else:
            assert False, node
        
        if "Whetstone" in relics_obtained:
            for card in data["relic_stats"]["Whetstone"]:
                upgrade_card(deck, card)
        if "War Paint" in relics_obtained:
            for card in data["relic_stats"]["War Paint"]:
                upgrade_card(deck, card)

    master_deck = data["master_deck"]
    master_deck_set = set(master_deck)
    deck_set = set(deck)
    if deck_set != master_deck_set:
        print(f"{master_deck_set.difference(deck_set)} should have been added, {deck_set.difference(master_deck_set)} should have been removed")
    elif len(deck) != len(master_deck):
        print(f"Expected {len(master_deck)} != {len(deck)} cards")

def try_rebuild_deck(data, draft_dataset):
    if not data["is_ascension_mode"]: return
    if data["character_chosen"] != "IRONCLAD": return
    if data["ascension_level"] < 10: return
    if not data["victory"]: return

    try:
        rebuild_deck(data, draft_dataset)
    except Exception as e:
        json.dump(data, open("./example_ironclad.run", "w"), indent=4)
        print("dumped to example_ironclad.run")
        raise e

def main():
    """
    - Optimized for advanced .run files
    - Does not handle card formatting well
    - Baalor 400 dataset
    """
    directory = "./Baalor400/Wins 201-400/IRONCLAD"
    files = os.listdir(directory)
    draft_dataset = []

    print(f"Analysing {len(files)} files")
    for idx, filename in enumerate(files):
        data = json.load(open(os.path.join(directory, filename), "r"))
        try_rebuild_deck(data, draft_dataset)
    
    idx = len(draft_dataset) - 1
    while idx > -1:
        draft_data = draft_dataset[idx]
        do_remove = False
        for key, cards in draft_data.items():
            for card in cards:
                if card_to_name(card) not in ALL_CARDS_FORMATTED:
                    do_remove = True
        if do_remove:
            draft_dataset.pop(idx)
            print(f"Removed weird {draft_data}")
        idx -= 1
    json.dump(draft_dataset, open("./draft_dataset.data", "w"), indent=4)
    print(f"Dumped dataset of {len(draft_dataset)} draft decisions")

if __name__ == "__main__":
    main()
