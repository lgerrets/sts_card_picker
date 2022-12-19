
import copy
from tkinter import ALL
import numpy as np
import json
from typing import List
import os, os.path
from collections import defaultdict

IRONCLAD_CARDS = ["Strike_R", "Defend_R", "Bash", "Anger", "Body Slam", "Clash", "Cleave", "Clothesline", "Headbutt", "Heavy Blade", "Iron Wave", "Perfected Strike", "Pommel Strike", "Sword Boomerang", "Thunderclap", "Twin Strike", "Wild Strike", "Blood for Blood", "Carnage", "Dropkick", "Hemokinesis", "Pummel", "Rampage", "Reckless Charge", "Searing Blow", "Sever Soul", "Uppercut", "Whirlwind", "Bludgeon", "Feed", "Fiend Fire", "Immolate", "Reaper", "Armaments", "Flex", "Havoc", "Shrug It Off", "True Grit", "Warcry", "Battle Trance", "Bloodletting", "Burning Pact", "Disarm", "Dual Wield", "Entrench", "Flame Barrier", "Ghostly Armor", "Infernal Blade", "Intimidate", "Power Through", "Rage", "Second Wind", "Seeing Red", "Sentinel", "Shockwave", "Spot Weakness", "Double Tap", "Exhume", "Impervious", "Limit Break", "Offering", "Combust", "Dark Embrace", "Evolve", "Feel No Pain", "Fire Breathing", "Inflame", "Metallicize", "Rupture", "Barricade", "Berserk", "Brutality", "Corruption", "Demon Form", "Juggernaut", ]
SILENT_CARDS = ["Strike", "Defend", "Neutralize", "Survivor", "Bane", "Dagger Spray", "Dagger Throw", "Flying Knee", "Poisoned Stab", "Quick Slash", "Slice", "Sneaky Strike", "Sucker Punch", "All-Out Attack", "Backstab", "Choke", "Dash", "Endless Agony", "Eviscerate", "Finisher", "Flechettes", "Heel Hook", "Masterful Stab", "Predator", "Riddle with Holes", "Skewer", "Die Die Die", "Glass Knife", "Grand Finale", "Unload", "Acrobatics", "Backflip", "Blade Dance", "Cloak and Dagger", "Deadly Poison", "Deflect", "Dodge and Roll", "Outmaneuver", "Piercing Wail", "Prepared", "Blur", "Bouncing Flask", "Calculated Gamble", "Catalyst", "Concentrate", "Crippling Cloud", "Distraction", "Escape Plan", "Expertise", "Leg Sweep", "Reflex", "Setup", "Tactician", "Terror", "Adrenaline", "Alchemize", "Bullet Time", "Burst", "Corpse Explosion", "Doppelganger", "Malaise", "Nightmare", "Phantasmal Killer", "Storm of Steel", "Accuracy", "Caltrops", "Footwork", "Infinite Blades", "Noxious Fumes", "Well-Laid Plans", "A Thousand Cuts", "After Image", "Envenom", "Tools of the Trade", "Wraith Form",]
DEFECT_CARDS = ["Strike", "Defend", "Dualcast", "Zap", "Ball Lightning", "Barrage", "Beam Cell", "Claw", "Cold Snap", "Compile Driver", "Go for the Eyes", "Rebound", "Streamline", "Sweeping Beam", "Blizzard", "Bullseye", "Doom and Gloom", "FTL", "Melter", "Rip and Tear", "Scrape", "Sunder", "All for One", "Core Surge", "Hyperbeam", "Meteor Strike", "Thunder Strike", "Charge Battery", "Coolheaded", "Hologram", "Leap", "Recursion", "Stack", "Steam Barrier", "TURBO", "Aggregate", "Auto-Shields", "Boot Sequence", "Chaos", "Chill", "Consume", "Darkness", "Double Energy", "Equilibrium", "Force Field", "Fusion", "Genetic Algorithm", "Glacier", "Overclock", "Recycle", "Reinforced Body", "Reprogram", "Skim", "Tempest", "White Noise", "Amplify", "Fission", "Multi-Cast", "Rainbow", "Reboot", "Seek", "Capacitor", "Defragment", "Heatsinks", "Hello World", "Loop", "Self Repair", "Static Discharge", "Storm", "Biased Cognition", "Buffer", "Creative AI", "Echo Form", "Electrodynamics", "Machine Learning", ]
WATCHER_CARDS = ["Strike", "Defend", "Eruption", "Vigilance", "Bowling Bash", "Consecrate", "Crush Joints", "Cut Through Fate", "Empty Fist", "Flurry of Blows", "Flying Sleeves", "Follow-Up", "Just Lucky", "Sash Whip", "Carve Reality", "Conclude", "Fear No Evil", "Reach Heaven", "Sands of Time", "Signature Move", "Talk to the Hand", "Tantrum", "Wallop", "Weave", "Wheel Kick", "Windmill Strike", "Brilliance", "Lesson Learned", "Ragnarok", "Crescendo", "Empty Body", "Evaluate", "Halt", "Pressure Points", "Prostrate", "Protect", "Third Eye", "Tranquility", "Collect", "Deceive Reality", "Empty Mind", "Foreign Influence", "Indignation", "Inner Peace", "Meditate", "Perseverance", "Pray", "Sanctity", "Simmering Fury", "Swivel", "Wave of the Hand", "Worship", "Wreath of Flame", "Alpha", "Blasphemy", "Conjure Blade", "Deus Ex Machina", "Judgment", "Omniscience", "Scrawl", "Spirit Shield", "Vault", "Wish", "Battle Hymn", "Fasting", "Foresight", "Like Water", "Mental Fortress", "Nirvana", "Rushdown", "Study", "Deva Form", "Devotion", "Establishment", "Master Reality", ]
COLORLESS_CARDS = ["Dramatic Entrance", "Flash of Steel", "Mind Blast", "Swift Strike", "HandOfGreed", "Bite", "Expunger", "Ritual Dagger", "Shiv", "Smite", "Through Violence", "Bandage Up", "Blind", "Dark Shackles", "Deep Breath", "Discovery", "Enlightenment", "Finesse", "Forethought", "Good Instincts", "Impatience", "Jack Of All Trades", "Madness", "Panacea", "PanicButton", "Purity", "Trip", "Apotheosis", "Chrysalis", "Master of Strategy", "Metamorphosis", "Secret Technique", "Secret Weapon", "The Bomb", "Thinking Ahead", "Transmutation", "Violence", "Apparition", "Beta", "Insight", "J.A.X.", "Miracle", "Safety", "Magnetism", "Mayhem", "Panache", "Sadistic Nature", "Omega", ]
CURSE_CARDS = ["Ascender's Bane", "Clumsy", "Curse of the Bell", "Decay", "Doubt", "Injury", "Necronomicurse", "Normality", "Pain", "Parasite", "Pride", "Regret", "Shame", "Writhe"]
MISC_COLLECTIBLE_CARDS = ["Ghostly", "RitualDagger", "AscendersBane"]

ALL_CARDS = IRONCLAD_CARDS + SILENT_CARDS + DEFECT_CARDS + WATCHER_CARDS + COLORLESS_CARDS + CURSE_CARDS + MISC_COLLECTIBLE_CARDS

def is_a_card(card : str):
    name = card_to_name(card)
    return (name in ALL_CARDS)

def format_string(card_name : str):
    card_name = card_name.replace(' ', '')
    card_name = card_name.lower()
    return card_name

ALL_CARDS_FORMATTED = [format_string(_) for _ in ALL_CARDS]

DUMMY_DICT = {}
DUMMY_LIST = []

def card_to_name(card : str) -> str:
    name = card.split("+")[0]
    return name

def card_to_n_upgrades(card : str) -> int:
    splits = card.split("+")
    if len(splits) > 1:
        return int(splits[1])
    else:
        return 0
    
def add_card(deck : List[str], card : str, modifiers=DUMMY_DICT):
    if is_a_card(card): # don't add potions and relics
        if (card in CURSE_CARDS) and (modifiers.get("omamori_counter", 0) > 0):
            modifiers["omamori_counter"] -= 1
            return
        deck.append(card)
    
def card_name_in_deck(deck : List[str], card : str):
    card_name = card_to_name(card)
    names = [card_to_name(deck_card) for deck_card in deck]
    return card_name in names

def remove_card(deck : List[str], card : str):
    if card in deck:
        deck.remove(card)
    else:
        card_name = card_to_name(card)
        names = [card_to_name(deck_card) for deck_card in deck]
        if card_name in names: # NOTE: the event Falling does not record the upgrade part of the card name
            idx = names.index(card_name)
            deck.pop(idx)
        else:
            assert is_a_card(card), card
            # assert False, f"{card} not in {deck}" # NOTE: curses obtained with Cursed Key are not recorded

def upgrade_card(deck : List[str], card : str):
    remove_card(deck, card)
    if "+" in card:
        name, n_upgrades = card.split("+")
        n_upgrades = int(n_upgrades)
    else:
        name = card
        n_upgrades = 0
    card = f"{name}+{n_upgrades+1}"
    add_card(deck, card)

class Certainty:
    (DEFINITELY, MAYBE) = range(2)

class CardInformation:
    (ATTACK, SKILL) = range(2)

class UndeterminedCard:
    def __init__(self, certainty : Certainty, card_info : CardInformation):
        self.certainty = certainty
        self.card_info = card_info

class FloorDelta:
    DEFINITELY_SOMETHING = format_string("DEFINITELY_SOMETHING")
    MAYBE_SOMETHING = format_string("MAYBE_SOMETHING")

    def __init__(
        self,
        floor : int,
        cards_added : List[str] = DUMMY_LIST,
        cards_removed : List[str] = DUMMY_LIST,
        cards_upgraded : List[str] = DUMMY_LIST,
        cards_transformed : List[str] = DUMMY_LIST,
        cards_skipped : List[str] = DUMMY_LIST,
        relics_added : List[str] = DUMMY_LIST,
        relics_removed : List[str] = DUMMY_LIST,
        gold_delta : int = 0,
        hp_delta : int = 0,
    ):
        self.floor = floor
        self.cards_added = [format_string(_) for _ in cards_added]
        self.cards_removed = [format_string(_) for _ in cards_removed]
        self.cards_upgraded = [format_string(_) for _ in cards_upgraded]
        self.cards_transformed = [format_string(_) for _ in cards_transformed]
        self.cards_skipped = [format_string(_) for _ in cards_skipped]
        self.relics_added = [format_string(_) for _ in relics_added]
        self.relics_removed = [format_string(_) for _ in relics_removed]
        self.gold_delta = gold_delta
        self.hp_delta = hp_delta

class History:
    def __init__(self):
        self.floor_deltas : List[FloorDelta] = []
    
    def add(self, floor_delta : FloorDelta):
        self.floor_deltas.append(floor_delta)

        if "pandora'sbox" in floor_delta.relics_added:
        # UndeterminedCard

                # elif relic == "Pandora's Box":
                #     for card in data["relic_stats"]["Pandora's Box"]:
                #         add_card(deck, card)
                #     for card_to_remove in ["Strike_R", "Defend_R"]:
                #         while 1:
                #             if not card_name_in_deck(deck, card_to_remove):
                #                 break
                #             remove_card(deck, card_to_remove)
                # elif relic == "Astrolabe":
                #     for card in data["relic_stats"]["Astrolabe"]: # NOTE: Astrolabe does not record which cards were removed
                #         add_card(deck, card)
                #         upgrade_card(deck, card)
    
                # NOTE: tiny house does not record which card was upgraded

        # if "Whetstone" in relics_obtained:
        #     for card in data["relic_stats"]["Whetstone"]:
        #         upgrade_card(deck, card)
        # if "War Paint" in relics_obtained:
        #     for card in data["relic_stats"]["War Paint"]:
        #         upgrade_card(deck, card)
    
    def wrap_up(self, master_deck : List[str]):
        master_deck = [format_string(_) for _ in master_deck]

        deck = ["Defend_R"]*4 + ["Strike_R"]*5 + ["Bash", "AscendersBane"]
        modifiers = {}

                # if relic == "Omamori":
                #     modifiers["omamori_counter"] = 2

def rebuild_deck_from_vanilla_run(data : dict, draft_dataset : list):
    history = History()

    ### TODO: Neow

    floor = 0
    for node in data["path_per_floor"]:
        floor += 1
        floor_delta = {}

        for card_choice in data["card_choices"]:
            if card_choice["floor"] == floor:
                got_at_least_one_reward = True
                if card_choice["picked"] not in ["SKIP", "Singing Bowl"]:
                    floor_delta["cards_added"] = floor_delta.get("cards_added", []) + [card_choice["picked"]]
                floor_delta["cards_skipped"] = floor_delta.get("cards_skipped", []) + [card_choice["not_picked"]]
        
        if (node == "M") or (node == "E") or (node == "B") or (node == "NEOW"):
            pass
        elif node == "?":
            for event_choice in data["event_choices"]:
                if event_choice["floor"] == floor:
                    if "cards_obtained" in event_choice:
                        for card in event_choice["cards_obtained"]:
                            floor_delta["cards_added"] = floor_delta.get("cards_added", []) + [card]
                    if "cards_upgraded" in event_choice:
                        for card in event_choice["cards_upgraded"]:
                            floor_delta["cards_ugraded"] = floor_delta.get("cards_ugraded", []) + [card]
                    if "cards_removed" in event_choice:
                        for card in event_choice["cards_removed"]:
                            floor_delta["cards_removed"] = floor_delta.get("cards_removed", []) + [card]
                    if "cards_transformed" in event_choice:
                        for card in event_choice["cards_transformed"]:
                            floor_delta["cards_transformed"] = floor_delta.get("cards_transformed", []) + [card]
        elif node == "R":
            for campfire_choice in data["campfire_choices"]:
                if campfire_choice["floor"] == floor:
                    assert campfire_choice["key"] in ["REST", "SMITH", "RECALL", "DIG", "LIFT", "PURGE"], campfire_choice["key"]
                    if campfire_choice["key"] == "SMITH":
                        floor_delta["cards_ugraded"] = floor_delta.get("cards_ugraded", []) + [campfire_choice["data"]]
                    elif campfire_choice["key"] == "PURGE":
                        floor_delta["cards_removed"] = floor_delta.get("cards_removed", []) + [campfire_choice["data"]]
        elif node == "T":
            pass
        elif node == "$":
            for purged_card, purged_floor in zip(data["items_purged"], data["items_purged_floors"]):
                if purged_floor == floor:
                    floor_delta["cards_removed"] = floor_delta.get("cards_removed", []) + [purged_card]
            for purchased_card, purchase_floor in zip(data["items_purchased"], data["item_purchase_floors"]):
                if purchase_floor == floor:
                    floor_delta["cards_added"] = floor_delta.get("cards_added", []) + [purchased_card]
        elif node is None:
            pass
        else:
            assert False, node
        
        history.add(floor_delta)

    master_deck = data["master_deck"]
    history.wrap_up(master_deck)


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
    directory = "./Baalor400/Wins 201-400/IRONCLAD"
    files = os.listdir(directory)
    draft_dataset = []

    print(f"Analysing {len(files)} files")
    for idx, filename in enumerate(files):
        data = json.load(open(os.path.join(directory, filename), "r"))
        try_rebuild_deck(data, draft_dataset)
    
    datas = json.load(open("./2019-05-31-00-53#1028.json", "r"))
    for data in datas:
        assert set(data.keys()) == {'event'}
        data = data["event"]
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

def main2():
    draft_dataset = []
    datas = json.load(open("./2019-05-31-00-53#1028.json", "r"))
    for data in datas:
        assert set(data.keys()) == {'event'}
        data = data["event"]
        try_rebuild_deck(data, draft_dataset)

if __name__ == "__main__":
    # main()
    main2()


