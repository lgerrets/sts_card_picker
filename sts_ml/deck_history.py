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

def rebuild_deck(data : dict, draft_dataset : list):
    assert data["is_ascension_mode"]
    assert data["character_chosen"] == "IRONCLAD"
    assert data["ascension_level"] == 20
    # assert data["victory"]
    # assert len(data["path_per_floor"]) == 57

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

    node_to_index = defaultdict(lambda: 0)

    floor = -1
    for node in ["NEOW"] + data["path_per_floor"]:
        floor += 1
        index = node_to_index[node]
        node_to_index[node] += 1
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
    
def main():
    directory = "./Baalor400/Wins 201-400/IRONCLAD"
    files = os.listdir(directory)
    draft_dataset = []

    print(f"Analysing {len(files)} files")
    for idx, filename in enumerate(files):
        data = json.load(open(os.path.join(directory, filename), "r"))
        try:
            rebuild_deck(data, draft_dataset)
        except Exception as e:
            print(idx, filename)
            json.dump(data, open("./example_ironclad.run", "w"), indent=4)
            print("dumped to example_ironclad.run")
            raise e
    
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


