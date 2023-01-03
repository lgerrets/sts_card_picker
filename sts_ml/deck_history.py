
import copy
from tkinter import ALL
import numpy as np
import json
from typing import List
import os, os.path
from collections import defaultdict

IRONCLAD_CARDS = ["Strike_R", "Defend_R", "Bash", "Anger", "Body Slam", "Clash", "Cleave", "Clothesline", "Headbutt", "Heavy Blade", "Iron Wave", "Perfected Strike", "Pommel Strike", "Sword Boomerang", "Thunderclap", "Twin Strike", "Wild Strike", "Blood for Blood", "Carnage", "Dropkick", "Hemokinesis", "Pummel", "Rampage", "Reckless Charge", "Searing Blow", "Sever Soul", "Uppercut", "Whirlwind", "Bludgeon", "Feed", "Fiend Fire", "Immolate", "Reaper", "Armaments", "Flex", "Havoc", "Shrug It Off", "True Grit", "Warcry", "Battle Trance", "Bloodletting", "Burning Pact", "Disarm", "Dual Wield", "Entrench", "Flame Barrier", "Ghostly Armor", "Infernal Blade", "Intimidate", "Power Through", "Rage", "Second Wind", "Seeing Red", "Sentinel", "Shockwave", "Spot Weakness", "Double Tap", "Exhume", "Impervious", "Limit Break", "Offering", "Combust", "Dark Embrace", "Evolve", "Feel No Pain", "Fire Breathing", "Inflame", "Metallicize", "Rupture", "Barricade", "Berserk", "Brutality", "Corruption", "Demon Form", "Juggernaut", ]
SILENT_CARDS = ["Strike", "Defend", "Neutralize", "Survivor", "Bane", "Dagger Spray", "Dagger Throw", "Flying Knee", "Poisoned Stab", "Quick Slash", "Slice", "Sneaky Strike", "Sucker Punch", "All-Out Attack", "Backstab", "Choke", "Dash", "Endless Agony", "Eviscerate", "Finisher", "Flechettes", "Heel Hook", "Masterful Stab", "Predator", "Riddle with Holes", "Skewer", "Die Die Die", "Glass Knife", "Grand Finale", "Unload", "Acrobatics", "Backflip", "Blade Dance", "Cloak and Dagger", "Deadly Poison", "Deflect", "Dodge and Roll", "Outmaneuver", "Piercing Wail", "Prepared", "Blur", "Bouncing Flask", "Calculated Gamble", "Catalyst", "Concentrate", "Crippling Cloud", "Distraction", "Escape Plan", "Expertise", "Leg Sweep", "Reflex", "Setup", "Tactician", "Terror", "Adrenaline", "Alchemize", "Bullet Time", "Burst", "Corpse Explosion", "Doppelganger", "Malaise", "Nightmare", "Phantasmal Killer", "Storm of Steel", "Accuracy", "Caltrops", "Footwork", "Infinite Blades", "Noxious Fumes", "Well Laid Plans", "A Thousand Cuts", "After Image", "Envenom", "Tools of the Trade", "Wraith Form",]
DEFECT_CARDS = ["Strike", "Defend", "Dualcast", "Zap", "Ball Lightning", "Barrage", "Beam Cell", "Claw", "Cold Snap", "Compile Driver", "Go for the Eyes", "Rebound", "Streamline", "Sweeping Beam", "Blizzard", "Bullseye", "Doom and Gloom", "FTL", "Melter", "Rip and Tear", "Scrape", "Sunder", "All for One", "Core Surge", "Hyperbeam", "Meteor Strike", "Thunder Strike", "Charge Battery", "Coolheaded", "Hologram", "Leap", "Recursion", "Stack", "Steam Barrier", "TURBO", "Aggregate", "Auto Shields", "Boot Sequence", "Chaos", "Chill", "Consume", "Darkness", "Double Energy", "Equilibrium", "Force Field", "Fusion", "Genetic Algorithm", "Glacier", "Overclock", "Recycle", "Reinforced Body", "Reprogram", "Skim", "Tempest", "White Noise", "Amplify", "Fission", "Multi-Cast", "Rainbow", "Reboot", "Seek", "Capacitor", "Defragment", "Heatsinks", "Hello World", "Loop", "Self Repair", "Static Discharge", "Storm", "Biased Cognition", "Buffer", "Creative AI", "Echo Form", "Electrodynamics", "Machine Learning", ]
WATCHER_CARDS = ["Strike", "Defend", "Eruption", "Vigilance", "Bowling Bash", "Consecrate", "Crush Joints", "Cut Through Fate", "Empty Fist", "Flurry of Blows", "Flying Sleeves", "Follow Up", "Just Lucky", "Sash Whip", "Carve Reality", "Conclude", "Fear No Evil", "Reach Heaven", "Sands of Time", "Signature Move", "Talk to the Hand", "Tantrum", "Wallop", "Weave", "Wheel Kick", "Windmill Strike", "Brilliance", "Lesson Learned", "Ragnarok", "Crescendo", "Empty Body", "Evaluate", "Halt", "Pressure Points", "Prostrate", "Protect", "Third Eye", "Tranquility", "Collect", "Deceive Reality", "Empty Mind", "Foreign Influence", "Indignation", "Inner Peace", "Meditate", "Perseverance", "Pray", "Sanctity", "Simmering Fury", "Swivel", "Wave of the Hand", "Worship", "Wreath of Flame", "Alpha", "Blasphemy", "Conjure Blade", "Deus Ex Machina", "Judgment", "Omniscience", "Scrawl", "Spirit Shield", "Vault", "Wish", "Battle Hymn", "Fasting", "Foresight", "Like Water", "Mental Fortress", "Nirvana", "Rushdown", "Study", "Deva Form", "Devotion", "Establishment", "Master Reality", ]
COLORLESS_CARDS = ["Dramatic Entrance", "Flash of Steel", "Mind Blast", "Swift Strike", "HandOfGreed", "Bite", "Expunger", "Ritual Dagger", "Shiv", "Smite", "Through Violence", "Bandage Up", "Blind", "Dark Shackles", "Deep Breath", "Discovery", "Enlightenment", "Finesse", "Forethought", "Good Instincts", "Impatience", "Jack Of All Trades", "Madness", "Panacea", "PanicButton", "Purity", "Trip", "Apotheosis", "Chrysalis", "Master of Strategy", "Metamorphosis", "Secret Technique", "Secret Weapon", "The Bomb", "Thinking Ahead", "Transmutation", "Violence", "Apparition", "Beta", "Insight", "J.A.X.", "Miracle", "Safety", "Magnetism", "Mayhem", "Panache", "Sadistic Nature", "Omega", ]
CURSE_CARDS = ["Ascender s Bane", "Clumsy", "Curse of the Bell", "Decay", "Doubt", "Injury", "Necronomicurse", "Normality", "Pain", "Parasite", "Pride", "Regret", "Shame", "Writhe"]
MISC_COLLECTIBLE_CARDS = ["Ghostly"]
OLDER_CARDS = ["Conserve Battery", "Underhanded Strike", "Path To Victory", "Clear The Mind", "Steam", "Redo", "Lock On", "Judgement", "Gash"]

ALL_CARDS = IRONCLAD_CARDS + SILENT_CARDS + DEFECT_CARDS + WATCHER_CARDS + COLORLESS_CARDS + CURSE_CARDS + MISC_COLLECTIBLE_CARDS + OLDER_CARDS

class UnknownCard(Exception):
    def __init__(self, card, *args: object) -> None:
        super().__init__(*args)
        self.card = card

def is_a_card(card : str):
    name = card_to_name(card)
    return (name in ALL_CARDS)

def format_string(card_name : str):
    card_name = card_name.replace(' ', '')
    card_name = card_name.lower()
    for color in ['r', 'g', 'b', 'p']:
        card_name = card_name.replace(f'strike{color}', 'strike')
        card_name = card_name.replace(f'defend{color}', 'defend')
    return card_name

CURSE_CARDS_FORMATTED = [format_string(_) for _ in CURSE_CARDS]
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

def card_to_upgrade(card : str, delta_upgrades : int = 1):
    if "+" in card:
        name, n_upgrades = card.split("+")
        n_upgrades = int(n_upgrades)
    else:
        name = card
        n_upgrades = 0
    n_upgrades += delta_upgrades
    if n_upgrades > 0:
        card = f"{name}+{n_upgrades}"
    elif n_upgrades == 0:
        card = f"{name}"
    else:
        assert False, n_upgrades
    return card

def upgrade_card(deck : List[str], card : str):
    remove_card(deck, card)
    card = card_to_upgrade(card)
    add_card(deck, card)

class Certainty:
    (DEFINITELY, MAYBE) = range(2)

class CardInformation:
    (ATTACK, SKILL) = range(2)

class UndeterminedCard:
    def __init__(self, certainty : Certainty, card_info : CardInformation):
        self.certainty = certainty
        self.card_info = card_info

DEFINITELY_SOMETHING = format_string("DEFINITELY_SOMETHING")

class FloorDelta:
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
        event_name : str = None,
        player_choice : str = None,
        node : str = "",
        chest_opened : bool = False,
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
        self.event_name = event_name
        self.player_choice = player_choice
        self.node = node
        self.chest_opened = chest_opened

        for card in self.cards_added + self.cards_removed + self.cards_upgraded + self.cards_transformed + self.cards_skipped:
            if card_to_name(card) not in ALL_CARDS_FORMATTED + [DEFINITELY_SOMETHING]:
                raise UnknownCard(card)

        self.cards_removed_or_transformed = []

        self.unresolved_removed_cards = []
        self.unresolved_upgraded_cards = []
        self.unresolved_transformed_cards = []
        self.unresolved_removed_relics = []
        self.triggered = False
    
    def is_unresolved(self):
        ret = (len(self.unresolved_removed_cards) + len(self.unresolved_upgraded_cards) + len(self.unresolved_transformed_cards) + len(self.unresolved_removed_relics)) > 0
        ret |= DEFINITELY_SOMETHING in (self.cards_added + self.cards_removed + self.cards_upgraded + self.cards_transformed)
        return ret

class FloorState:
    def __init__(
        self,
        floor : int,
        cards : List[str] = None,
        relics : List[str] = None,
        gold : int = 0,
        hp : int = 0,
    ):
        self.floor = floor
        self.cards = [format_string(_) for _ in cards] if cards is not None else []
        self.relics = [format_string(_) for _ in relics] if relics is not None else []
        self.gold = gold
        self.hp = hp
        self.modifiers = {}

def dispatch_found_cards(deck : list, cards_to_find : list, not_found_cards : list):
    found_cards = []
    new_not_found_cards = []
    deck = copy.deepcopy(deck)
    for card in cards_to_find + not_found_cards:
        if card in deck:
            found_cards.append(card)
            deck.remove(card)
        else:
            new_not_found_cards.append(card)
    return found_cards, new_not_found_cards

class History:
    def __init__(self, initial_floor_state):
        self.last_resolved_floor_delta_idx = -1
        self.floor_deltas : List[FloorDelta] = []
        self.floor_states : List[FloorState] = [initial_floor_state] # states are previous to picking/skipping rewards at that floor
    
    def add(self, floor_delta : FloorDelta):
        floor_state = copy.deepcopy(self.floor_states[-1])
        self.floor_states.append(floor_state)
        self.floor_deltas.append(floor_delta)

        self.update_state_from_deltas(floor_state, floor_delta)
        
        if self.last_resolved_floor_delta_idx < floor_delta.floor:
            self.bakward_forward()
    
    def update_state_from_deltas(self, floor_state : FloorState, floor_delta : FloorDelta, correcting_floor_delta : FloorDelta = None):
        """
        state and delta to compute the next state

        This is where we should be reading from state.modifiers (eg omamori) and trigger on-obtained relics (eg pandora's box)
        """
        floor_state.floor += 1

        if floor_delta.is_unresolved() and (correcting_floor_delta is not None):
            while (DEFINITELY_SOMETHING in floor_delta.cards_added) and len(correcting_floor_delta.cards_added):
                idx = floor_delta.cards_added.index(DEFINITELY_SOMETHING)
                floor_delta.cards_added.pop(idx)
                floor_delta.cards_added.insert(idx, correcting_floor_delta.cards_added.pop(-1))
            while (DEFINITELY_SOMETHING in floor_delta.unresolved_removed_cards) and len(correcting_floor_delta.cards_removed):
                idx = floor_delta.unresolved_removed_cards.index(DEFINITELY_SOMETHING)
                floor_delta.unresolved_removed_cards.pop(idx)
                floor_delta.cards_removed.insert(idx, correcting_floor_delta.cards_removed.pop(-1))
            while (DEFINITELY_SOMETHING in floor_delta.unresolved_transformed_cards) and len(correcting_floor_delta.cards_transformed):
                idx = floor_delta.unresolved_transformed_cards.index(DEFINITELY_SOMETHING)
                floor_delta.unresolved_transformed_cards.pop(idx)
                floor_delta.cards_transformed.insert(idx, correcting_floor_delta.cards_transformed.pop(-1))
            while (DEFINITELY_SOMETHING in floor_delta.unresolved_removed_cards) and len(correcting_floor_delta.cards_removed_or_transformed):
                idx = floor_delta.unresolved_removed_cards.index(DEFINITELY_SOMETHING)
                floor_delta.unresolved_removed_cards.pop(idx)
                floor_delta.cards_removed.insert(idx, correcting_floor_delta.cards_removed_or_transformed.pop(-1))
            while (DEFINITELY_SOMETHING in floor_delta.unresolved_transformed_cards) and len(correcting_floor_delta.cards_removed_or_transformed):
                idx = floor_delta.unresolved_transformed_cards.index(DEFINITELY_SOMETHING)
                floor_delta.unresolved_transformed_cards.pop(idx)
                floor_delta.cards_transformed.insert(idx, correcting_floor_delta.cards_removed_or_transformed.pop(-1))
            while (DEFINITELY_SOMETHING in floor_delta.unresolved_upgraded_cards) and len(correcting_floor_delta.cards_upgraded):
                idx = floor_delta.unresolved_upgraded_cards.index(DEFINITELY_SOMETHING)
                floor_delta.unresolved_upgraded_cards.pop(idx)
                floor_delta.cards_upgraded.insert(idx, correcting_floor_delta.cards_upgraded.pop(-1))

        # here we do some processing specific to relics

        # for some relics, we have to do the update only once, because we update possibly non-empty list attributes ; typically, we can be sure that boss relics 
        if not floor_delta.triggered:
            if "whetstone" in floor_delta.relics_added:
                floor_delta.cards_upgraded += [DEFINITELY_SOMETHING] * 2

            if "warpaint" in floor_delta.relics_added:
                floor_delta.cards_upgraded += [DEFINITELY_SOMETHING] * 2
            
            if "pandora'sbox" in floor_delta.relics_added:
                transformed = [card for card in floor_state.cards if card in {'strike_r', 'defend_r'}]
                pandoras_n_transforms = len(transformed)
                floor_delta.cards_transformed = transformed
                floor_delta.cards_added = [DEFINITELY_SOMETHING for i in range(pandoras_n_transforms)]

            if "astrolabe" in floor_delta.relics_added:
                n_transforms = 3
                floor_delta.cards_transformed = [DEFINITELY_SOMETHING for i in range(n_transforms)]
                floor_delta.cards_added = [DEFINITELY_SOMETHING for i in range(n_transforms)]

            if "tinyhouse" in floor_delta.relics_added:
                floor_delta.cards_added += [DEFINITELY_SOMETHING]
                floor_delta.cards_upgraded += [DEFINITELY_SOMETHING]

            if "omamori" in floor_delta.relics_added:
                floor_state.modifiers["omamori_counter"] = 2

            if ("cursedkey" in floor_state.relics) and floor_delta.chest_opened:
                floor_delta.cards_added += [DEFINITELY_SOMETHING]
            
        if floor_delta.event_name is not None:
            if floor_delta.event_name == "Vampires":
                if "bite" in floor_delta.cards_added: # also remove all strikes
                    floor_delta.cards_removed = [card for card in self.floor_states[floor_delta.floor].cards if card.startswith('strike')]

            if (floor_delta.event_name == "Duplicator") and (floor_delta.player_choice == "Copied"):
                deck = self.floor_states[floor_delta.floor].cards
                duplicated_card_name = floor_delta.cards_added[0]
                if duplicated_card_name not in deck: # this event does not log the upgrade number of the copied card ; so we retrive a card that has the same base name in the deck ; NOTE this may not be accurate due to Searing Blow
                    for card in deck:
                        if card_to_name(card) == duplicated_card_name:
                            floor_delta.cards_added = [card]
                            break

        # add cards
        for card in floor_delta.cards_added:
            if (floor_state.modifiers.get("omamori_counter", 0) > 0) and (card in CURSE_CARDS_FORMATTED):
                floor_state.modifiers["omamori_counter"] -= 1
            else:
                floor_state.cards.append(card)

        # remove cards, retry to assign unresolved
        found_cards, new_not_found_cards = dispatch_found_cards(floor_state.cards, floor_delta.cards_removed, floor_delta.unresolved_removed_cards)
        floor_delta.cards_removed = found_cards
        floor_delta.unresolved_removed_cards = new_not_found_cards
        for card in found_cards:
            floor_state.cards.remove(card)
        
        # upgrade cards, retry to assign unresolved
        found_cards, new_not_found_cards = dispatch_found_cards(floor_state.cards, floor_delta.cards_upgraded, floor_delta.unresolved_upgraded_cards)
        floor_delta.cards_upgraded = found_cards
        floor_delta.unresolved_upgraded_cards = new_not_found_cards
        for card in found_cards:
            upgraded_card = card_to_upgrade(card)
            idx = floor_state.cards.index(card)
            floor_state.cards.pop(idx)
            floor_state.cards.insert(idx, upgraded_card)
        if correcting_floor_delta is not None:
            for card in new_not_found_cards:
                upgraded_card = card_to_upgrade(card)
                if upgraded_card in correcting_floor_delta.cards_added:
                    correcting_floor_delta.cards_added.remove(upgraded_card)
                    correcting_floor_delta.cards_added.append(card)
        
        # transform cards, retry to assign unresolved
        found_cards, new_not_found_cards = dispatch_found_cards(floor_state.cards, floor_delta.cards_transformed, floor_delta.unresolved_transformed_cards)
        floor_delta.cards_transformed = found_cards
        floor_delta.unresolved_transformed_cards = new_not_found_cards
        for card in found_cards:
            floor_state.cards.remove(card)
        
        # add relics
        floor_state.relics += floor_delta.relics_added

        if (self.last_resolved_floor_delta_idx == floor_delta.floor - 1) and (not floor_delta.is_unresolved()):
            self.last_resolved_floor_delta_idx += 1
        floor_delta.triggered = True

    def bakward_forward(self, delta_to_master : FloorDelta = None):
        if delta_to_master is None:
            accumulated_delta = FloorDelta(floor=None)

            # go backward through floors that have unresolved cards in their deltas (eg should have added a card but it was not in the deck), and collect them
            backward_delta_idx = len(self.floor_deltas)
            while backward_delta_idx > self.last_resolved_floor_delta_idx + 1:
                backward_delta_idx -= 1
                floor_delta = self.floor_deltas[backward_delta_idx]

                if not floor_delta.is_unresolved():
                    continue
                
                # append cards that are still undetermined
                accumulated_delta.cards_added += list(floor_delta.unresolved_removed_cards)
                accumulated_delta.cards_added += list(floor_delta.unresolved_upgraded_cards)
                accumulated_delta.cards_added += list(floor_delta.unresolved_transformed_cards)
        else:
            accumulated_delta = delta_to_master
        
        forward_delta_idx = len(self.floor_deltas)

        # resolve issues at latest floors first (probably better because it's closer to the ground truth master_deck ?)
        while forward_delta_idx > self.last_resolved_floor_delta_idx + 1:
            forward_delta_idx -= 1
            floor_delta = self.floor_deltas[forward_delta_idx]
            floor_state = self.floor_states[forward_delta_idx]
            next_floor_state = copy.deepcopy(floor_state)
            self.update_state_from_deltas(next_floor_state, floor_delta, accumulated_delta)
            if next_floor_state.cards != self.floor_states[forward_delta_idx + 1].cards: # noticed an update, let's propagate to later floor
                self.floor_states[forward_delta_idx + 1] = next_floor_state
                forward_pass_idx = forward_delta_idx + 1
                while forward_pass_idx < len(self.floor_states) - 1:
                    next_floor_state = copy.deepcopy(self.floor_states[forward_pass_idx])
                    floor_delta = self.floor_deltas[forward_pass_idx]
                    self.update_state_from_deltas(next_floor_state, floor_delta, accumulated_delta)
                    self.floor_states[forward_pass_idx + 1] = next_floor_state
                    forward_pass_idx += 1
    
    def wrap_up(self, master_deck : List[str]):
        master_deck = [format_string(_) for _ in master_deck]
        master_deck_copy = copy.deepcopy(master_deck)

        infered_deck = self.floor_states[-1].cards
        infered_deck_copy = copy.deepcopy(infered_deck)

        delta_to_master = FloorDelta(floor=None)

        idx = 0
        while idx < len(infered_deck_copy):
            card = infered_deck_copy[idx]
            if card in master_deck_copy:
                master_deck_copy.remove(card)
                infered_deck_copy.pop(idx)
            else:
                idx += 1
        idx = 0
        while idx < len(infered_deck_copy):
            card = infered_deck_copy[idx]
            if card_to_upgrade(card) in master_deck_copy:
                master_deck_copy.remove(card_to_upgrade(card))
                delta_to_master.cards_upgraded.append(card)
                infered_deck_copy.pop(idx)
            else:
                idx += 1
        while len(master_deck_copy):
            card = master_deck_copy.pop(0)
            delta_to_master.cards_added.append(card)
            if DEFINITELY_SOMETHING in infered_deck_copy:
                infered_deck_copy.remove(DEFINITELY_SOMETHING)
        while len(infered_deck_copy):
            card = infered_deck_copy.pop(0)
            if card != DEFINITELY_SOMETHING:
                delta_to_master.cards_removed_or_transformed.append(card)

        self.bakward_forward(delta_to_master)

        return delta_to_master

def filter_run(data : dict):
    if not data["is_ascension_mode"]: return False
    if data["character_chosen"] != "IRONCLAD": return False
    if data["ascension_level"] < 10: return False
    if not data["victory"]: return False
    return True

def rebuild_deck_from_vanilla_run(data : dict, run_rows : list):
    default_ret = False, FloorDelta(floor=0)
    
    if not filter_run(data):
        return default_ret

    initial_floor_state = FloorState(
        floor = 0,
        cards = ["Defend_R"]*4 + ["Strike_R"]*5 + ["Bash", "AscendersBane"],
    )

    history = History(initial_floor_state)

    floor = -1
    act = 1
    for node in ["NEOW"] + data["path_per_floor"]:
        floor += 1
        floor_delta_dict = {}

        keys_added = [] # TODO red blue green keys

        for card_choice in data["card_choices"]:
            if card_choice["floor"] == floor:
                got_at_least_one_reward = True
                if card_choice["picked"] not in ["SKIP", "Singing Bowl"]:
                    floor_delta_dict["cards_added"] = floor_delta_dict.get("cards_added", []) + [card_choice["picked"]]
                not_picked = card_choice["not_picked"] if isinstance(card_choice["not_picked"], list) else [card_choice["not_picked"]]
                floor_delta_dict["cards_skipped"] = floor_delta_dict.get("cards_skipped", []) + not_picked

        for relics_obtained_data in data["relics_obtained"]:
            if relics_obtained_data["floor"] == floor:
                relic = format_string(relics_obtained_data["key"])
                floor_delta_dict["relics_added"] = floor_delta_dict.get("relics_added", []) + [relic]
                if relic == "necronomicon": # NOTE: necronomicon
                    floor_delta_dict["cards_added"] = floor_delta_dict.get("cards_added", []) + ["necronomicurse"]

        if (node == "M") or (node == "E") or (node == "B"):
            if node == "B":
                if act <= 2:
                    if "picked" in data["boss_relics"][act-1]:
                        floor_delta_dict["relics_added"] = [data["boss_relics"][act-1]["picked"]]
                act += 1
        elif node == "NEOW":
            if data.get("neow_bonus", "") == "BOSS_RELIC":
                floor_delta_dict["relics_added"] = [data["relics"][0]]
            elif data.get("neow_bonus", "") == "ONE_RANDOM_RARE_CARD":
                floor_delta_dict["cards_added"] = [DEFINITELY_SOMETHING]
            elif data.get("neow_bonus", "") == "REMOVE_TWO":
                floor_delta_dict["cards_removed"] = [DEFINITELY_SOMETHING]*2
        elif node == "?":
            for event_choice in data["event_choices"]:
                if event_choice["floor"] == floor:
                    if "cards_obtained" in event_choice:
                        for card in event_choice["cards_obtained"]:
                            floor_delta_dict["cards_added"] = floor_delta_dict.get("cards_added", []) + [card]
                    if "cards_upgraded" in event_choice:
                        for card in event_choice["cards_upgraded"]:
                            floor_delta_dict["cards_upgraded"] = floor_delta_dict.get("cards_upgraded", []) + [card]
                    if "cards_removed" in event_choice:
                        for card in event_choice["cards_removed"]:
                            floor_delta_dict["cards_removed"] = floor_delta_dict.get("cards_removed", []) + [card]
                    if "cards_transformed" in event_choice:
                        for card in event_choice["cards_transformed"]:
                            floor_delta_dict["cards_transformed"] = floor_delta_dict.get("cards_transformed", []) + [card]
                    floor_delta_dict["event_name"] = event_choice["event_name"]
                    floor_delta_dict["player_choice"] = event_choice["player_choice"]
                    break
        elif node == "R":
            for campfire_choice in data["campfire_choices"]:
                if campfire_choice["floor"] == floor:
                    assert campfire_choice["key"] in ["REST", "SMITH", "RECALL", "DIG", "LIFT", "PURGE"], campfire_choice["key"]
                    if campfire_choice["key"] == "SMITH":
                        floor_delta_dict["cards_upgraded"] = floor_delta_dict.get("cards_upgraded", []) + [campfire_choice["data"]]
                    elif campfire_choice["key"] == "PURGE":
                        floor_delta_dict["cards_removed"] = floor_delta_dict.get("cards_removed", []) + [campfire_choice["data"]]
        elif node == "T":
            if len(floor_delta_dict.get("relics_added", [])) or len(keys_added):
                floor_delta_dict["chest_opened"] = True
        elif node == "$":
            for purged_card, purged_floor in zip(data["items_purged"], data["items_purged_floors"]):
                if purged_floor == floor:
                    floor_delta_dict["cards_removed"] = floor_delta_dict.get("cards_removed", []) + [purged_card]
            for purchased_item, purchase_floor in zip(data["items_purchased"], data["item_purchase_floors"]):
                if (purchase_floor == floor):
                    if (card_to_name(format_string(purchased_item)) in ALL_CARDS_FORMATTED):
                        floor_delta_dict["cards_added"] = floor_delta_dict.get("cards_added", []) + [purchased_item]
                    else:
                        floor_delta_dict["relics_added"] = floor_delta_dict.get("relics_added", []) + [purchased_item]
        elif node is None:
            pass
        else:
            assert False, node
        
        floor_delta = FloorDelta(floor=floor, node=node, **floor_delta_dict)
        
        history.add(floor_delta)

    master_deck = data["master_deck"]
    delta_to_master = history.wrap_up(master_deck)

    for floor_state, floor_delta in zip(history.floor_states, history.floor_deltas):
        if len(floor_delta.cards_skipped):
            data_row = {
                "deck": [card for card in floor_state.cards if card_to_name(card) != DEFINITELY_SOMETHING],
                "cards_picked": [card for card in floor_delta.cards_added if card_to_name(card) != DEFINITELY_SOMETHING],
                "cards_skipped": [card for card in floor_delta.cards_skipped if card_to_name(card) != DEFINITELY_SOMETHING],
            }
            run_rows.append(data_row)

    return (len(delta_to_master.cards_added) == 0) and (len(delta_to_master.cards_removed_or_transformed) == 0) and (len(delta_to_master.cards_upgraded) == 0), delta_to_master

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

def main2():
    draft_dataset = []
    # json_path = "./2019-05-31-00-53#1028.json"
    # json_path = "./november/november.json"
    # json_path = "./november/50000.json"
    # json_path = "./november/1000.json"
    json_path = "./november/50000_win_a20_ic.json"
    datas = json.load(open(json_path, "r"))

    datas = [data for data in datas if filter_run(data["event"])]

    total_diff = 0
    computed_run = 0
    for run_idx, data in enumerate(datas):
        assert set(data.keys()) == {'event'}
        data = data["event"]
        run_rows = []
        try:
            success, delta_to_master = rebuild_deck_from_vanilla_run(data, run_rows)
        except UnknownCard as e:
            print(f"{run_idx}: Unknown card {e.card}")
            continue
        draft_dataset += run_rows
        diff = len(delta_to_master.cards_added) + len(delta_to_master.cards_removed_or_transformed) + len(delta_to_master.cards_upgraded)
        total_diff += diff
        if diff or success:
            computed_run += 1
            json.dump(data, open("./example_vanilla.run", "w"), indent=4)
            if diff >= 1:
                print(f"{run_idx}: diff = {diff} ; to add = {delta_to_master.cards_added} ; to remove = {delta_to_master.cards_removed_or_transformed} ; to upgrade = {delta_to_master.cards_upgraded}")
    print(f"Diff score over {computed_run} runs = {total_diff}")

    filepath = "./november_dataset.data"
    json.dump(draft_dataset, open(filepath, "w"))
    print(f"Dumped dataset of {len(draft_dataset)} samples into {filepath}")

if __name__ == "__main__":
    # main()
    main2()

