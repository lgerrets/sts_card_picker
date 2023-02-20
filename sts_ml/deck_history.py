import os
from enum import Enum
import subprocess
from glob import glob
import datetime
import re
import copy
from tkinter import ALL
import numpy as np
import json
from typing import List, Tuple
import os, os.path
from collections import defaultdict

BUILD_VERSION_REGEX = re.compile('[0-9]{4}-[0-9]{2}-[0-9]{2}$')

# IC_BASE_CARDS = ["Strike_R", "Defend_R", "Bash"]
# IC_ATTACK_CARDS = ["Strike_R", "Bash", "Anger", "Body Slam", "Clash", "Cleave", "Clothesline", "Headbutt", "Heavy Blade", "Iron Wave", "Perfected Strike", "Pommel Strike", "Sword Boomerang", "Thunderclap", "Twin Strike", "Wild Strike", "Blood for Blood", "Carnage", "Dropkick", "Hemokinesis", "Pummel", "Rampage", "Reckless Charge", "Searing Blow", "Sever Soul", "Uppercut", "Whirlwind", "Bludgeon", "Feed", "Fiend Fire", "Immolate", "Reaper"]
# IC_SKILL_CARDS = ["Defend_R", "Armaments", "Flex", "Havoc", "Shrug It Off", "True Grit", "Warcry", "Battle Trance", "Bloodletting", "Burning Pact", "Disarm", "Dual Wield", "Entrench", "Flame Barrier", "Ghostly Armor", "Infernal Blade", "Intimidate", "Power Through", "Rage", "Second Wind", "Seeing Red", "Sentinel", "Shockwave", "Spot Weakness", "Double Tap", "Exhume", "Impervious", "Limit Break", "Offering"]
# IC_POWER_CARDS = ["Combust", "Dark Embrace", "Evolve", "Feel No Pain", "Fire Breathing", "Inflame", "Metallicize", "Rupture", "Barricade", "Berserk", "Brutality", "Corruption", "Demon Form", "Juggernaut"]
# IRONCLAD_CARDS = IC_BASE_CARDS + IC_ATTACK_CARDS + IC_SKILL_CARDS + IC_POWER_CARDS

# SILENT_CARDS = ["Strike", "Defend", "Neutralize", "Survivor", "Bane", "Dagger Spray", "Dagger Throw", "Flying Knee", "Poisoned Stab", "Quick Slash", "Slice", "Sneaky Strike", "Sucker Punch", "All-Out Attack", "Backstab", "Choke", "Dash", "Endless Agony", "Eviscerate", "Finisher", "Flechettes", "Heel Hook", "Masterful Stab", "Predator", "Riddle with Holes", "Skewer", "Die Die Die", "Glass Knife", "Grand Finale", "Unload", "Acrobatics", "Backflip", "Blade Dance", "Cloak and Dagger", "Deadly Poison", "Deflect", "Dodge and Roll", "Outmaneuver", "Piercing Wail", "Prepared", "Blur", "Bouncing Flask", "Calculated Gamble", "Catalyst", "Concentrate", "Crippling Cloud", "Distraction", "Escape Plan", "Expertise", "Leg Sweep", "Reflex", "Setup", "Tactician", "Terror", "Adrenaline", "Alchemize", "Bullet Time", "Burst", "Corpse Explosion", "Doppelganger", "Malaise", "Nightmare", "Phantasmal Killer", "Storm of Steel", "Accuracy", "Caltrops", "Footwork", "Infinite Blades", "Noxious Fumes", "Well Laid Plans", "A Thousand Cuts", "After Image", "Envenom", "Tools of the Trade", "Wraith Form",]
# DEFECT_CARDS = ["Strike", "Defend", "Dualcast", "Zap", "Ball Lightning", "Barrage", "Beam Cell", "Claw", "Cold Snap", "Compile Driver", "Go for the Eyes", "Rebound", "Streamline", "Sweeping Beam", "Blizzard", "Bullseye", "Doom and Gloom", "FTL", "Melter", "Rip and Tear", "Scrape", "Sunder", "All for One", "Core Surge", "Hyperbeam", "Meteor Strike", "Thunder Strike", "Charge Battery", "Coolheaded", "Hologram", "Leap", "Recursion", "Stack", "Steam Barrier", "TURBO", "Aggregate", "Auto Shields", "Boot Sequence", "Chaos", "Chill", "Consume", "Darkness", "Double Energy", "Equilibrium", "Force Field", "Fusion", "Genetic Algorithm", "Glacier", "Overclock", "Recycle", "Reinforced Body", "Reprogram", "Skim", "Tempest", "White Noise", "Amplify", "Fission", "Multi-Cast", "Rainbow", "Reboot", "Seek", "Capacitor", "Defragment", "Heatsinks", "Hello World", "Loop", "Self Repair", "Static Discharge", "Storm", "Biased Cognition", "Buffer", "Creative AI", "Echo Form", "Electrodynamics", "Machine Learning", ]
# WATCHER_CARDS = ["Strike", "Defend", "Eruption", "Vigilance", "Bowling Bash", "Consecrate", "Crush Joints", "Cut Through Fate", "Empty Fist", "Flurry of Blows", "Flying Sleeves", "Follow Up", "Just Lucky", "Sash Whip", "Carve Reality", "Conclude", "Fear No Evil", "Reach Heaven", "Sands of Time", "Signature Move", "Talk to the Hand", "Tantrum", "Wallop", "Weave", "Wheel Kick", "Windmill Strike", "Brilliance", "Lesson Learned", "Ragnarok", "Crescendo", "Empty Body", "Evaluate", "Halt", "Pressure Points", "Prostrate", "Protect", "Third Eye", "Tranquility", "Collect", "Deceive Reality", "Empty Mind", "Foreign Influence", "Indignation", "Inner Peace", "Meditate", "Perseverance", "Pray", "Sanctity", "Simmering Fury", "Swivel", "Wave of the Hand", "Worship", "Wreath of Flame", "Alpha", "Blasphemy", "Conjure Blade", "Deus Ex Machina", "Judgment", "Omniscience", "Scrawl", "Spirit Shield", "Vault", "Wish", "Battle Hymn", "Fasting", "Foresight", "Like Water", "Mental Fortress", "Nirvana", "Rushdown", "Study", "Deva Form", "Devotion", "Establishment", "Master Reality", ]
# COLORLESS_CARDS = ["Dramatic Entrance", "Flash of Steel", "Mind Blast", "Swift Strike", "HandOfGreed", "Bite", "Expunger", "Ritual Dagger", "Shiv", "Smite", "Through Violence", "Bandage Up", "Blind", "Dark Shackles", "Deep Breath", "Discovery", "Enlightenment", "Finesse", "Forethought", "Good Instincts", "Impatience", "Jack Of All Trades", "Madness", "Panacea", "PanicButton", "Purity", "Trip", "Apotheosis", "Chrysalis", "Master of Strategy", "Metamorphosis", "Secret Technique", "Secret Weapon", "The Bomb", "Thinking Ahead", "Transmutation", "Violence", "Apparition", "Beta", "Insight", "J.A.X.", "Miracle", "Safety", "Magnetism", "Mayhem", "Panache", "Sadistic Nature", "Omega", ]
# CURSE_CARDS = ["Ascender s Bane", "Clumsy", "Curse of the Bell", "Decay", "Doubt", "Injury", "Necronomicurse", "Normality", "Pain", "Parasite", "Pride", "Regret", "Shame", "Writhe"]
# MISC_COLLECTIBLE_CARDS = ["Ghostly"]
OLDER_CARDS = ["Conserve Battery", "Underhanded Strike", "Path To Victory", "Clear The Mind", "Steam", "Redo", "Lock On", "Judgement", "Gash"]

# ALL_CARDS = IRONCLAD_CARDS + SILENT_CARDS + DEFECT_CARDS + WATCHER_CARDS + COLORLESS_CARDS + CURSE_CARDS + MISC_COLLECTIBLE_CARDS + OLDER_CARDS

BASE_GAME_RELICS = ["Burning Blood", "Cracked Core", "PureWater", "Ring of the Snake", "Akabeko", "Anchor", "Ancient Tea Set", "Art of War", "Bag of Marbles", "Bag of Preparation", "Blood Vial", "TestModSTS:BottledPlaceholderRelic", "Bronze Scales", "Centennial Puzzle", "CeramicFish", "Damaru", "DataDisk", "Dream Catcher", "Happy Flower", "Juzu Bracelet", "Lantern", "MawBank", "MealTicket", "Nunchaku", "Oddly Smooth Stone", "Omamori", "Orichalcum", "Pen Nib", "TestModSTS:PlaceholderRelic2", "Potion Belt", "PreservedInsect", "Red Skull", "Regal Pillow", "TestModSTS:DefaultClickableRelic", "Smiling Mask", "Snake Skull", "Strawberry", "Boot", "Tiny Chest", "Toy Ornithopter", "Vajra", "War Paint", "Whetstone", "Blue Candle", "Bottled Flame", "Bottled Lightning", "Bottled Tornado", "Darkstone Periapt", "Yang", "Eternal Feather", "Frozen Egg 2", "Cables", "Gremlin Horn", "HornCleat", "InkBottle", "Kunai", "Letter Opener", "Matryoshka", "Meat on the Bone", "Mercury Hourglass", "Molten Egg 2", "Mummified Hand", "Ninja Scroll", "Ornamental Fan", "Pantograph", "Paper Crane", "Paper Frog", "Pear", "Question Card", "Self Forming Clay", "Shuriken", "Singing Bowl", "StrikeDummy", "Sundial", "Symbiotic Virus", "TeardropLocket", "The Courier", "Toxic Egg 2", "White Beast Statue", "Bird Faced Urn", "Calipers", "CaptainsWheel", "Champion Belt", "Charon's Ashes", "CloakClasp", "Dead Branch", "Du-Vu Doll", "Emotion Chip", "FossilizedHelix", "Gambling Chip", "Ginger", "Girya", "GoldenEye", "Ice Cream", "Incense Burner", "Lizard Tail", "Magic Flower", "Mango", "Old Coin", "Peace Pipe", "Pocketwatch", "Prayer Wheel", "Shovel", "StoneCalendar", "The Specimen", "Thread and Needle", "Tingsha", "Torii", "Tough Bandages", "TungstenRod", "Turnip", "Unceasing Top", "WingedGreaves", "Astrolabe", "Black Blood", "Black Star", "Busted Crown", "Calling Bell", "Coffee Dripper", "Cursed Key", "Ectoplasm", "Empty Cage", "FrozenCore", "Fusion Hammer", "HolyWater", "HoveringKite", "Inserter", "Mark of Pain", "Nuclear Battery", "Pandora's Box", "Philosopher's Stone", "Ring of the Serpent", "Runic Cube", "Runic Dome", "Runic Pyramid", "SacredBark", "SlaversCollar", "Snecko Eye", "Sozu", "Tiny House", "Velvet Choker", "VioletLotus", "WristBlade", "Bloody Idol", "CultistMask", "Enchiridion", "FaceOfCleric", "Golden Idol", "GremlinMask", "Mark of the Bloom", "MutagenicStrength", "Nloth's Gift", "NlothsMask", "Necronomicon", "NeowsBlessing", "Nilry's Codex", "Odd Mushroom", "Red Mask", "Spirit Poop", "SsserpentHead", "WarpedTongs", "Brimstone", "Cauldron", "Chemical X", "ClockworkSouvenir", "DollysMirror", "Frozen Eye", "HandDrill", "Lee's Waffle", "Medical Kit", "Melange", "Membership Card", "OrangePellets", "Orrery", "PrismaticShard", "Runic Capacitor", "Sling", "Strange Spoon", "TheAbacus", "Toolbox", "TwistedFunnel"]

# DECISIVE_RELICS = ["Akabeko", "Anchor", "Ancient Tea Set", "Art of War", "Bag of Marbles", "Bag of Preparation", "Bronze Scales", "Centennial Puzzle", "Happy Flower", "Lantern", "Nunchaku", "Oddly Smooth Stone", "Omamori", "Orichalcum", "Pen Nib", "Blue Candle", "Gremlin Horn", "HornCleat", "InkBottle", "Kunai", "Letter Opener", "Matryoshka", "Meat on the Bone", "Mummified Hand", "Ornamental Fan", "Pantograph", "Paper Frog", "Question Card", "Self Forming Clay", "Shuriken", "Singing Bowl", "StrikeDummy", "Sundial", "Symbiotic Virus", "TeardropLocket", "The Courier", "Toxic Egg 2", "White Beast Statue", "Bird Faced Urn", "Calipers", "CaptainsWheel", "Champion Belt", "Charon's Ashes", "CloakClasp", "Dead Branch", "Du-Vu Doll", "Emotion Chip", "FossilizedHelix", "Gambling Chip", "Ginger", "Girya", "Ice Cream", "Incense Burner", "Lizard Tail", "Magic Flower", "Pocketwatch", "StoneCalendar", "Thread and Needle", "Torii", "TungstenRod", "Unceasing Top", "Black Star", "Busted Crown", "Coffee Dripper", "Cursed Key", "Snecko Eye", "Sozu", "Velvet Choker", "Enchiridion", "GremlinMask", "Mark of the Bloom", "MutagenicStrength", "Necronomicon", "Nilry's Codex", "Odd Mushroom", "Red Mask", "Brimstone", "Chemical X", "ClockworkSouvenir", "Medical Kit", "OrangePellets", "Strange Spoon", ] # a heuristic subset of BASE_GAME_RELIC

BASE_GAME_POTIONS = ["BloodPotion", "Poison Potion", "FocusPotion", "BottledMiracle", "Block Potion", "Dexterity Potion", "Energy Potion", "Explosive Potion", "Fire Potion", "Strength Potion", "Swift Potion", "Weak Potion", "FearPotion", "AttackPotion", "SkillPotion", "PowerPotion", "ColorlessPotion", "SteroidPotion", "SpeedPotion", "BlessingOfTheForge", "TestModSTS:PlaceholderPotion", "ElixirPotion", "CunningPotion", "PotionOfCapacity", "StancePotion", "Regen Potion", "Ancient Potion", "LiquidBronze", "GamblersBrew", "EssenceOfSteel", "DuplicationPotion", "DistilledChaos", "LiquidMemories", "HeartOfIron", "GhostInAJar", "EssenceOfDarkness", "Ambrosia", "CultistPotion", "Fruit Juice", "SneckoOil", "FairyPotion", "SmokeBomb", "EntropicBrew"]

BASE_GAME_ATTACKS = ["Immolate", "Anger", "Cleave", "Reaper", "Iron Wave", "Reckless Charge", "Hemokinesis", "Body Slam", "Blood for Blood", "Clash", "Thunderclap", "Pummel", "Pommel Strike", "Twin Strike", "Bash", "Clothesline", "Rampage", "Sever Soul", "Whirlwind", "Fiend Fire", "Headbutt", "Wild Strike", "Heavy Blade", "Searing Blow", "Feed", "Bludgeon", "Perfected Strike", "Carnage", "Dropkick", "Sword Boomerang", "Uppercut", "Strike_R", "Grand Finale", "Glass Knife", "Underhanded Strike", "Dagger Spray", "Bane", "Unload", "Dagger Throw", "Choke", "Poisoned Stab", "Endless Agony", "Riddle With Holes", "Skewer", "Quick Slash", "Finisher", "Die Die Die", "Heel Hook", "Eviscerate", "Dash", "Backstab", "Slice", "Flechettes", "Masterful Stab", "Strike_G", "Neutralize", "Sucker Punch", "All Out Attack", "Flying Knee", "Predator", "Go for the Eyes", "Core Surge", "Ball Lightning", "Sunder", "Streamline", "Compile Driver", "All For One", "Blizzard", "Barrage", "Meteor Strike", "Rebound", "Melter", "Gash", "Sweeping Beam", "FTL", "Rip and Tear", "Lockon", "Scrape", "Beam Cell", "Cold Snap", "Strike_B", "Thunder Strike", "Hyperbeam", "Doom and Gloom", "Consecrate", "BowlingBash", "WheelKick", "FlyingSleeves", "JustLucky", "FlurryOfBlows", "TalkToTheHand", "WindmillStrike", "CarveReality", "Wallop", "SashWhip", "Eruption", "LessonLearned", "CutThroughFate", "ReachHeaven", "Ragnarok", "FearNoEvil", "SandsOfTime", "Conclude", "FollowUp", "Brilliance", "CrushJoints", "Tantrum", "Weave", "SignatureMove", "Strike_P", "EmptyFist", "Shiv", "Dramatic Entrance", "RitualDagger", "Bite", "Smite", "Expunger", "Hand of Greed", "Flash of Steel", "ThroughViolence", "Swift Strike", "Mind Blast"]
BASE_GAME_SKILLS = ["Spot Weakness", "Warcry", "Offering", "Exhume", "Power Through", "Dual Wield", "Flex", "Infernal Blade", "Intimidate", "True Grit", "Impervious", "Shrug It Off", "Flame Barrier", "Burning Pact", "Shockwave", "Seeing Red", "Disarm", "Armaments", "Havoc", "Rage", "Limit Break", "Entrench", "Defend_R", "Sentinel", "Battle Trance", "Second Wind", "Bloodletting", "Ghostly Armor", "Double Tap", "Crippling Poison", "Cloak And Dagger", "Storm of Steel", "Deadly Poison", "Leg Sweep", "Bullet Time", "Catalyst", "Tactician", "Blade Dance", "Deflect", "Night Terror", "Expertise", "Blur", "Setup", "Burst", "Acrobatics", "Doppelganger", "Adrenaline", "Calculated Gamble", "Escape Plan", "Terror", "Phantasmal Killer", "Malaise", "Reflex", "Survivor", "Defend_G", "Corpse Explosion", "Venomology", "Bouncing Flask", "Backflip", "Outmaneuver", "Concentrate", "Prepared", "PiercingWail", "Distraction", "Dodge and Roll", "Genetic Algorithm", "Zap", "Steam Power", "Fission", "Glacier", "Consume", "Redo", "Fusion", "Amplify", "Reboot", "Aggregate", "Chaos", "Stack", "Seek", "Rainbow", "Chill", "BootSequence", "Coolheaded", "Tempest", "Turbo", "Undo", "Force Field", "Darkness", "Double Energy", "Reinforced Body", "Conserve Battery", "Defend_B", "Dualcast", "Auto Shields", "Reprogram", "Hologram", "Leap", "Recycle", "Skim", "White Noise", "Multi-Cast", "Steam", "DeusExMachina", "Vengeance", "Sanctity", "Halt", "Protect", "Indignation", "ThirdEye", "ForeignInfluence", "Crescendo", "SpiritShield", "ClearTheMind", "EmptyBody", "WreathOfFlame", "Collect", "InnerPeace", "Omniscience", "Wish", "DeceiveReality", "Alpha", "Vault", "Scrawl", "Blasphemy", "Defend_P", "WaveOfTheHand", "Meditate", "Perseverance", "Swivel", "Worship", "Vigilance", "PathToVictory", "Evaluate", "EmptyMind", "Prostrate", "ConjureBlade", "Judgement", "Pray", "Beta", "Dark Shackles", "J.A.X.", "Panic Button", "Trip", "FameAndFortune", "Impatience", "The Bomb", "Insight", "Miracle", "Blind", "Bandage Up", "Secret Technique", "Deep Breath", "Violence", "Secret Weapon", "Apotheosis", "Forethought", "Enlightenment", "Purity", "Panacea", "Transmutation", "Ghostly", "Chrysalis", "Discovery", "Finesse", "Master of Strategy", "Good Instincts", "Jack Of All Trades", "Safety", "Metamorphosis", "Thinking Ahead", "Madness", "Apparition"]
BASE_GAME_POWERS = ["Inflame", "Brutality", "Juggernaut", "Berserk", "Metallicize", "Combust", "Dark Embrace", "Barricade", "Feel No Pain", "Corruption", "Rupture", "Demon Form", "Fire Breathing", "Evolve", "A Thousand Cuts", "After Image", "Tools of the Trade", "Caltrops", "Wraith Form v2", "Envenom", "Well Laid Plans", "Noxious Fumes", "Infinite Blades", "Accuracy", "Footwork", "Storm", "Hello World", "Creative AI", "Echo Form", "Self Repair", "Loop", "Static Discharge", "Heatsinks", "Buffer", "Electrodynamics", "Machine Learning", "Biased Cognition", "Capacitor", "Defragment", "Wireheading", "BattleHymn", "DevaForm", "LikeWater", "Establishment", "Fasting2", "Adaptation", "MentalFortress", "Study", "Devotion", "Nirvana", "MasterReality", "Sadistic Nature", "LiveForever", "BecomeAlmighty", "Panache", "Mayhem", "Magnetism", "Omega"]
BASE_GAME_CURSES = ["Regret", "Writhe", "AscendersBane", "Decay", "Necronomicurse", "Pain", "Parasite", "Doubt", "Injury", "Clumsy", "CurseOfTheBell", "Normality", "Pride", "Shame"]

BASE_GAME_ENEMIES = ["Blue Slaver", "Cultist", "Jaw Worm", "Looter", "2 Louse", "Small Slimes", "Gremlin Gang", "Red Slaver", "Large Slime", "Exordium Thugs", "Exordium Wildlife", "3 Louse", "2 Fungi Beasts", "Lots of Slimes", "Gremlin Nob", "Lagavulin", "3 Sentries", "Lagavulin Event", "The Mushroom Lair", "The Guardian", "Hexaghost", "Slime Boss", "2 Thieves", "3 Byrds", "Chosen", "Shell Parasite", "Spheric Guardian", "Cultist and Chosen", "3 Cultists", "4 Byrds", "Chosen and Byrds", "Sentry and Sphere", "Snake Plant", "Snecko", "Centurion and Healer", "Shelled Parasite and Fungi", "Book of Stabbing", "Gremlin Leader", "Slavers", "Masked Bandits", "Colosseum Slavers", "Colosseum Nobs", "Automaton", "Champ", "Collector", "3 Darklings", "3 Shapes", "Orb Walker", "Transient", "Reptomancer", "Spire Growth", "Maw", "4 Shapes", "Sphere and 2 Shapes", "Jaw Worm Horde", "Snecko and Mystics", "Writhing Mass", "2 Orb Walkers", "Nemesis", "Giant Head", "Mysterious Sphere", "Mind Bloom Boss Battle", "Time Eater", "Awakened One", "Donu and Deca", "The Heart", "Shield and Spear", "The Eyes", "Apologetic Slime", "Flame Bruiser 1 Orb", "Flame Bruiser 2 Orb", "Slaver and Parasite", "Snecko and Mystics"]
BOSS_ENEMIES = ["The Guardian", "Hexaghost", "Slime Boss", "Automaton", "Champ", "Collector", "Time Eater", "Awakened One", "Donu and Deca", "The Heart"]

ALL_CARDS = BASE_GAME_ATTACKS + BASE_GAME_SKILLS + BASE_GAME_POWERS + BASE_GAME_CURSES + OLDER_CARDS
PAD_TOKEN = "PAD_TOKEN"

Color = Enum('Color', ['RED', 'GREEN', 'BLUE', 'PURPLE', 'COLORLESS', 'STATUS', 'CURSE'])
Rarity = Enum('Rarity', ['STARTER', 'COMMON', 'UNCOMMON', 'RARE'])
CardType = Enum('CardType', ['ATTACK', 'SKILL', 'POWER'])

COLORLESS_CARDS = ["Dramatic Entrance", "Flash of Steel", "Mind Blast", "Swift Strike", "Hand of Greed", "Bandage Up", "Blind", "Dark Shackles", "Deep Breath", "Discovery", "Enlightenment", "Finesse", "Forethought", "Good Instincts", "Impatience", "Jack Of All Trades", "Madness", "Panacea", "Panic Button", "Purity", "Trip", "Apotheosis", "Chrysalis", "Master of Strategy", "Metamorphosis", "Secret Technique", "Secret Weapon", "The Bomb", "Thinking Ahead", "Transmutation", "Violence", "Magnetism", "Mayhem", "Panache", "Sadistic Nature"]
assert set(COLORLESS_CARDS).issubset(set(ALL_CARDS)), set(COLORLESS_CARDS).difference(set(ALL_CARDS))

class UnknownCard(Exception):
    def __init__(self, card, *args: object) -> None:
        super().__init__(*args)
        self.card = card

def is_a_card(card : str):
    name = card_to_name(card)
    return (name in ALL_CARDS)

def format_string(card_name : str):
    if isinstance(card_name, UndeterminedCard):
        return card_name
    card_name = card_name.replace(' ', '')
    card_name = card_name.lower()
    # card_name = card_name.replace("'", '')
    # card_name = card_name.replace("-", '')
    # card_name = card_name.replace(".", '')
    # for color in ['r', 'g', 'b', 'p']:
    #     card_name = card_name.replace(f'strike_{color}', 'strike')
    #     card_name = card_name.replace(f'defend_{color}', 'defend')
    return card_name

class UndeterminedCard:
    @staticmethod
    def list_has_undetermined(card_list : List) -> int:
        for idx, card in enumerate(card_list):
            if isinstance(card, UndeterminedCard):
                return idx
        return -1
    
    def find_matching_undetermined(card_list : List, card_str : str):
        for idx, card in enumerate(card_list):
            if isinstance(card, UndeterminedCard):
                if card.match(card_str):
                    return idx
        return -1

    def __init__(
        self,
        color : Color = None,
        rarity : Rarity = None,
        card_type : CardType = None,
        upgraded : bool = None,
    ):
        self.color = color
        self.rarity = rarity
        self.card_type = card_type
        self.upgraded = upgraded
    
    def match(self, card : str):
        if isinstance(card, UndeterminedCard):
            return False

        card_name = card_to_name(card)
        
        if self.color is not None:
            if (self.color in COLOR_TO_CARDS) and (card_name not in COLOR_TO_CARDS[self.color]):
                return False
        
        if self.rarity is not None:
            pass # TODO

        if self.card_type is not None:
            if card_name not in CARD_TYPE_TO_CARDS[self.card_type]:
                return False
        
        if self.upgraded is not None:
            n_upgrades = card_to_n_upgrades(card)
            if self.upgraded != (n_upgrades > 0):
                return False
        
        return True
    
    def find_match(self, card_list : List[str], reversed_order=True):
        it = list(enumerate(card_list))
        if reversed_order:
            it = reversed(it)
        for idx, card in it:
            if self.match(card):
                return idx
        return -1
    
    def repr_attribute(self, ret_str, attr, opened_bracket):
        if opened_bracket:
            ret_str += "/"
        else:
            ret_str += "("
        ret_str += str(attr)
        opened_bracket = True
        return ret_str, opened_bracket

    def __repr__(self) -> str:
        ret = "UndeterminedCard"
        opened_bracket = False

        if self.color is not None:
            ret, opened_bracket = self.repr_attribute(ret, self.color, opened_bracket)
        
        if self.rarity is not None:
            ret, opened_bracket = self.repr_attribute(ret, self.rarity, opened_bracket)

        if self.card_type is not None:
            ret, opened_bracket = self.repr_attribute(ret, self.card_type, opened_bracket)
            
        if self.upgraded is not None:
            ret, opened_bracket = self.repr_attribute(ret, self.upgraded, opened_bracket)
        
        if opened_bracket:
            ret += ")"

        return ret
DEFINITELY_SOMETHING = format_string("DEFINITELY_SOMETHING")

ATTACK_CARDS_FORMATTED = [format_string(_) for _ in BASE_GAME_ATTACKS]
SKILL_CARDS_FORMATTED = [format_string(_) for _ in BASE_GAME_SKILLS]
POWER_CARDS_FORMATTED = [format_string(_) for _ in BASE_GAME_POWERS]
COLORLESS_CARDS_FORMATTED = [format_string(_) for _ in COLORLESS_CARDS]
CURSE_CARDS_FORMATTED = [format_string(_) for _ in BASE_GAME_CURSES]
ALL_CARDS_FORMATTED = [format_string(_) for _ in ALL_CARDS]
ALL_CARDS_FORMATTED_SET = set(ALL_CARDS_FORMATTED)
ALL_RELICS_FORMATTED = [format_string(_) for _ in BASE_GAME_RELICS]
# DECISIVE_RELICS_FORMATTED = [format_string(_) for _ in DECISIVE_RELICS]

COLOR_TO_CARDS = {
    Color.COLORLESS: COLORLESS_CARDS_FORMATTED,
    Color.CURSE: CURSE_CARDS_FORMATTED,
}

CARD_TYPE_TO_CARDS = {
    CardType.ATTACK: ATTACK_CARDS_FORMATTED,
    CardType.SKILL: SKILL_CARDS_FORMATTED,
    CardType.POWER: POWER_CARDS_FORMATTED,
}

DUMMY_DICT = {}
DUMMY_LIST = []

def card_to_name(card : str) -> str:
    if isinstance(card, UndeterminedCard):
        return card
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
        if (card in CURSE_CARDS_FORMATTED) and (modifiers.get("omamori_counter", 0) > 0):
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

def valid_build_number(string, character):
    pattern = re.compile('[0-9]{4}-[0-9]{2}-[0-9]{2}$')
    if pattern.match(string):
        m = re.search('(.+)-(.+)-(.+)', string)
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))

        date = datetime.date(year, month, day)
        if date >= datetime.date(2020, 1, 16):
            return True
        elif character in ['IRONCLAD', 'THE_SILENT', 'DEFECT'] and date >= datetime.date(2019, 1, 23):
            return True

    return False

def is_bad_data(data : dict):
    verbose = 1

    # Corrupted files
    necessary_fields = ['damage_taken', 'event_choices', 'card_choices', 'relics_obtained', 'campfire_choices',
                        'items_purchased', 'item_purchase_floors', 'items_purged', 'items_purged_floors',
                        'character_chosen', 'boss_relics', 'floor_reached', 'master_deck', 'relics']
    for field in necessary_fields:
        if field not in data:
            if verbose:
                print(f'File missing field: {field}')
            return True

    # Modded games
    key = 'character_chosen'
    if key not in data or data[key] not in ['IRONCLAD', 'THE_SILENT', 'DEFECT', 'WATCHER']:
        if verbose:
            print(f'Modded character: {data[key]}')
        return True

    # Watcher files since full release of watcher (v2.0) and ironclad, silent, defect since v1.0
    # key = 'build_version'
    # if key not in data or valid_build_number(data[key], data['character_chosen']) is False:
    #     return True

    # key = 'relics'
    # if key not in data or set(data[key]).issubset(BASE_GAME_RELICS) is False:
    #     return True

    key = 'master_deck'
    card_names = set([card_to_name(format_string(card)) for card in data[key]])
    if key not in data or card_names.issubset(ALL_CARDS_FORMATTED_SET) is False:
        if verbose:
            print(f'Modded file. Cards: {card_names - ALL_CARDS_FORMATTED_SET}')
        return True

    # Non standard runs
    key = 'is_trial'
    if key not in data or data[key] is True:
        return True

    key = 'is_daily'
    if key not in data or data[key] is True:
        return True

    key = 'daily_mods'
    if key in data:
        return True

    key = 'chose_seed'
    if key not in data or data[key] is True:
        return True

    if len(data["gold_per_floor"]) and data["gold_per_floor"][0] >= 500: return True

    # Endless mode
    key = 'is_endless'
    if key not in data or data[key] is True:
        return True

    key = 'circlet_count'
    if key not in data or data[key] > 0:
        return True

    key = 'floor_reached'
    if key not in data or data[key] > 60:
        return True

    # Really bad players or give ups
    key = 'floor_reached'
    if key not in data or data[key] < 4:
        return True

    key = 'score'
    if key not in data or data[key] < 10:
        return True

    key = 'player_experience'
    if key not in data or data[key] < 100:
        return True

    return False

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
        maybe_got_parasite : bool = False,
        act_delta : int = 0,
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
        self.maybe_got_parasite = maybe_got_parasite
        self.act_delta = act_delta

        for card in self.cards_added + self.cards_removed + self.cards_upgraded + self.cards_transformed + self.cards_skipped:
            if not isinstance(card, UndeterminedCard):
                if card_to_name(card) not in ALL_CARDS_FORMATTED_SET:
                    raise UnknownCard(card)

        self.cards_removed_or_transformed = []

        self.unresolved_removed_cards = []
        self.unresolved_upgraded_cards = []
        self.unresolved_transformed_cards = []
        self.unresolved_removed_relics = []
        self.triggered = False
    
    def is_unresolved(self):
        ret = (len(self.unresolved_removed_cards) + len(self.unresolved_upgraded_cards) + len(self.unresolved_transformed_cards) + len(self.unresolved_removed_relics)) > 0
        ret |= (UndeterminedCard.list_has_undetermined(self.cards_added + self.cards_removed + self.cards_upgraded + self.cards_transformed) != -1)
        return ret
    
    def _repr(self, char, items):
        if len(items):
            ret = char + ",".join([str(item) for item in items]) + " "
        else:
            ret = ""
        return ret

    def __repr__(self):
        ret = f"{self.floor}, {self.is_unresolved()}, "
        ret += self._repr("+", self.cards_added)
        ret += self._repr("-", self.cards_removed)
        ret += self._repr("^", self.cards_upgraded)
        ret += self._repr("~", self.cards_transformed)
        ret += self._repr("_", self.cards_transformed)
        ret += self._repr("~", self.cards_skipped)
        ret += self._repr("+", self.relics_added)
        ret += self._repr("-", self.relics_removed)
        ret += self._repr("-~", self.cards_removed_or_transformed)

        ret += self._repr("u-", self.unresolved_removed_cards)
        ret += self._repr("u^", self.unresolved_upgraded_cards)
        ret += self._repr("u~", self.unresolved_transformed_cards)
        ret += self._repr("u~", self.unresolved_removed_relics)

        return ret

    def __sub__(self, other : "FloorDelta"):
        ret = FloorDelta(
            floor = self.floor - other.floor,
            gold_delta = self.gold_delta - other.gold_delta,
            hp_delta = self.hp_delta - other.hp_delta,
        )

        ret.cards_added = list_difference(self.cards_added, other.cards_added)[0]
        ret.cards_removed = list_difference(self.cards_removed, other.cards_removed)[0]
        ret.cards_upgraded = list_difference(self.cards_upgraded, other.cards_upgraded)[0]
        ret.cards_transformed = list_difference(self.cards_transformed, other.cards_transformed)[0]
        ret.cards_transformed = list_difference(self.cards_transformed, other.cards_transformed)[0]
        ret.cards_skipped = list_difference(self.cards_skipped, other.cards_skipped)[0]
        ret.relics_added = list_difference(self.relics_added, other.relics_added)[0]
        ret.relics_removed = list_difference(self.relics_removed, other.relics_removed)[0]
        ret.cards_removed_or_transformed = list_difference(self.cards_removed_or_transformed, other.cards_removed_or_transformed)[0]
        ret.unresolved_removed_cards = list_difference(self.unresolved_removed_cards, other.unresolved_removed_cards)[0]
        ret.unresolved_upgraded_cards = list_difference(self.unresolved_upgraded_cards, other.unresolved_upgraded_cards)[0]
        ret.unresolved_transformed_cards = list_difference(self.unresolved_transformed_cards, other.unresolved_transformed_cards)[0]
        ret.unresolved_removed_relics = list_difference(self.unresolved_removed_relics, other.unresolved_removed_relics)[0]

        return ret

class FloorState:
    def __init__(
        self,
        floor : int,
        cards : List[str] = None,
        relics : List[str] = None,
        gold : int = 0,
        hp : int = 0,
        act: int = 0,
    ):
        self.floor = floor
        self.act = act
        self.cards = [format_string(_) for _ in cards] if cards is not None else []
        self.relics = [format_string(_) for _ in relics] if relics is not None else []
        self.gold = gold
        self.hp = hp
        self.modifiers = {}
    
    def __repr__(self) -> str:
        ret = str(self.floor)
        if len(self.relics):
            ret += ", " + ", ".join(self.relics)
        if len(self.cards):
            ret += ", " + ", ".join(self.cards)

        return ret
    
    def __sub__(self, other : "FloorState"):
        ret = FloorState(
            floor = self.floor - other.floor,
            cards = list_difference(self.cards, other.cards)[0],
            relics = list_difference(self.relics, other.relics)[0],
            gold = self.gold - other.gold,
            hp = self.hp - other.hp,
        )
        return ret
    
    def copy(self) -> "FloorState":
        ret = FloorState(
            floor = self.floor,
            cards = copy.deepcopy(self.cards),
            relics = copy.deepcopy(self.relics),
            gold = self.gold,
            hp = self.hp,
        )
        ret.modifiers = copy.deepcopy(self.modifiers)
        return ret
    
    def __eq__(self, other):
        ret = True
        ret &= self.cards == other.cards
        ret &= self.relics == other.relics
        ret &= self.gold == other.gold
        ret &= self.hp == other.hp
        return ret

def dispatch_found_cards(deck : list, cards_to_find : list, not_found_cards : list, do_allow_wrong_upgrade : bool = False):
    found_cards = []
    new_not_found_cards = []
    deck = copy.deepcopy(deck)
    for card in cards_to_find + not_found_cards:
        if (card in deck) and not isinstance(card, UndeterminedCard):
            found_cards.append(card)
            deck.remove(card)
        else:
            new_not_found_cards.append(card)
    
    if do_allow_wrong_upgrade:
        not_found_cards = new_not_found_cards
        new_not_found_cards = []
        deck_card_names = [card_to_name(card) for card in deck]
        for card in not_found_cards:
            if card_to_name(card) in deck_card_names:
                idx = deck_card_names.index(card_to_name(card))
                deck_card_names.pop(idx)
                card = deck.pop(idx)
                found_cards.append(card)
            else:
                new_not_found_cards.append(card)

    return found_cards, new_not_found_cards

def list_difference(list1 : list, list2 : list) -> Tuple[list, list]:
    exclusively_in_list1 = copy.deepcopy(list1)
    exclusively_in_list2 = copy.deepcopy(list2)

    idx = 0
    while idx < len(exclusively_in_list1):
        item = exclusively_in_list1[idx]
        if item in exclusively_in_list2:
            exclusively_in_list2.remove(item)
            exclusively_in_list1.pop(idx)
        else:
            idx += 1
    
    return exclusively_in_list1, exclusively_in_list2


class History:
    verbose = 0

    def __init__(self, initial_floor_state):
        self.last_resolved_floor_delta_idx = -1
        self.floor_deltas : List[FloorDelta] = []
        self.floor_states : List[FloorState] = [initial_floor_state] # states are previous to picking/skipping rewards at that floor
    
    def add(self, floor_delta : FloorDelta):
        floor_state = self.floor_states[-1].copy()
        self.floor_states.append(floor_state)
        self.floor_deltas.append(floor_delta)

        self.update_state_from_deltas(floor_state, floor_delta)
        
        # for perf, only do a backward-forward pass if we have work AND new clues
        if floor_delta.is_unresolved() and (self.last_resolved_floor_delta_idx < floor_delta.floor):
            self.bakward_forward()
    
    @staticmethod
    def try_to_resolve(resolved_cards, maybe_unresolved_cards, correcting_delta_cards, ignore_str=False):
        unresolved_idx = len(maybe_unresolved_cards) - 1
        while 0 <= unresolved_idx < len(maybe_unresolved_cards):
            unresolved_card = maybe_unresolved_cards[unresolved_idx]
            if isinstance(unresolved_card, UndeterminedCard):
                correcting_idx = unresolved_card.find_match(correcting_delta_cards, reversed_order=True)
                if correcting_idx != -1:
                    maybe_unresolved_cards.pop(unresolved_idx)
                    correcting_card = correcting_delta_cards.pop(correcting_idx)
                    resolved_cards.append(correcting_card)
            elif isinstance(unresolved_card, str):
                if (not ignore_str) and (unresolved_card in correcting_delta_cards):
                    maybe_unresolved_cards.pop(unresolved_idx)
                    correcting_delta_cards.remove(unresolved_card)
                    resolved_cards.append(unresolved_card)
            else:
                assert False
            unresolved_idx -= 1
    
    def update_state_from_deltas(self, floor_state : FloorState, floor_delta : FloorDelta, correcting_floor_delta : FloorDelta = None):
        """
        state and delta to compute the next state

        This is where we should be reading from state.modifiers (eg omamori) and trigger on-obtained relics (eg pandora's box)
        """
        floor_state.floor += 1
        floor_state.act += floor_delta.act_delta

        if correcting_floor_delta is not None:
            if "parasite" in correcting_floor_delta.cards_added:
                if floor_delta.maybe_got_parasite and "parasite" not in floor_delta.cards_added:
                    correcting_floor_delta.cards_added.remove("parasite")
                    floor_delta.cards_added.append("parasite")

            if floor_delta.is_unresolved():
                History.try_to_resolve(floor_delta.cards_added, floor_delta.cards_added, correcting_floor_delta.cards_added, ignore_str=True)
                History.try_to_resolve(floor_delta.cards_removed, floor_delta.unresolved_removed_cards, correcting_floor_delta.cards_removed)
                History.try_to_resolve(floor_delta.cards_transformed, floor_delta.unresolved_transformed_cards, correcting_floor_delta.cards_transformed)
                History.try_to_resolve(floor_delta.cards_removed, floor_delta.unresolved_removed_cards, correcting_floor_delta.cards_removed_or_transformed)
                History.try_to_resolve(floor_delta.cards_transformed, floor_delta.unresolved_transformed_cards, correcting_floor_delta.cards_removed_or_transformed)
                History.try_to_resolve(floor_delta.cards_upgraded, floor_delta.unresolved_upgraded_cards, correcting_floor_delta.cards_upgraded)

        # here we do some processing specific to relics

        # for some relics, we have to do the update only once, because we update possibly non-empty list attributes ; typically, we can be sure that boss relics 
        if not floor_delta.triggered:
            if "whetstone" in floor_delta.relics_added:
                floor_delta.cards_upgraded += [UndeterminedCard(card_type=CardType.ATTACK) for i in range(2)]

            if "warpaint" in floor_delta.relics_added:
                floor_delta.cards_upgraded += [UndeterminedCard(card_type=CardType.SKILL) for i in range(2)]
            
            if "pandora'sbox" in floor_delta.relics_added:
                transformed = [card for card in floor_state.cards if card in {'strike_r', 'defend_r'}]
                pandoras_n_transforms = len(transformed)
                floor_delta.cards_transformed = transformed
                floor_delta.cards_added = [UndeterminedCard() for i in range(pandoras_n_transforms)]

            if "astrolabe" in floor_delta.relics_added:
                n_transforms = 3
                floor_delta.cards_transformed = [UndeterminedCard() for i in range(n_transforms)]
                floor_delta.cards_added = [UndeterminedCard(upgraded=True) for i in range(n_transforms)]

            if "tinyhouse" in floor_delta.relics_added:
                floor_delta.cards_added += [UndeterminedCard()] # TODO: it's actually a skippable card reward
                floor_delta.cards_upgraded += [UndeterminedCard()]

            if "omamori" in floor_delta.relics_added:
                floor_state.modifiers["omamori_counter"] = 2

            if ("cursedkey" in floor_state.relics) and floor_delta.chest_opened:
                floor_delta.cards_added += [UndeterminedCard(color=Color.CURSE)]
            
            if ("callingbell" in floor_delta.relics_added):
                floor_delta.cards_added += ["curseofthebell"]
            
            if ("emptycage" in floor_delta.relics_added):
                floor_delta.cards_removed += [UndeterminedCard() for i in range(2)]
            
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
        
        # add relics
        floor_state.relics += floor_delta.relics_added

        # add cards
        for card in floor_delta.cards_added:
            card_is_a_curse = (card in CURSE_CARDS_FORMATTED) or (isinstance(card, UndeterminedCard) and (card.color == Color.CURSE))
            if (floor_state.modifiers.get("omamori_counter", 0) > 0) and card_is_a_curse:
                floor_state.modifiers["omamori_counter"] -= 1
            elif isinstance(card, UndeterminedCard):
                pass
            else:
                if card_to_n_upgrades(card) == 0:
                    card_name = card_to_name(card)
                    if ((card_name in ATTACK_CARDS_FORMATTED) and ("moltenegg2" in floor_state.relics)) or \
                    ((card_name in SKILL_CARDS_FORMATTED) and ("toxicegg2" in floor_state.relics)) or \
                    ((card_name in POWER_CARDS_FORMATTED) and ("frozenegg2" in floor_state.relics)):
                        card = card_to_upgrade(card)
                floor_state.cards.append(card)

        do_allow_wrong_upgrade = floor_delta.node == "?" # event_choices.*.cards_removed does not record the upgrade state of the removed cards

        # remove cards, retry to assign unresolved
        found_cards, new_not_found_cards = dispatch_found_cards(floor_state.cards, floor_delta.cards_removed, floor_delta.unresolved_removed_cards, do_allow_wrong_upgrade=do_allow_wrong_upgrade)
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
                if isinstance(card, UndeterminedCard):
                    continue
                upgraded_card = card_to_upgrade(card)
                if upgraded_card in correcting_floor_delta.cards_added:
                    correcting_floor_delta.cards_added.remove(upgraded_card)
                    correcting_floor_delta.cards_added.append(card)
                elif (card in correcting_floor_delta.cards_added) or isinstance(card, UndeterminedCard):
                    pass
                else:
                    # assert False
                    pass
        
        # transform cards, retry to assign unresolved
        found_cards, new_not_found_cards = dispatch_found_cards(floor_state.cards, floor_delta.cards_transformed, floor_delta.unresolved_transformed_cards)
        floor_delta.cards_transformed = found_cards
        floor_delta.unresolved_transformed_cards = new_not_found_cards
        for card in found_cards:
            floor_state.cards.remove(card)

        if (self.last_resolved_floor_delta_idx == floor_delta.floor - 1) and (not floor_delta.is_unresolved()):
            self.last_resolved_floor_delta_idx += 1
        floor_delta.triggered = True
    
    def log(self, *msg):
        if History.verbose:
            print(*msg)

    def bakward_forward(self, delta_to_master : FloorDelta = None):
        if delta_to_master is None:
            accumulated_delta = FloorDelta(floor=None)

            # go backward through floors that have unresolved cards in their deltas (eg should have removed a card but it was not in the deck), and collect them
            backward_delta_idx = len(self.floor_deltas)
            while backward_delta_idx > self.last_resolved_floor_delta_idx + 1:
                backward_delta_idx -= 1
                floor_delta = self.floor_deltas[backward_delta_idx]

                if not floor_delta.is_unresolved():
                    continue
                
                # append cards that were assumed to have been added
                accumulated_delta.cards_added += [card for card in floor_delta.unresolved_removed_cards if not isinstance(card, UndeterminedCard)]
                accumulated_delta.cards_added += [card for card in floor_delta.unresolved_upgraded_cards if not isinstance(card, UndeterminedCard)]
                accumulated_delta.cards_added += [card for card in floor_delta.unresolved_transformed_cards if not isinstance(card, UndeterminedCard)]
        else:
            accumulated_delta = delta_to_master
        
        forward_delta_idx = len(self.floor_deltas)

        # resolve issues at latest floors first (probably better because it's closer to the ground truth master_deck ? yes, say there are unknown upgrades (eg whetstone, war paint, tiny house) before and after unknown removes/transforms (eg pandora's box), if the earlier upgrades were resolved first, they would be used to upgrade the missing bash+ that is in the master deck, but actually maybe that was the later upgrade and the earlier upgrade was sunk into a card that was then removed/transformed)
        while forward_delta_idx > self.last_resolved_floor_delta_idx + 1:
            forward_delta_idx -= 1
            floor_delta = self.floor_deltas[forward_delta_idx]
            floor_state = self.floor_states[forward_delta_idx]
            next_floor_state = floor_state.copy()
            self.update_state_from_deltas(next_floor_state, floor_delta, accumulated_delta)
            if next_floor_state.cards != self.floor_states[forward_delta_idx + 1].cards: # noticed an update, let's propagate to later floors
                self.log("Update!")
                self.log("old", self.floor_states[forward_delta_idx + 1])
                self.log("new", next_floor_state)
                self.floor_states[forward_delta_idx + 1] = next_floor_state
                forward_pass_idx = forward_delta_idx + 1
                while forward_pass_idx < len(self.floor_states) - 1:
                    next_floor_state = self.floor_states[forward_pass_idx].copy()
                    floor_delta = self.floor_deltas[forward_pass_idx]
                    self.update_state_from_deltas(next_floor_state, floor_delta, accumulated_delta)
                    self.log("old", self.floor_states[forward_pass_idx + 1])
                    self.log("new", next_floor_state)
                    self.floor_states[forward_pass_idx + 1] = next_floor_state
                    forward_pass_idx += 1
    
    def wrap_up(self, master_deck : List[str]):
        self.log("wrap up")

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
            found = UndeterminedCard.find_matching_undetermined(infered_deck_copy, card)
            if found != -1:
                infered_deck_copy.pop(found)
        while len(infered_deck_copy):
            card = infered_deck_copy.pop(0)
            if not isinstance(card, UndeterminedCard):
                delta_to_master.cards_removed_or_transformed.append(card)

        delta_to_master_copy = copy.deepcopy(delta_to_master)
        self.bakward_forward(delta_to_master)

        if History.verbose >= 1:
            print("Infered deltas")
            for delta in self.floor_deltas:
                print(delta)

            print("Infered states")
            for state in self.floor_states:
                print(state.floor, sorted(state.cards))

        return delta_to_master

def filter_run(data : dict):
    if not data["is_ascension_mode"]: return False
    if data["character_chosen"] != "IRONCLAD": return False
    if data["ascension_level"] < 10: return False
    # if not data["victory"]: return False

    if is_bad_data(data): return False

    return True

def rebuild_deck_from_vanilla_run(data : dict):
    if data["character_chosen"] == "IRONCLAD":
        initial_deck = ["Defend_R"]*4 + ["Strike_R"]*5 + ["Bash", "AscendersBane"]
        class_color = Color.RED
    else:
        raise NotImplementedError(data["character_chosen"])
    initial_floor_state = FloorState(
        floor = 0,
        cards = initial_deck,
    )

    history = History(initial_floor_state)

    floor_samples = []
    floor = -1
    act = 1
    entered_secret_portal = False
    path_per_floor = ["NEOW"] + copy.deepcopy(data["path_per_floor"])
    for node in path_per_floor:
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
            found_combat = False
            damage_taken_data = {}
            for damage_taken_data in data["damage_taken"]:
                if damage_taken_data["floor"] == floor:
                    found_combat = True
                    break

            if node == "B":
                if act <= 2:
                    if (len(data["boss_relics"]) >= act) and ("picked" in data["boss_relics"][act-1]):
                        floor_delta_dict["relics_added"] = [data["boss_relics"][act-1]["picked"]]

            elif node == "M":
                if (act == 3) and found_combat and (damage_taken_data["enemies"] == "Writhing Mass"):
                    floor_delta_dict["maybe_got_parasite"] = True
        
        elif node == "null":
            if floor + 1 != len(path_per_floor):
                floor_delta_dict["act_delta"] = 1
                act += 1
                assert act <= 4

        elif node == "NEOW":
            neow_bonus = data.get("neow_bonus", "")
            if neow_bonus == "":
                pass
            elif neow_bonus == "BOSS_RELIC":
                floor_delta_dict["relics_added"] = [data["relics"][0]]
            elif neow_bonus == "ONE_RANDOM_RARE_CARD":
                floor_delta_dict["cards_added"] = [UndeterminedCard(color=class_color)]
            elif neow_bonus == "REMOVE_CARD":
                floor_delta_dict["cards_removed"] = [UndeterminedCard(color=class_color)]
            elif neow_bonus == "REMOVE_TWO":
                floor_delta_dict["cards_removed"] = [UndeterminedCard(color=class_color) for i in range(2)]
            elif neow_bonus == "TRANSFORM_CARD":
                floor_delta_dict["cards_added"] = [UndeterminedCard(color=class_color)]
                floor_delta_dict["cards_transformed"] = [UndeterminedCard(color=class_color)]
            elif neow_bonus == "TRANSFORM_TWO_CARDS":
                floor_delta_dict["cards_added"] = [UndeterminedCard(color=class_color) for i in range(2)]
                floor_delta_dict["cards_transformed"] = [UndeterminedCard(color=class_color) for i in range(2)]
            elif neow_bonus == "RANDOM_COMMON_RELIC":
                floor_delta_dict["relics_added"] = [data["relics"][1]]
            elif neow_bonus == "ONE_RARE_RELIC":
                floor_delta_dict["relics_added"] = [data["relics"][1]]
            elif neow_bonus == "RANDOM_COLORLESS":
                floor_delta_dict["cards_added"] = [UndeterminedCard(color=Color.COLORLESS)]
            elif neow_bonus == "RANDOM_COLORLESS_2":
                floor_delta_dict["cards_added"] = [UndeterminedCard(color=Color.COLORLESS)]
            elif neow_bonus == "THREE_ENEMY_KILL":
                pass
            elif neow_bonus == "HUNDRED_GOLD":
                pass
            elif neow_bonus == "TWO_FIFTY_GOLD":
                pass
            elif neow_bonus == "THREE_CARDS":
                floor_delta_dict["cards_added"] = [UndeterminedCard(color=class_color)]
            elif neow_bonus == "THREE_RARE_CARDS":
                floor_delta_dict["cards_added"] = [UndeterminedCard(color=class_color, rarity=Rarity.RARE)]
            elif neow_bonus == "TWENTY_PERCENT_HP_BONUS":
                pass
            elif neow_bonus == "TEN_PERCENT_HP_BONUS":
                pass
            elif neow_bonus == "THREE_SMALL_POTIONS":
                pass
            elif neow_bonus == "UPGRADE_CARD":
                floor_delta_dict["cards_upgraded"] = [UndeterminedCard(color=class_color)]
            else:
                assert False, data["neow_bonus"]
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
                    if "relics_obtained" in event_choice:
                        for card in event_choice["relics_obtained"]:
                            floor_delta_dict["relics_added"] = floor_delta_dict.get("relics_added", []) + [card]
                    floor_delta_dict["event_name"] = event_choice["event_name"]
                    if (event_choice["event_name"] == "SecretPortal") and (event_choice["player_choice"] == "Took Portal"):
                        entered_secret_portal = True
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
    act_reached = act

    for floor_state, floor_delta in zip(history.floor_states, history.floor_deltas):
        if len(floor_delta.cards_skipped):
            floor_sample = {
                "relics": [relic for relic in floor_state.relics if (relic != DEFINITELY_SOMETHING) and (relic in ALL_RELICS_FORMATTED)],
                "deck": [card for card in floor_state.cards if not isinstance(card, UndeterminedCard)],
                "cards_picked": [card for card in floor_delta.cards_added if not isinstance(card, UndeterminedCard)],
                "cards_skipped": [card for card in floor_delta.cards_skipped if not isinstance(card, UndeterminedCard)],
                "victory": data["victory"],
                "floor": floor_state.floor,
                "act": floor_state.act,
                "floor_reached": data["floor_reached"],
                "act_reached": act_reached,
            }
            floor_samples.append(floor_sample)

    error = (len(delta_to_master.cards_added) > 0) or (len(delta_to_master.cards_removed_or_transformed) > 0) or (len(delta_to_master.cards_upgraded) > 0)
    warning = history.last_resolved_floor_delta_idx < len(data["path_per_floor"])
    return floor_samples, not error, not warning, delta_to_master

def test_reconstruct():
    relics_obtained = [
        {
            "key": "Whetstone", # upgrade 2 'Strike'
            "floor": 1,
        },
        {
            "key": "Pandora's Box", # get 9 'Sword Boomerangs'
            "floor": 2,
        },
        {
            "key": "Whetstone", # upgrade 'Bash' and 'Sword Boomerangs'
            "floor": 3,
        },
        {
            "key": "Omamori",
            "floor": 4,
        },
        {
            "key": "Cursed Key",
            "floor": 5,
        },
        {
            "key": "Calling Bell", # get a curse from 'Cursed Key' but negated by 'Omamori', get a curse from 'Callingbell' but negated by 'Omamori'
            "floor": 6,
        },
        {
            "key": "Empty Cage", # get 'Pain' from cursed key, remove 2 'Sword Boomerangs'
            "floor": 7,
        },
    ]

    master_decks = [
        ["Strike_r"] * 5 + ["Defend_r"] * 4 + ["Bash", "AscendersBane"],
        ["Strike_r"] * 5 + ["Defend_r"] * 4 + ["Bash", "AscendersBane"], # neow, nothing happened
        ["Strike_r"] * 3 + ["Strike_r+1"] * 2 + ["Defend_r"] * 4 + ["Bash", "AscendersBane"],
        ["Sword Boomerang"] * 9 + ["Bash", "AscendersBane"],
        ["Sword Boomerang"] * 8 + ["Sword Boomerang+1", "Bash+1", "AscendersBane"],
        ["Sword Boomerang"] * 8 + ["Sword Boomerang+1", "Bash+1", "AscendersBane"],
        ["Sword Boomerang"] * 8 + ["Sword Boomerang+1", "Bash+1", "AscendersBane"],
        ["Sword Boomerang"] * 8 + ["Sword Boomerang+1", "Bash+1", "AscendersBane"],
        ["Sword Boomerang"] * 6 + ["Sword Boomerang+1", "Bash+1", "AscendersBane", "Pain"],
    ]

    # max_floor = len(master_decks) - 1
    max_floor = 4

    data = {
        "is_ascension_mode": True,
        "character_chosen": "IRONCLAD",
        "ascension_level": 20,
        "victory": True,

        "master_deck": master_decks[max_floor],
        "relics": [],
        "path_per_floor": ["T"]*(max_floor-1), # -1 because path_per_floor does not count neow
        "relics_obtained": relics_obtained,
        "card_choices": [],
        "neow_bonus": "",
        "event_choices": [],
        "campfire_choices": [],
        "damage_taken": [],
        "items_purchased": [],
        "item_purchase_floors": [],
        "items_purged": [],
        "items_purged_floors": [],
        "boss_relics": [],
        "floor_reached": 51,

        "is_trial": False,
        "is_daily": False,
        "chose_seed": False,
        "is_endless": False,
        "circlet_count": 0,
        "score": 1000,
        "player_experience": 100,
    }

    History.verbose = 1
    floor_samples, success, no_warning, delta_to_master = rebuild_deck_from_vanilla_run(data)
    print(success, no_warning, delta_to_master)

    print("Oracle states")
    for floor, master_deck in enumerate(master_decks[:max_floor+1]):
        print(floor, sorted(master_deck))


def test_reconstruct2():
    relics_obtained = [
        {
            "key": "Omamori",
            "floor": 1,
        },
        {
            "key": "Cursed Key",
            "floor": 2,
        },
        {
            "key": "Calling Bell", # get a curse from 'Cursed Key' but negated by 'Omamori', get a curse from 'Callingbell' but negated by 'Omamori'
            "floor": 3,
        },
        {
            "key": "Empty Cage", # get 'Pain' from cursed key, remove 2 'Strike'
            "floor": 4,
        },
    ]

    master_decks = [
        ["Strike_r"] * 5 + ["Defend_r"] * 4 + ["Bash", "AscendersBane"],
        ["Strike_r"] * 5 + ["Defend_r"] * 4 + ["Bash", "AscendersBane"], # neow, nothing happened
        ["Strike_r"] * 5 + ["Defend_r"] * 4 + ["Bash", "AscendersBane"], # omamori
        ["Strike_r"] * 5 + ["Defend_r"] * 4 + ["Bash", "AscendersBane"], # cursed key
        ["Strike_r"] * 5 + ["Defend_r"] * 4 + ["Bash", "AscendersBane"], # calling bell
        ["Strike_r"] * 3 + ["Defend_r"] * 4 + ["Bash", "AscendersBane", "Pain"], # empty cage: remove 2 strikes, get pain from cursed key
    ]

    max_floor = len(master_decks) - 1
    # max_floor = 4

    data = {
        "is_ascension_mode": True,
        "character_chosen": "IRONCLAD",
        "ascension_level": 20,
        "victory": True,

        "master_deck": master_decks[max_floor],
        "relics": [],
        "path_per_floor": ["T"]*(max_floor-1), # -1 because path_per_floor does not count neow
        "relics_obtained": relics_obtained,
        "card_choices": [],
        "neow_bonus": "",
        "event_choices": [],
        "campfire_choices": [],
        "damage_taken": [],
        "items_purchased": [],
        "item_purchase_floors": [],
        "items_purged": [],
        "items_purged_floors": [],
        "boss_relics": [],
        "floor_reached": 51,

        "is_trial": False,
        "is_daily": False,
        "chose_seed": False,
        "is_endless": False,
        "circlet_count": 0,
        "score": 1000,
        "player_experience": 100,
    }

    History.verbose = 1
    floor_samples, success, no_warning, delta_to_master = rebuild_deck_from_vanilla_run(data)
    print(success, no_warning, delta_to_master)

    print("Oracle states")
    for floor, master_deck in enumerate(master_decks[:max_floor+1]):
        print(floor, sorted(master_deck))

def test_reconstruct_easy():
    relics_obtained = [
        {
            "key": "Whetstone", # upgrade 2 'Strike'
            "floor": 1,
        },
    ]

    master_decks = [
        ["Strike_r"] * 5 + ["Defend_r"] * 4 + ["Bash", "AscendersBane"],
        ["Strike_r"] * 3 + ["Strike_r+1"] * 2 + ["Defend_r"] * 4 + ["Bash", "AscendersBane"],
    ]

    max_floor = 1
    data = {
        "is_ascension_mode": True,
        "character_chosen": "IRONCLAD",
        "ascension_level": 20,
        "victory": True,

        "master_deck": master_decks[max_floor],
        "relics": [],
        "path_per_floor": ["T"]*max_floor,
        "relics_obtained": relics_obtained[:max_floor],
        "card_choices": [],
        "neow_bonus": "",
        "event_choices": [],
        "campfire_choices": [],
        "damage_taken": [],
        "items_purchased": [],
        "item_purchase_floors": [],
        "items_purged": [],
        "items_purged_floors": [],
        "boss_relics": [],
        "floor_reached": 51,

        "is_trial": False,
        "is_daily": False,
        "chose_seed": False,
        "is_endless": False,
        "circlet_count": 0,
        "score": 1000,
        "player_experience": 100,
    }

    History.verbose = 1
    floor_samples, success, no_warning, delta_to_master = rebuild_deck_from_vanilla_run(data)
    print(success, no_warning, delta_to_master)

def tokenize_dataset(databatch: list, tokens: list):
    for sample in databatch:
        processed_keys = []
        for key in sample:
            contains_list_of_items = isinstance(sample[key], list) and len(sample[key]) and isinstance(sample[key][0], str)
            if contains_list_of_items:
                old_list = sample[key]
                new_list = []
                for item in old_list:
                    splits = item.split("+")
                    name = splits[0]
                    token_idx = tokens.index(name)
                    new_name = str(token_idx)
                    if len(splits) > 1:
                        n_upgrades = splits[1]
                        new_name += f"+{n_upgrades}"
                    new_list.append(new_name)
                sample[key] = new_list
                processed_keys.append(key)
    return databatch

def detokenize(databatch: list, tokens: list):
    to_process_keys = ['relics', 'deck', 'cards_picked', 'cards_skipped']
    for sample in databatch:
        processed_keys = []
        for key in sample:
            if key in to_process_keys:
                old_list = sample[key]
                new_list = []
                for item in old_list:
                    splits = item.split("+")
                    token_idx = int(splits[0])
                    name = tokens[token_idx]
                    new_name = name
                    if len(splits) > 1:
                        n_upgrades = splits[1]
                        new_name += f"+{n_upgrades}"
                    new_list.append(new_name)
                sample[key] = new_list
                processed_keys.append(key)
    return databatch

def create_dataset(
    source_json_path,
    debug_start_idx = None,
    debug_end_idx = None,
):
    if isinstance(source_json_path, str):
        source_json_paths = [source_json_path]
    source_json_paths = source_json_path

    draft_dataset = []
    total_diff_l0 = 0
    total_diff_l1 = 0
    computed_run = 0
    is_debugging = (debug_start_idx is not None) or (debug_end_idx is not None)
    tokens = [PAD_TOKEN] + ALL_CARDS_FORMATTED + ALL_RELICS_FORMATTED

    for source_json_path in source_json_paths:
        datas = json.load(open(source_json_path, "r"))
        datas = [data for data in datas if filter_run(data["event"])]
        for run_idx, data in enumerate(datas):
            if debug_start_idx is not None and run_idx < debug_start_idx:
                continue
            if debug_end_idx is not None and run_idx > debug_end_idx:
                break
            assert set(data.keys()) == {'event'}
            data = data["event"]
            if is_debugging:
                json.dump(data, open("./example_vanilla.run", "w"), indent=4)
            try:
                floor_samples, success, no_warning, delta_to_master = rebuild_deck_from_vanilla_run(data)
            except UnknownCard as e:
                print(f"{run_idx}: Unknown card '{e.card}'")
                if e.card.lower().startswith("sk?"): # one or several runs in the dataset have this issue, maybe a localization issue from STS?
                    continue
                raise e
            except Exception as e:
                print(run_idx, ": Got exception:", repr(e))
                continue
            diff = len(delta_to_master.cards_added) + len(delta_to_master.cards_removed_or_transformed) + len(delta_to_master.cards_upgraded)
            total_diff_l1 += diff
            if diff < 3:
                floor_samples = tokenize_dataset(floor_samples, tokens)
                draft_dataset += floor_samples
            if diff or success:
                if diff >= 1:
                    print(f"{run_idx}: diff = {diff} ; to add = {delta_to_master.cards_added} ; to remove = {delta_to_master.cards_removed_or_transformed} ; to upgrade = {delta_to_master.cards_upgraded}")
                computed_run += 1
                total_diff_l0 += int(diff > 0)
    print(f"Diff score over {computed_run} runs = {total_diff_l1}, {total_diff_l0} could not be reconstructed.")

    git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    json_basename = ".".join(os.path.basename(source_json_path).split(".")[:-1])
    filepath = f"./{json_basename}_{git_hash}_{len(draft_dataset)}.data"
    data = {
        "dataset": draft_dataset,
        "items": tokens,
    }
    json.dump(data, open(filepath, "w"))
    print(f"Dumped dataset of {len(draft_dataset)} samples and {len(tokens)} tokens into {filepath}")

def compile_datas():
    """
    From .json batches of runs in ./SlayTheData, filter out by character class / wins / ascension level...
    """
    batch_idx = 0
    def dump_batch(compiled_datas):
        write_path = f"SlayTheData_a10+_ic_{batch_idx}_{len(compiled_datas)}.json"
        json.dump(compiled_datas, open(write_path, "w"))
        print(f"Dumped {write_path}")
        compiled_datas.clear()

    glob_expr = "./SlayTheData/*"
    compiled_datas = []
    next_print_log_10 = 0
    total_runs = 0
    for filepath in glob(glob_expr):
        try:
            datas = json.load(open(filepath, "r"))
        except Exception as e:
            print(repr(e))
            print(f"Invalid file {filepath}")
            continue
        compiled_data_sub = []
        for data in datas:
            try:
                if filter_run(data["event"]):
                    compiled_data_sub.append(data)
            except Exception as e:
                path = "./invalid.run"
                print(repr(e))
                print(f"Invalid run, dumped into {path}")
                json.dump(data, open(path, "w"), indent=4)
        compiled_datas += compiled_data_sub
        total_runs += len(compiled_data_sub)
        if total_runs > 10**next_print_log_10:
            print(f"Compiled {total_runs} runs")
            next_print_log_10 = int(np.log10(total_runs)) + 1
        
        if len(compiled_datas) > 100000:
            dump_batch(compiled_datas)
            batch_idx += 1

    if len(compiled_datas) > 0:
        dump_batch(compiled_datas)

    print(f"Done. Compiled {total_runs} runs")

if __name__ == "__main__":
    create_dataset(
        source_json_path = [
            "./SlayTheData_a10+_ic_0_100008.json",
            "./SlayTheData_a10+_ic_1_100131.json",
            "./SlayTheData_a10+_ic_2_128944.json",
        ],
        debug_start_idx = None,
        debug_end_idx = None,
    )
    # compile_datas()
    # test_reconstruct()
    # test_reconstruct2()
    # test_reconstruct_easy()

