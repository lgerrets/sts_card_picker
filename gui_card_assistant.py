"""
This script is intended to be called by CommunicationMod. It serves as an assistant to humans. The model will automatically run inferences in certain situations to give some advice:
- at card rewards or shops, inputs to the model are the current [deck, relics, set of offered cards]
"""

import hashlib
import os
import collections
import json
import sys
import logging

CUR_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CUR_DIR, "external", "spirecomm"))

logging.basicConfig(filename=os.path.join(CUR_DIR, "gui.log"), filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__file__)

import spirecomm.communication.coordinator as coord

from sts_ml.deck_history import format_string, card_to_upgrade
from sts_ml.train import TRAINING_DIR
from sts_ml.registry import instanciate_model

os.environ["KIVY_NO_CONSOLELOG"] = "1"

from kivy.app import App
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.core.window import Window

def pretty_format(data):
    return json.dumps(data, indent=4)

class Base(BoxLayout):

    def __init__(self, coordinator, model):
        super().__init__(orientation='vertical')
        self.coordinator = coordinator
        self.model = model

        self.game_output = TextInput(size_hint=(1, 10))
        self.game_output.text = "blank"
        self.game_output.readonly = True
        self.add_widget(self.game_output)

        self.agent_output = TextInput(size_hint=(1, 2))
        self.agent_output.text = "blank"
        self.agent_output.readonly = True
        self.add_widget(self.agent_output)

        self.history_text = TextInput(size_hint=(1, 2))
        self.add_widget(self.history_text)

        self.output_text = TextInput(size_hint=(1, 1))
        self.add_widget(self.output_text)

        self.button = Button(text='Send', size_hint=(1, 1))
        self.button.bind(on_press=self.send_output)
        self.add_widget(self.button)

        self.max_history_lines = 5
        self.history_lines = collections.deque(maxlen=self.max_history_lines)

        Window.bind(on_key_up=self.key_callback)

        self.last_infer_hash = ""

    def do_communication(self, dt):
        message = self.coordinator.get_next_raw_message()
        if message is not None:
            data = json.loads(message)
            self.game_output.text = pretty_format(data)
            if data.get("game_state", {}).get("screen_type", "") == "CARD_REWARD":
                reward_card_data = data["game_state"]["screen_state"]["cards"]
                reward_card_names = [format_string(dat["id"]) for dat in reward_card_data]
                reward_card_n_upgrades = [dat["upgrades"] for dat in reward_card_data]
                m = hashlib.sha256()
                m.update("".join(reward_card_names).encode())
                new_hash = m.digest()

                if new_hash != self.last_infer_hash:
                    self.last_infer_hash = new_hash

                    deck_card_data = data["game_state"]["deck"]
                    deck_card_names = [format_string(dat["id"]) for dat in deck_card_data]
                    deck_card_n_upgrades = [dat["upgrades"] for dat in deck_card_data]

                    relic_data = data["game_state"]["relics"]
                    relic_names = [format_string(dat["id"]) for dat in relic_data]
                    
                    sample = {
                        "deck": [card_to_upgrade(name, n_upgrades) for name, n_upgrades in zip(deck_card_names, deck_card_n_upgrades)],
                        "cards_skipped": [card_to_upgrade(name, n_upgrades) for name, n_upgrades in zip(reward_card_names, reward_card_n_upgrades)],
                        "cards_picked": [],
                        "relics": relic_names,
                    }
                    names, scores = self.model.predict_one(sample)
                    self.agent_output.text = pretty_format(dict(zip(names, scores.tolist())))
            elif data.get("game_state", {}).get("screen_type", "") == "SHOP_SCREEN":
                reward_card_data = data["game_state"]["screen_state"]["cards"]
                reward_card_names = [format_string(dat["id"]) for dat in reward_card_data]
                reward_card_n_upgrades = [dat["upgrades"] for dat in reward_card_data]
                m = hashlib.sha256()
                m.update("".join(reward_card_names).encode())
                new_hash = m.digest()

                if new_hash != self.last_infer_hash:
                    self.last_infer_hash = new_hash

                    deck_card_data = data["game_state"]["deck"]
                    deck_card_names = [format_string(dat["id"]) for dat in deck_card_data]
                    deck_card_n_upgrades = [dat["upgrades"] for dat in deck_card_data]

                    relic_data = data["game_state"]["relics"]
                    relic_names = [format_string(dat["id"]) for dat in relic_data]
                    
                    sample = {
                        "deck": [card_to_upgrade(name, n_upgrades) for name, n_upgrades in zip(deck_card_names, deck_card_n_upgrades)],
                        "cards_skipped": [card_to_upgrade(name, n_upgrades) for name, n_upgrades in zip(reward_card_names, reward_card_n_upgrades)],
                        "cards_picked": [],
                        "relics": relic_names,
                    }
                    names, scores = self.model.predict_one(sample)
                    self.agent_output.text = pretty_format(dict(zip(names, scores.tolist())))
        self.coordinator.execute_next_action_if_ready()

    def send_output(self, instance=None, text=None):
        if text is None:
            text = self.output_text.text
        text = text.strip()
        print(text, end='\n', flush=True)
        self.history_lines.append(text)
        self.history_text.text = "\n".join(self.history_lines)
        self.output_text.text = ""

    def key_callback(self, window, keycode, *args):
        if keycode == 13:
            self.send_output()


class CommunicationApp(App):

    def __init__(self, coordinator, model):
        super().__init__()
        self.coordinator = coordinator
        self.model = model

    def build(self):
        base = Base(self.coordinator, self.model)
        Clock.schedule_interval(base.do_communication, 1.0 / 60.0)
        return base


def launch_gui():
    try:
        communication_coordinator = coord.Coordinator()
        communication_coordinator.signal_ready()

        model_name = "2023-03-11-20-30-37_CardModel_blocks4x256_split0.8_relics"
        training_dir = os.path.join(CUR_DIR, TRAINING_DIR, model_name)
        model = instanciate_model(training_dir=training_dir, ckpt=2355)

        CommunicationApp(communication_coordinator, model).run()
    except Exception as e:
        logger.exception(str(repr(e)) + "\n")

if __name__ == "__main__":
    launch_gui()
