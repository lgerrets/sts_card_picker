# SlayTheSpire Card Picker

## Features

- Process vanilla .run files to try to determine what was the state of the deck and relics at each card rewards and shop decisions
- Train and use deep AIs from human data
  - Uses 'multi head attention' to best deal with the varying size of the inputs (eg deck size and relics)
- GUI assistant that hooks into STS to make predictions at card rewards and shops, using CommunicationMod

## What?

The goal is to teach an AI to pick cards at card rewards after combats, or at shops, or generally at any point where a 'draft' type of decision must be made. We aim to make it learn based on data of (hopefully expert) human runs.

First, one known challenge is that .run files that STS generate do not log everything that happens in a run. For instance, just by looking at the .run, we may have no way to find out what curse was obtained at a chest if the player had a Cursed Key and later removed it with Empty Cage. Still, we need to reconstruct what choices the player had to make at each floor to build our training dataset.

Then, we train a model (using one particular type of deep neural network that handles well input data of **varying** size) to mimic the choices that players made, conditionned by their deck of cards or relics.

My bot achieves ~55% accuracy on the left-out test dataset. As a comparison, note that a bot that just skips all cards has 32% accuracy on my data, and a bot that picks one randomly has 25% accuracy (indeed most of the time picks are 1 out of 4 options).

## Minimal requirements

Windows only, in a powershell:

```
Install-Module -Name Set-PsEnv
Set-Alias -Name dotenv -Value Set-PsEnv
```

All platforms:

```
dotenv
python -m venv env
./env/scripts/activate
pip install -r requirements.txt
```

As a dev environment, I recommend VSCode which has great python and jupyter extensions.

## Human assistant GUI

This part was made possible by the great works from ForgottenArbiter at https://github.com/ForgottenArbiter/CommunicationMod and https://github.com/ForgottenArbiter/spirecomm

### Install

Additional setup:

```
cd external
git clone https://github.com/ForgottenArbiter/spirecomm
```

Also get the mod CommunicationMod eg through Steam. Then edit your game's `...\ModTheSpire\CommunicationMod\config.properties`

```
command=...\\sts_card_picker\\env\\Scripts\\python.exe ...\\sts_card_picker\\gui_card_assistant.py
```

### Run

Run STS. Click Mods > CommunicationMod > Start process.

A window should pop up. If not, error logs are dumped into `...\\sts_card_picker\\gui.log`

Play the game normally. At card rewards and shops, the GUI should automatically update and show you scores for each card.

Example: [img](doc/gui_assistant.png)

## Train a model

```
dotenv
./env/scripts/activate
```

First look at [main.ipynb](main.ipynb) which has more detailed documentation, and optionnally run it.

Then, the files to look at in the order of the workflow are:

```
python sts_ml/deck_history.py
python sts_ml/train.py
python sts_ml/infer.py
```

