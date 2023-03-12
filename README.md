# SlayTheSpire Card Picker

## What?

The goal is to teach an AI to pick cards at card rewards after combats, or at shops, or generally at any point where a 'draft' type of decision must be made. We aim to make it learn based on data of (hopefully expert) human runs.

First, one known challenge is that .run files that STS generate do not log everything that happens in a run. For instance, just by looking at the .run, we may have no way to find out what curse was obtained at a chest if the player had a Cursed Key and later removed it with Empty Cage. Still, we need to reconstruct what choices the player had to make at each floor to build our training dataset.

Then, we train a model (using one particular type of deep neural network that handles well input data of **varying** size) to mimic the choices that players made, conditionned by their deck of cards or relics.

My bot achieves ~55% accuracy on the left-out test dataset. As a comparison, note that a bot that just skips all cards has 32% accuracy on my data, and a bot that picks one randomly has 25% accuracy (indeed most of the time picks are 1 out of 4 options).

## For devs

### Install

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

cd external
git clone https://github.com/ForgottenArbiter/spirecomm
```

As a dev environment, I use VSCode which has great python and jupyter extensions.

### Run

```
dotenv
./env/scripts/activate
```

It's recommended to first look at [main.ipynb](main.ipynb) and optionnally run it.

Then, the files to look at in the order of the workflow are:

```
python sts_ml/deck_history.py
python sts_ml/train.py
python sts_ml/infer.py
```

