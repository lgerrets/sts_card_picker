## .run processing

- Be careful of modded .run
- For now we drop out samples from the dataset that contain unknown cards
- Finetuning the processing of vanilla .run
- Have richer information on undetermined card changes
- Add relics and some more state to the input of the model
- In second part of bakward_forward, resolving later floors first does not make sense because the accumulated delta is the total delta!!!
- Deal with Cursed key?
- Deal with parasite from wriggling horror?
- Tinyhouse: skipped cards are not logged, the added card is put together with the rare pick

## Model

- I'm padding the entire dataset once to minimize out of distribution but this may be bad for perf and generalize poorly?
- Model could predict richer labels, eg the damage taken as SlayTheSpireFightPredictor

## References

- https://github.com/alexdriedger/SlayTheSpireFightPredictor
  - nice and clean code
  - seems to well manage certain constraints for the backward pass (eg the upgraded card was a skill, so look for an upgraded skill in the master deck)
  - their model does an average of the embeddings, but doing attention as I'm doing should be a great low hanging fruit improvement?
  - I would be almost tempted to fork their repo given how clean and polished it is, but I'm happy with my backward/forward phylosophy for processing .run + pytorch over tf and sklearn

### Datas

Listing potentially interesting data ressources

- official 77 millions .run dataset 
- official 'november' dataset (part of the 77M dataset)
- Baalor 400 wins
- Jorbs
- https://spirelogs.com
  - https://dev.spirelogs.com/archive is down ? :(

## Examples

              cards expert_scores predicted_scores
0          defend_r          deck             deck
1          defend_r          deck             deck
2          defend_r          deck             deck
3          defend_r          deck             deck
4          strike_r          deck             deck
5          strike_r          deck             deck
6              bash          deck             deck
7     ascendersbane          deck             deck
8   perfectedstrike          deck             deck
9          truegrit          deck             deck
10     battletrance          deck             deck
11     powerthrough          deck             deck
12           evolve          deck             deck
13        armaments          deck             deck
14     flamebarrier          deck             deck
15        fiendfire          deck             deck
16        armaments          deck             deck
17        armaments          deck             deck
18     bloodletting          deck             deck
19        shockwave          deck             deck
20     feelnopain+1          deck             deck
21         offering          deck             deck
22       entrench+1          deck             deck
23     feelnopain+1          deck             deck
24       shrugitoff          deck             deck
25     shrugitoff+1          deck             deck
26           disarm          deck             deck
27     powerthrough          deck             deck
28         disarm+1          deck             deck
29          blind+1          deck             deck
30   searingblow+10          deck             deck
31         truegrit             0         0.235461
32         headbutt             1         0.141313
33      clothesline             0         0.023279

            cards expert_scores predicted_scores
0        defend_r          deck             deck
1        defend_r          deck             deck
2        defend_r          deck             deck
3        defend_r          deck             deck
4        strike_r          deck             deck
5        strike_r          deck             deck
6        strike_r          deck             deck
7        strike_r          deck             deck
8        strike_r          deck             deck
9            bash          deck             deck
10  ascendersbane          deck             deck
11   powerthrough          deck             deck
12   battletrance          deck             deck
13      armaments             0          0.36048
14       truegrit             1         0.338117
15       ironwave             0         0.039726

            cards expert_scores predicted_scores
0        defend_r          deck             deck
1        defend_r          deck             deck
2        defend_r          deck             deck
3        defend_r          deck             deck
4        strike_r          deck             deck
5        strike_r          deck             deck
6        strike_r          deck             deck
7        strike_r          deck             deck
8        strike_r          deck             deck
9            bash          deck             deck
10  ascendersbane          deck             deck
11     feelnopain          deck             deck
12   pommelstrike          deck             deck
13     secondwind          deck             deck
14         reaper          deck             deck
15        carnage          deck             deck
16     truegrit+1             1         0.327993
17    clothesline             0         0.097137
18         flex+1             0         0.062112

### Better model

             cards expert_scores predicted_scores
0         defend_r          deck             deck
1         defend_r          deck             deck
2         defend_r          deck             deck
3         defend_r          deck             deck
4         strike_r          deck             deck
5         strike_r          deck             deck
6         strike_r          deck             deck
7    ascendersbane          deck             deck
8          carnage          deck             deck
9       twinstrike          deck             deck
10     fiendfire+1          deck             deck
11    pommelstrike          deck             deck
12      shrugitoff          deck             deck
13     whirlwind+1          deck             deck
14  pommelstrike+1          deck             deck
15       inflame+1          deck             deck
16       doubletap          deck             deck
17      uppercut+1          deck             deck
18       shockwave          deck             deck
19      headbutt+1          deck             deck
20         havoc+1          deck             deck
21         havoc+1          deck             deck
22        exhume+1          deck             deck
23          bash+1          deck             deck
24      strike_r+1          deck             deck
25   darkembrace+1          deck             deck
26  bloodletting+1          deck             deck
27    bloodletting          deck             deck
28  battletrance+1          deck             deck
29        exhume+1          deck             deck
30         havoc+1             1          0.99995
31    shrugitoff+1             1         0.998808
32    wildstrike+1             0         0.000096
33     armaments+1             0         0.000022
34     seversoul+1             0         0.000005
35  powerthrough+1             0         0.000004

                cards expert_scores predicted_scores
0            defend_r          deck             deck
1            defend_r          deck             deck
2            defend_r          deck             deck
3            defend_r          deck             deck
4            strike_r          deck             deck
5            strike_r          deck             deck
6            strike_r          deck             deck
7                bash          deck             deck
8       ascendersbane          deck             deck
9            truegrit          deck             deck
10       pommelstrike          deck             deck
11  perfectedstrike+1          deck             deck
12  perfectedstrike+1          deck             deck
13    perfectedstrike          deck             deck
14        thunderclap          deck             deck
15    perfectedstrike          deck             deck
16        demonform+1          deck             deck
17               feed          deck             deck
18             disarm          deck             deck
19             disarm          deck             deck
20        darkembrace          deck             deck
21            havoc+1             1         0.999854
22       bloodletting             0         0.000366
23     swordboomerang             0         0.000003

Testing a new slightly new sample, looks like it can generalize.

                cards expert_scores predicted_scores
0            defend_r          deck             deck
1            defend_r          deck             deck
2            defend_r          deck             deck
3            defend_r          deck             deck
4            strike_r          deck             deck
5            strike_r          deck             deck
6            strike_r          deck             deck
7                bash          deck             deck
8       ascendersbane          deck             deck
9            truegrit          deck             deck
10       pommelstrike          deck             deck
11  perfectedstrike+1          deck             deck
12  perfectedstrike+1          deck             deck
13    perfectedstrike          deck             deck
14        thunderclap          deck             deck
15    perfectedstrike          deck             deck
16        demonform+1          deck             deck
17               feed          deck             deck
18             disarm          deck             deck
19        darkembrace          deck             deck
20            havoc+1             1         0.999941
21       bloodletting             0         0.000094
22     swordboomerang             0         0.000003

