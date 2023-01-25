## Priority / wip

- Pad per batch, not once on the whole dataset. Use torch datasets and datloaders
- Train with latest modifs: mask pad tokens, relics

## .run processing

- Be careful of modded .run
- For now we drop out samples from the dataset that contain unknown cards
- Finetuning the processing of vanilla .run
- Add relics and some more state to the input of the model
- In second part of bakward_forward, resolving later floors first does not make sense because the accumulated delta is the total delta!!!
- Tinyhouse: skipped cards are not logged, the added card is put together with the rare pick
- .run processing: go from fuzzy logic to a constraint programming approach with backtracking

### Records of diff scores


#### old 2

(env) PS C:\Users\Lucas\Documents\Mes Projets\sts_ml> python .\sts_ml\deck_history.py
3: diff = 3 ; to add = [] ; to remove = [] ; to upgrade = ['j.a.x.', 'madness', 'madness']
10: diff = 1 ; to add = [] ; to remove = [] ; to upgrade = ['feed']
11: diff = 4 ; to add = ['exhume', 'entrench'] ; to remove = ['defend_r', 'defend_r'] ; to upgrade = []
12: diff = 1 ; to add = ['feelnopain+1'] ; to remove = [] ; to upgrade = []
13: diff = 1 ; to add = [] ; to remove = ['strike_r+1'] ; to upgrade = []
14: diff = 2 ; to add = ['strike_r'] ; to remove = ['strike_r+1'] ; to upgrade = []
20: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
21: diff = 5 ; to add = ['clumsy', 'reaper', 'rupture'] ; to remove = ['strike_r', 'strike_r'] ; to upgrade = []
23: diff = 1 ; to add = ['bloodletting+1'] ; to remove = [] ; to upgrade = []
26: diff = 2 ; to add = ['parasite'] ; to remove = [] ; to upgrade = ['havoc']
31: diff = 1 ; to add = ['battletrance'] ; to remove = [] ; to upgrade = []
32: diff = 5 ; to add = ['doubt', 'injury', 'demonform+1', 'immolate+1'] ; to remove = [] ; to upgrade = ['strike_r']
33: diff = 2 ; to add = ['impervious', 'wildstrike+1'] ; to remove = [] ; to upgrade = []
35: diff = 1 ; to add = ['forethought'] ; to remove = [] ; to upgrade = []
39: diff = 4 ; to add = ['feelnopain', 'disarm'] ; to remove = ['strike_r'] ; to upgrade = ['perfectedstrike']
41: diff = 3 ; to add = [] ; to remove = ['strike_r'] ; to upgrade = ['defend_r', 'swordboomerang']
43: diff = 1 ; to add = ['blind'] ; to remove = [] ; to upgrade = []
44: diff = 1 ; to add = [] ; to remove = [] ; to upgrade = ['bash']
45: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
49: diff = 2 ; to add = ['flex'] ; to remove = ['flex+1'] ; to upgrade = []
50: diff = 4 ; to add = ['clumsy'] ; to remove = ['curseofthebell'] ; to upgrade = ['impervious', 'seeingred']
51: diff = 2 ; to add = ['parasite'] ; to remove = [] ; to upgrade = ['bash']
52: diff = 1 ; to add = ['rupture'] ; to remove = [] ; to upgrade = []
53: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
54: diff = 1 ; to add = [] ; to remove = ['curseofthebell'] ; to upgrade = []
57: diff = 4 ; to add = ['reaper', 'anger', 'thunderclap'] ; to remove = ['defend_r+1'] ; to upgrade = []
58: diff = 1 ; to add = [] ; to remove = ['evolve+1'] ; to upgrade = []
60: diff = 1 ; to add = ['anger'] ; to remove = [] ; to upgrade = []
61: diff = 2 ; to add = [] ; to remove = ['defend_r'] ; to upgrade = ['strike_r']
62: diff = 2 ; to add = ['shame'] ; to remove = [] ; to upgrade = ['bash']
65: diff = 4 ; to add = ['searingblow+12', 'searingblow+4'] ; to remove = ['searingblow+3', 'searingblow+3'] ; to upgrade = []
66: diff = 1 ; to add = ['offering'] ; to remove = [] ; to upgrade = []
67: diff = 4 ; to add = ['corruption', 'seeingred'] ; to remove = ['defend_r', 'strike_r'] ; to upgrade = []
68: diff = 1 ; to add = ['pain'] ; to remove = [] ; to upgrade = []
71: diff = 1 ; to add = [] ; to remove = [] ; to upgrade = ['strike_r']
74: diff = 1 ; to add = [] ; to remove = [] ; to upgrade = ['bash']
75: diff = 4 ; to add = ['juggernaut'] ; to remove = ['defend_r', 'defend_r', 'defend_r'] ; to upgrade = []
77: diff = 1 ; to add = ['bludgeon+1'] ; to remove = [] ; to upgrade = []
78: diff = 1 ; to add = [] ; to remove = ['curseofthebell'] ; to upgrade = []
79: diff = 1 ; to add = ['bludgeon+1'] ; to remove = [] ; to upgrade = []
81: diff = 8 ; to add = ['feed', 'bloodforblood+1', 'limitbreak+1', 'berserk+1'] ; to remove = ['strike_r', 'strike_r', 'strike_r', 'curseofthebell'] ; to upgrade = []
89: diff = 4 ; to add = ['searingblow+5'] ; to remove = ['searingblow+4', 'searingblow+5', 'searingblow+5'] ; to upgrade = []
97: diff = 2 ; to add = ['injury', 'parasite'] ; to remove = [] ; to upgrade = []
99: diff = 4 ; to add = ['havoc', 'bodyslam'] ; to remove = ['strike_r', 'strike_r'] ; to upgrade = []
101: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
102: diff = 1 ; to add = ['impatience'] ; to remove = [] ; to upgrade = []
104: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
105: diff = 1 ; to add = ['offering'] ; to remove = [] ; to upgrade = []
107: diff = 1 ; to add = ['armaments+1'] ; to remove = [] ; to upgrade = []
114: diff = 27 ; to add = ['doubt', 'parasite', 'juggernaut+1', 'dualwield'] ; to remove = ['defend_r', 'defend_r', 'defend_r'] ; to upgrade = ['strike_r', 'strike_r', 'strike_r', 'bash', 'brutality', 'pommelstrike', 'recklesscharge', 'goodinstincts', 'pummel', 'intimidate', 'truegrit', 'wildstrike', 'evolve', 'barricade', 'wildstrike', 'violence', 'limitbreak', 'powerthrough', 'shockwave', 'juggernaut']
115: diff = 2 ; to add = [] ; to remove = [] ; to upgrade = ['defend_r', 'defend_r']
117: diff = 1 ; to add = ['brutality+1'] ; to remove = [] ; to upgrade = []
119: diff = 6 ; to add = ['decay', 'writhe', 'clumsy'] ; to remove = ['regret', 'regret', 'regret'] ; to upgrade = []
120: diff = 27 ; to add = ['cleave+1', 'cleave+1', 'cleave+1', 'cleave+1', 'cleave+1'] ; to remove = ['strike_r', 'strike_r', 'strike_r', 'strike_r', 'strike_r'] ; to upgrade = ['defend_r', 'defend_r', 'defend_r', 'defend_r', 'bash', 'flex', 'dropkick', 'bloodforblood', 'darkshackles', 'inflame', 'juggernaut', 'infernalblade', 'metallicize', 'flex', 'bloodforblood', 'panache', 'doubletap']
122: diff = 2 ; to add = [] ; to remove = [] ; to upgrade = ['defend_r', 'defend_r']
126: diff = 1 ; to add = ['limitbreak'] ; to remove = [] ; to upgrade = []
127: diff = 1 ; to add = ['brutality'] ; to remove = [] ; to upgrade = []
130: diff = 2 ; to add = ['panacea'] ; to remove = [] ; to upgrade = ['juggernaut']
131: diff = 4 ; to add = ['parasite', 'injury'] ; to remove = ['doubt', 'doubt'] ; to upgrade = []
132: diff = 4 ; to add = ['clumsy', 'regret', 'pain'] ; to remove = ['curseofthebell'] ; to upgrade = []
134: diff = 6 ; to add = ['twinstrike', 'burningpact+1'] ; to remove = ['defend_r', 'defend_r'] ; to upgrade = ['defend_r', 'strike_r']
135: diff = 1 ; to add = [] ; to remove = [] ; to upgrade = ['defend_r']
138: diff = 7 ; to add = ['bodyslam+1'] ; to remove = ['strike_r', 'strike_r', 'strike_r', 'pain', 'regret'] ; to upgrade = ['bash']
139: diff = 4 ; to add = ['powerthrough', 'swordboomerang', 'ghostlyarmor'] ; to remove = ['strike_r+1'] ; to upgrade = []
141: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
142: diff = 1 ; to add = ['searingblow+1'] ; to remove = [] ; to upgrade = []
144: diff = 5 ; to add = ['rampage+1', 'secondwind', 'twinstrike+1'] ; to remove = ['strike_r+1', 'strike_r+1'] ; to upgrade = []
145: diff = 1 ; to add = [] ; to remove = ['curseofthebell'] ; to upgrade = []
146: diff = 1 ; to add = ['exhume'] ; to remove = [] ; to upgrade = []
148: diff = 2 ; to add = [] ; to remove = ['defend_r', 'defend_r'] ; to upgrade = []
150: diff = 5 ; to add = ['offering', 'injury', 'clumsy'] ; to remove = ['curseofthebell'] ; to upgrade = ['inflame']
151: diff = 1 ; to add = ['metallicize+1'] ; to remove = [] ; to upgrade = []
157: diff = 5 ; to add = ['blind', 'doubletap', 'clash'] ; to remove = ['defend_r+1', 'strike_r+1'] ; to upgrade = []
160: diff = 2 ; to add = ['offering+1', 'whirlwind+1'] ; to remove = [] ; to upgrade = []
163: diff = 1 ; to add = ['impervious'] ; to remove = [] ; to upgrade = []
164: diff = 3 ; to add = ['parasite', 'doubt'] ; to remove = ['curseofthebell'] ; to upgrade = []
167: diff = 1 ; to add = ['flashofsteel'] ; to remove = [] ; to upgrade = []
170: diff = 3 ; to add = ['limitbreak+1', 'impervious'] ; to remove = ['strike_r'] ; to upgrade = []
173: diff = 8 ; to add = ['dramaticentrance', 'feed', 'anger', 'armaments', 'secondwind'] ; to remove = ['apotheosis', 'apotheosis'] ; to upgrade = ['apotheosis']
177: diff = 7 ; to add = ['trip', 'parasite', 'clumsy', 'parasite', 'regret'] ; to remove = ['curseofthebell', 'shame'] ; to upgrade = []
178: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
179: diff = 1 ; to add = ['dramaticentrance'] ; to remove = [] ; to upgrade = []
180: diff = 1 ; to add = ['demonform+1'] ; to remove = [] ; to upgrade = []
188: diff = 1 ; to add = ['limitbreak+1'] ; to remove = [] ; to upgrade = []
190: diff = 3 ; to add = ['injury', 'clumsy'] ; to remove = ['curseofthebell'] ; to upgrade = []
192: diff = 2 ; to add = ['shrugitoff'] ; to remove = ['strike_r'] ; to upgrade = []
193: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
195: diff = 1 ; to add = ['exhume'] ; to remove = [] ; to upgrade = []
197: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
198: diff = 2 ; to add = ['blind', 'firebreathing+1'] ; to remove = [] ; to upgrade = []
200: diff = 2 ; to add = ['searingblow+1'] ; to remove = ['strike_r'] ; to upgrade = []
201: diff = 1 ; to add = ['pain'] ; to remove = [] ; to upgrade = []
203: diff = 1 ; to add = [] ; to remove = [] ; to upgrade = ['defend_r']
204: diff = 4 ; to add = ['clumsy'] ; to remove = ['writhe', 'regret', 'regret'] ; to upgrade = []
210: diff = 5 ; to add = ['shrugitoff', 'thunderclap'] ; to remove = ['defend_r', 'defend_r', 'defend_r'] ; to upgrade = []
213: diff = 1 ; to add = [] ; to remove = [] ; to upgrade = ['defend_r']
214: diff = 2 ; to add = [] ; to remove = [] ; to upgrade = ['defend_r', 'powerthrough']
215: diff = 1 ; to add = [] ; to remove = [] ; to upgrade = ['bash']
217: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
218: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
220: diff = 1 ; to add = ['madness'] ; to remove = [] ; to upgrade = []
223: diff = 2 ; to add = ['finesse', 'goodinstincts'] ; to remove = [] ; to upgrade = []
226: diff = 2 ; to add = ['reaper', 'infernalblade+1'] ; to remove = [] ; to upgrade = []
227: diff = 4 ; to add = [] ; to remove = ['strike_r', 'strike_r', 'strike_r', 'regret'] ; to upgrade = []
228: diff = 10 ; to add = ['truegrit', 'bodyslam'] ; to remove = ['strike_r', 'strike_r', 'strike_r', 'strike_r'] ; to upgrade = ['defend_r', 'defend_r', 'defend_r', 'defend_r']
230: diff = 7 ; to add = ['ghostly+1', 'ghostly+1', 'ghostly+1', 'ghostly+1', 'ghostly+1', 'metallicize'] ; to remove = ['strike_r'] ; to upgrade = []
232: diff = 1 ; to add = ['reaper'] ; to remove = [] ; to upgrade = []
234: diff = 2 ; to add = ['reaper'] ; to remove = ['strike_r'] ; to upgrade = []
235: diff = 1 ; to add = ['demonform'] ; to remove = [] ; to upgrade = []
239: diff = 2 ; to add = ['swiftstrike', 'bandageup'] ; to remove = [] ; to upgrade = []
240: diff = 6 ; to add = ['twinstrike', 'offering', 'immolate+1'] ; to remove = ['defend_r', 'strike_r'] ; to upgrade = ['flamebarrier']
241: diff = 6 ; to add = ['parasite', 'regret'] ; to remove = ['strike_r', 'writhe', 'writhe', 'writhe'] ; to upgrade = []
243: diff = 5 ; to add = ['injury', 'doubt', 'writhe'] ; to remove = ['curseofthebell', 'decay'] ; to upgrade = []
244: diff = 13 ; to add = ['clothesline', 'havoc', 'havoc', 'normality', 'ironwave', 'inflame', 'battletrance'] ; to remove = ['pain', 'pain', 'pain', 'pain', 'pain', 'pain'] ; to upgrade = []
245: diff = 8 ; to add = ['feelnopain+1'] ; to remove = ['defend_r', 'strike_r', 'strike_r', 'immolate'] ; to upgrade = ['strike_r', 'bash', 'truegrit']
249: diff = 1 ; to add = ['juggernaut'] ; to remove = [] ; to upgrade = []
250: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
253: diff = 3 ; to add = ['hemokinesis+1', 'offering+1'] ; to remove = [] ; to upgrade = ['finesse']
256: diff = 2 ; to add = ['secrettechnique', 'madness'] ; to remove = [] ; to upgrade = []
257: diff = 5 ; to add = ['armaments', 'battletrance+1', 'clash+1'] ; to remove = ['defend_r', 'strike_r'] ; to upgrade = []
259: diff = 4 ; to add = ['injury', 'regret', 'regret'] ; to remove = ['curseofthebell'] ; to upgrade = []
261: diff = 4 ; to add = ['seversoul', 'havoc'] ; to remove = ['strike_r', 'strike_r'] ; to upgrade = []
262: diff = 1 ; to add = ['fiendfire'] ; to remove = [] ; to upgrade = []
263: diff = 1 ; to add = ['bodyslam'] ; to remove = [] ; to upgrade = []
265: diff = 1 ; to add = ['doubt'] ; to remove = [] ; to upgrade = []
266: diff = 3 ; to add = ['regret', 'decay'] ; to remove = ['curseofthebell'] ; to upgrade = []
268: diff = 1 ; to add = [] ; to remove = [] ; to upgrade = ['bash']
269: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
270: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
272: diff = 1 ; to add = ['madness'] ; to remove = [] ; to upgrade = []
273: diff = 7 ; to add = ['limitbreak+1', 'bloodforblood+1', 'metallicize+1', 'warcry+1'] ; to remove = ['defend_r', 'defend_r', 'defend_r'] ; to upgrade = []
278: diff = 4 ; to add = ['panacea', 'limitbreak', 'injury'] ; to remove = [] ; to upgrade = ['bash']
280: diff = 2 ; to add = ['dramaticentrance', 'parasite'] ; to remove = [] ; to upgrade = []
281: diff = 2 ; to add = ['offering'] ; to remove = ['strike_r'] ; to upgrade = []
282: diff = 9 ; to add = ['j.a.x.+1', 'flamebarrier+1', 'twinstrike+1', 'intimidate+1', 'corruption'] ; to remove = ['pain', 'pain', 'pain', 'pain'] ; to upgrade = []
285: diff = 3 ; to add = [] ; to remove = ['defend_r', 'bash'] ; to upgrade = ['infernalblade']
287: diff = 13 ; to add = ['entrench+1', 'bodyslam+1', 'barricade+1'] ; to remove = ['defend_r', 'defend_r', 'defend_r', 'defend_r', 'strike_r', 'strike_r', 'strike_r', 'strike_r', 'strike_r', 'bash'] ; to upgrade = []
290: diff = 3 ; to add = ['shockwave'] ; to remove = ['defend_r', 'strike_r'] ; to upgrade = []
293: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
295: diff = 1 ; to add = ['mayhem+1'] ; to remove = [] ; to upgrade = []
298: diff = 2 ; to add = [] ; to remove = [] ; to upgrade = ['strike_r', 'bash']
299: diff = 4 ; to add = ['reaper', 'flamebarrier'] ; to remove = ['defend_r', 'strike_r'] ; to upgrade = []
300: diff = 5 ; to add = ['armaments+1'] ; to remove = ['strike_r', 'strike_r', 'rampage', 'regret'] ; to upgrade = []
302: diff = 1 ; to add = ['truegrit'] ; to remove = [] ; to upgrade = []
304: diff = 3 ; to add = ['clumsy', 'clumsy'] ; to remove = ['curseofthebell'] ; to upgrade = []
305: diff = 4 ; to add = ['fiendfire', 'searingblow+3'] ; to remove = ['searingblow+2', 'searingblow+1'] ; to upgrade = []
310: diff = 2 ; to add = [] ; to remove = [] ; to upgrade = ['defend_r', 'defend_r']
312: diff = 2 ; to add = ['finesse', 'parasite'] ; to remove = [] ; to upgrade = []
313: diff = 4 ; to add = ['berserk', 'inflame'] ; to remove = ['defend_r', 'strike_r'] ; to upgrade = []
317: diff = 1 ; to add = ['panacea'] ; to remove = [] ; to upgrade = []
318: diff = 2 ; to add = ['heavyblade+1', 'secondwind'] ; to remove = [] ; to upgrade = []
321: diff = 4 ; to add = ['battletrance', 'searingblow'] ; to remove = ['defend_r', 'strike_r'] ; to upgrade = []
324: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
326: diff = 9 ; to add = ['dramaticentrance', 'writhe', 'doubt', 'regret', 'writhe', 'parasite'] ; to remove = ['curseofthebell', 'apotheosis'] ; to upgrade = ['apotheosis']
327: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
328: diff = 1 ; to add = ['impervious+1'] ; to remove = [] ; to upgrade = []
329: diff = 1 ; to add = ['bandageup+1'] ; to remove = [] ; to upgrade = []
330: diff = 2 ; to add = [] ; to remove = [] ; to upgrade = ['defend_r', 'defend_r']
331: diff = 2 ; to add = ['searingblow+12'] ; to remove = ['searingblow+11'] ; to upgrade = []
335: diff = 1 ; to add = ['parasite'] ; to remove = [] ; to upgrade = []
336: diff = 4 ; to add = ['writhe', 'regret', 'shame'] ; to remove = ['curseofthebell'] ; to upgrade = []
338: diff = 3 ; to add = [] ; to remove = ['defend_r', 'strike_r', 'anger'] ; to upgrade = []
341: diff = 2 ; to add = [] ; to remove = [] ; to upgrade = ['defend_r', 'spotweakness']
342: diff = 1 ; to add = [] ; to remove = ['strike_r'] ; to upgrade = []
343: diff = 2 ; to add = ['searingblow+10'] ; to remove = ['strike_r'] ; to upgrade = []
345: diff = 11 ; to add = ['demonform', 'writhe'] ; to remove = [] ; to upgrade = ['defend_r', 'defend_r', 'defend_r', 'defend_r', 'strike_r', 'strike_r', 'strike_r', 'strike_r', 'strike_r']
347: diff = 2 ; to add = ['parasite', 'parasite'] ; to remove = [] ; to upgrade = []
348: diff = 2 ; to add = [] ; to remove = [] ; to upgrade = ['madness', 'madness']
349: diff = 2 ; to add = ['parasite'] ; to remove = [] ; to upgrade = ['defend_r']
351: diff = 1 ; to add = ['rage+1'] ; to remove = [] ; to upgrade = []
355: diff = 2 ; to add = ['bandageup', 'deepbreath'] ; to remove = [] ; to upgrade = []
357: diff = 2 ; to add = ['panacea', 'apotheosis'] ; to remove = [] ; to upgrade = []
360: diff = 5 ; to add = ['pain', 'doubt', 'clumsy', 'parasite'] ; to remove = ['curseofthebell'] ; to upgrade = []
361: diff = 3 ; to add = ['purity', 'dropkick', 'ironwave'] ; to remove = [] ; to upgrade = []
362: diff = 2 ; to add = [] ; to remove = [] ; to upgrade = ['defend_r', 'defend_r']
364: diff = 1 ; to add = ['darkembrace+1'] ; to remove = [] ; to upgrade = []
367: diff = 1 ; to add = ['exhume'] ; to remove = [] ; to upgrade = []
368: diff = 1 ; to add = ['dropkick+1'] ; to remove = [] ; to upgrade = []
373: diff = 9 ; to add = ['thinkingahead', 'shame', 'injury', 'regret', 'violence', 'madness'] ; to remove = ['curseofthebell'] ; to upgrade = ['defend_r', 'defend_r']


#### old

Processed SlayTheData_win_a20_ic_21400.json.

##### 2023/01/24

Diff score over 21399 runs = 19044, 7140 could not be reconstructed.
Dumped dataset of 429404 samples and 556 tokens into ./SlayTheData_win_a20_ic_21400_837d844_21399.data

##### ??

```
Diff score over 21399 runs = 26569, 9274 could not be reconstructed.
Dumped dataset of 403870 samples into SlayTheData_win_a20_ic_21400.data
```

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

