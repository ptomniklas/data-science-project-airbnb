# data-science-project-airbnb

## Nicht versionierte Artefakte

Große Modell-, Cache- und Ergebnisdateien werden nicht in Git versioniert. Dazu
gehoeren unter anderem lokale HuggingFace-/AllenNLP-Caches, BERT-Gewichte,
Span-ASTE-Pretrained-Archive sowie erzeugte Dateien unter `data/processed/`,
`data/analysis/`, `data/annotations/` und `Span-ASTE/data_airbnb/`.

Diese Artefakte sollen lokal neu erzeugt, extern abgelegt oder bei Bedarf ueber
Git LFS verwaltet werden. Die Skripte und README-Befehle dokumentieren, wie die
Dateien reproduziert werden koennen.

## Review-Preprocessing

Das Skript `scripts/01_reviews_bereinigen.py` bereitet die Airbnb-Reviews
für Satz-basierte Textanalyse vor:

- liest `reviews.csv.gz`
- behält nur wahrscheinlich englische Reviews/Sätze
- entfernt HTML-Tags und normalisiert Leerzeichen/Satzzeichen
- splittet Reviews in einzelne Sätze
- behält Metadaten: `listing_id`, `review_id`, `sentence_id`, `date`
- filtert sehr kurze, sehr lange, kaputte oder nicht-englische Satzfragmente

Standardlauf:

```bash
python3 scripts/01_reviews_bereinigen.py
```

Ausgaben:

- `data/processed/reviews_english_sentences_clean.csv`
- `data/processed/reviews_english_sentences_summary.json`

Wichtige Parameter:

```bash
python3 scripts/01_reviews_bereinigen.py --min-words 4 --min-chars 20 --max-chars 600
```

## Span-ASTE Input vorbereiten

Das Skript `scripts/02_span_aste_eingabe_erstellen.py` konvertiert die bereinigten
englischen Review-Saetze in das Span-ASTE-Prediction-Format:

```text
sentence#### #### ####[]
```

Standardlauf fuer einen ersten Test mit 1.000 Saetzen:

```bash
python3 scripts/02_span_aste_eingabe_erstellen.py
```

Ausgaben:

- `Span-ASTE/data_airbnb/01_span_aste_eingabe_1000_saetze.txt`
- `Span-ASTE/data_airbnb/01_span_aste_eingabe_1000_saetze_metadaten.csv`
- `Span-ASTE/data_airbnb/01_span_aste_eingabe_1000_saetze_zusammenfassung.json`

Die Metadata-Datei enthaelt `listing_id`, `review_id`, `sentence_id` und
`date`, damit die spaeteren ASTE-Triplets wieder zu den Airbnb-Daten
zurueckgefuehrt werden koennen.

Full-Run fuer alle bereinigten Saetze:

```bash
python3 scripts/02_span_aste_eingabe_erstellen.py \
  --limit 0 \
  --output Span-ASTE/data_airbnb/airbnb_aste_input_full.txt \
  --metadata Span-ASTE/data_airbnb/airbnb_aste_input_full_metadata.csv \
  --summary Span-ASTE/data_airbnb/airbnb_aste_input_full_summary.json
```

## Span-ASTE Prediction und Parsing

Fuer den ersten Modelltest wurde das offizielle `pretrained_14res` Modell aus
dem Span-ASTE Release verwendet. Da die lokale Umgebung keine CUDA-GPU meldet,
laeuft die Prediction auf CPU.

Smoke-Test mit 25 Saetzen:

```bash
head -n 25 Span-ASTE/data_airbnb/01_span_aste_eingabe_1000_saetze.txt > \
  Span-ASTE/data_airbnb/00_testlauf_25_saetze_eingabe.txt

cd Span-ASTE
TRANSFORMERS_CACHE=.cache/transformers \
TORCH_HOME=.cache/torch \
ALLENNLP_CACHE_ROOT=.cache/allennlp \
PYTHONPATH=aste \
.venv/bin/python -c "from wrapper import SpanModel; model=SpanModel(save_dir='pretrained/pretrained_14res', random_seed=0); model.predict('data_airbnb/00_testlauf_25_saetze_eingabe.txt', 'data_airbnb/00_testlauf_25_saetze_modellausgabe.txt')"
```

Sample-Prediction mit 1.000 Saetzen:

```bash
cd Span-ASTE
TRANSFORMERS_CACHE=.cache/transformers \
TORCH_HOME=.cache/torch \
ALLENNLP_CACHE_ROOT=.cache/allennlp \
PYTHONPATH=aste \
.venv/bin/python -c "from wrapper import SpanModel; model=SpanModel(save_dir='pretrained/pretrained_14res', random_seed=0); model.predict('data_airbnb/01_span_aste_eingabe_1000_saetze.txt', 'data_airbnb/02_span_aste_modellausgabe_1000_saetze.txt')"
```

Prediction-Output in eine flache CSV parsen:

```bash
python3 scripts/03_span_aste_ausgabe_parsen.py \
  --input Span-ASTE/data_airbnb/02_span_aste_modellausgabe_1000_saetze.txt \
  --metadata Span-ASTE/data_airbnb/01_span_aste_eingabe_1000_saetze_metadaten.csv \
  --output Span-ASTE/data_airbnb/03_triplets_modell_roh_1000_saetze.csv \
  --summary Span-ASTE/data_airbnb/03_triplets_modell_roh_1000_saetze_zusammenfassung.json
```

Ergebnis des 1.000-Satz-Samples:

- 1.000 Saetze verarbeitet
- 500 Saetze ohne Triplet
- 722 extrahierte Triplets

## Span-ASTE Triplet-Qualitaetskontrolle

Das Skript `scripts/04_triplets_qualitaetspruefung.py` markiert extrahierte Triplets mit
`keep`, `review` oder `drop`, normalisiert Aspekte und ordnet sie groben
Airbnb-Kategorien zu.

```bash
python3 scripts/04_triplets_qualitaetspruefung.py
```

Ausgaben fuer das 1.000-Satz-Sample:

- `Span-ASTE/data_airbnb/04_triplets_qualitaetspruefung_1000_saetze.csv`
- `Span-ASTE/data_airbnb/05_triplets_bereinigt_1000_saetze.csv`
- `Span-ASTE/data_airbnb/04_aspekt_regeln_qualitaetspruefung.csv`
- `Span-ASTE/data_airbnb/04_triplets_qualitaetspruefung_1000_saetze_zusammenfassung.json`

Ergebnis der regelbasierten QC:

- 722 Triplets insgesamt
- 680 automatisch nutzbare Triplets (`keep`)
- 40 verworfene Triplets (`drop`), vor allem Hostnamen/Personennamen und
  nicht-listingbezogene Aspekte
- 2 manuell zu pruefende Triplets (`review`)

## Span-ASTE Gold-Annotation vorbereiten

Die 1.000 Sample-Saetze sind bisher Modellvorhersagen, keine menschlichen
Gold-Labels. Fuer Fine-Tuning und eine saubere Evaluation wird deshalb eine
Pilot-Annotation vorbereitet.

```bash
python3 scripts/05_pilot_annotation_erstellen.py
```

Ausgaben:

- `data/annotations/span_aste_pilot_annotation_50.csv`
- `data/annotations/span_aste_pilot_annotation_50_summary.json`
- `data/annotations/span_aste_annotation_guidelines.md`

Die Pilotdatei enthaelt 50 Saetze: 40 mit Modellvorschlaegen und 10 ohne
Modell-Triplet. Die Spalten `model_*` dienen nur als Hilfe; die eigentlichen
manuellen Labels werden in den `gold_*` Spalten eingetragen.

Fuer die erweiterte Gold-Annotation wurde zusaetzlich eine 1.000er-Datei
angelegt:

```bash
python3 scripts/05_pilot_annotation_erstellen.py \
  --sample-size 1000 \
  --output data/annotations/span_aste_annotation_1000.csv \
  --summary data/annotations/span_aste_annotation_1000_summary.json
```

Diese Datei enthaelt 500 Saetze mit Modellvorschlaegen und 500 Saetze ohne
Modell-Triplet. Dadurch koennen sowohl falsche Modellvorschlaege korrigiert
als auch vom Modell verpasste Aspect-Opinion-Sentiment-Triplets erfasst werden.

Als schnelle Arbeitsgrundlage kann die 1.000er-Datei mit konservativen
Modellvorschlaegen vorbefuellt werden. Diese Silver-Datei ersetzt keine
menschliche Gold-Annotation:

```bash
python3 scripts/06_annotation_mit_modell_vorbefuellen.py
```

Ausgaben:

- `data/annotations/span_aste_annotation_1000_vorbefuellt.csv`
- `data/annotations/span_aste_annotation_1000_vorbefuellt_summary.json`

Ergebnis des Keep-only-Prefills:

- 1.000 Saetze verarbeitet
- 477 Saetze vorbefuellt
- 680 Triplets vorbefuellt
- 523 Saetze leer gelassen
- 42 Modell-Triplets uebersprungen

Die automatisch berechneten Token-Spans fuer diese vorbefuellte Datei wurden
mit folgendem Befehl erzeugt:

```bash
python3 scripts/08_token_spans_ergaenzen.py \
  --input data/annotations/span_aste_annotation_1000_vorbefuellt.csv \
  --output data/annotations/span_aste_annotation_1000_vorbefuellt.csv \
  --summary data/annotations/span_aste_annotation_1000_vorbefuellt_spans_summary.json
```

Fuer Zeilen ohne Modellvorschlag wurde zusaetzlich ein konservativer
regelbasierter Ergaenzungslauf erstellt:

```bash
python3 scripts/07_leere_annotationen_ergaenzen.py
```

Ausgaben:

- `data/annotations/span_aste_annotation_1000_ergaenzt.csv`
- `data/annotations/span_aste_annotation_1000_ergaenzt_summary.json`
- `data/annotations/span_aste_annotation_1000_ergaenzt_spans_summary.json`

Ergebnis nach Modell-Prefill plus Ergaenzung:

- 1.000 Saetze verarbeitet
- 704 Saetze mit Triplets
- 296 Saetze ohne Triplets
- 1.087 Triplets mit Token-Spans

Auch diese Datei bleibt eine Silver-Datei mit Review-Bedarf. Die 296
leeren Saetze enthalten vor allem faktische, nicht Airbnb-aspektbezogene oder
zu implizite Aussagen, bei denen eine automatische Annotation zu unsicher waere.

Nach der manuellen Annotation werden die Token-Positionen reproduzierbar aus
den Gold-Texten berechnet:

```bash
python3 scripts/08_token_spans_ergaenzen.py
```

Das Skript fuellt `gold_aspect_start`, `gold_aspect_end`,
`gold_opinion_start` und `gold_opinion_end` in der Annotationstabelle und
schreibt eine kurze Zusammenfassung nach
`data/annotations/span_aste_pilot_annotation_50_spans_summary.json`.

## Span-ASTE Pilot-Evaluation

Das Skript `scripts/09_pilot_annotation_auswerten.py` vergleicht die manuell
annotierten Gold-Triplets aus der Pilotdatei mit den Modellvorschlaegen.
Standardmaessig werden nur Modell-Triplets mit `qc_status=keep` bewertet:

```bash
python3 scripts/09_pilot_annotation_auswerten.py
```

Ausgaben:

- `data/annotations/span_aste_pilot_evaluation_50.csv`
- `data/annotations/span_aste_pilot_evaluation_50_summary.json`

Ergebnis der Keep-only-Evaluation:

- 103 Gold-Triplets
- 60 Modell-Triplets
- 33 exakte Triplet-Treffer
- Precision: 0.55
- Recall: 0.3204
- F1: 0.4049
- 7 Saetze mit vollstaendig exaktem Match
- 22 Saetze mit teilweisem Match
- 10 Saetze, in denen das Modell alle Gold-Triplets verpasst hat

Zum Vergleich koennen auch Modell-Triplets mit `review` oder `drop`
eingeschlossen werden:

```bash
python3 scripts/09_pilot_annotation_auswerten.py \
  --all-model-triplets \
  --output data/annotations/span_aste_pilot_evaluation_50_all_model_triplets.csv \
  --summary data/annotations/span_aste_pilot_evaluation_50_all_model_triplets_summary.json
```

Diese Variante erzielt F1 0.4260 bei 36 exakten Treffern, erzeugt aber auch
mehr zusaetzliche Modell-Triplets. Fuer die weitere Analyse ist deshalb die
Keep-only-Datei die konservativere Arbeitsgrundlage.

## 450er Gold-Review-Split vorbereiten

Aus der 1.000er Arbeitsdatei wird ein reproduzierbarer 450-Satz-Split fuer
Train/Dev/Test vorbereitet:

```bash
python3 scripts/11_gold_split_450_erstellen.py
```

Ausgaben:

- `data/annotations/span_aste_gold_review_450_split.csv`
- `data/annotations/span_aste_gold_review_450_split_summary.json`
- `Span-ASTE/data_airbnb/gold_review_450/airbnb_gold_450_train.txt`
- `Span-ASTE/data_airbnb/gold_review_450/airbnb_gold_450_dev.txt`
- `Span-ASTE/data_airbnb/gold_review_450/airbnb_gold_450_test.txt`

Der Split enthaelt exakt 300/75/75 Saetze. NEG und NEU werden vollstaendig
uebernommen, damit die kleinen Evaluationssplits nicht nur positive Beispiele
enthalten:

- Train: 300 Saetze, davon 19 NEG, 18 NEU, 83 ohne Triplet, 180 POS
- Dev: 75 Saetze, davon 5 NEG, 5 NEU, 21 ohne Triplet, 44 POS
- Test: 75 Saetze, davon 5 NEG, 4 NEU, 20 ohne Triplet, 46 POS

Wichtig: Diese Dateien sind noch nicht als finales Gold freigegeben. Alle
Zeilen tragen `gold_review_status=review_required`; `manual_review_ok` und
`manual_review_notes` sind fuer die manuelle Pruefung vorgesehen. Erst nach
dieser Pruefung sollten die Span-ASTE-Dateien fuer Fine-Tuning oder eine
finale Evaluation verwendet werden.

Wenn die Annotation transparent als KI-unterstuetzte Silver-Annotation
weiterverwendet werden soll, kann ein separater KI-Review-Lauf erzeugt werden:

```bash
python3 scripts/13_ki_review_gold_split_450.py

python3 scripts/08_token_spans_ergaenzen.py \
  --input data/annotations/span_aste_gold_review_450_split_ki_geprueft.csv \
  --output data/annotations/span_aste_gold_review_450_split_ki_geprueft.csv \
  --summary data/annotations/span_aste_gold_review_450_split_ki_geprueft_spans_summary.json

python3 scripts/14_gold_split_zu_span_aste_exportieren.py
```

Ausgaben:

- `data/annotations/span_aste_gold_review_450_split_ki_geprueft.csv`
- `data/annotations/span_aste_gold_review_450_split_ki_geprueft_summary.json`
- `data/annotations/span_aste_gold_review_450_split_ki_geprueft_spans_summary.json`
- `data/annotations/span_aste_gold_review_450_split_ki_geprueft_export_summary.json`
- `Span-ASTE/data_airbnb/gold_review_450_ki_geprueft/airbnb_gold_450_ki_geprueft_train.txt`
- `Span-ASTE/data_airbnb/gold_review_450_ki_geprueft/airbnb_gold_450_ki_geprueft_dev.txt`
- `Span-ASTE/data_airbnb/gold_review_450_ki_geprueft/airbnb_gold_450_ki_geprueft_test.txt`

Der KI-Review markiert alle Zeilen weiterhin als
`ai_reviewed_human_check_required`. Die Spalte `manual_review_ok` bleibt fuer
eine echte menschliche Endkontrolle reserviert. Im aktuellen Lauf wurden 17
zusaetzliche klare Triplet-Zeilen ergaenzt, 2 offensichtliche Triplets
korrigiert und 24 unsichere Zeilen fuer menschliche Entscheidung markiert.

## Erster Airbnb-Trainings- und Testlauf

Der lokale Span-ASTE-Wrapper wurde fuer CPU-Training angepasst:

- `cuda_device=-1` wird nun auch beim Training in die AllenNLP-Config
  uebernommen.
- `num_epochs` und `batch_size` koennen fuer Testlaeufe ueberschrieben werden.
- Batch Size groesser als 1 ist mit diesem Span-ASTE-Code nicht moeglich
  (`Multi-document minibatching not yet supported`).

Baseline-Test mit dem offiziellen `pretrained_14res` Modell auf dem
KI-geprueften Airbnb-Testsplit:

```text
Precision: 0.9231
Recall:    0.6000
F1:        0.7273
```

Die Baseline-Evaluation ist jetzt auch als reproduzierbarer exakter
Triplet-Vergleich dokumentiert:

```bash
python3 scripts/15_pretrained_baseline_evaluieren.py
```

Ausgaben:

- `data/annotations/pretrained_14res_ki_geprueft_test_evaluation.csv`
- `data/annotations/pretrained_14res_ki_geprueft_test_evaluation_summary.json`

Aktueller Stand auf dem KI-geprueften Silver-/Weak-Gold-Testsplit:

- 75 Testsaetze
- 100 Silver-Gold-Triplets
- 65 vorhergesagte Triplets
- 60 exakte Triplet-Treffer
- Precision: 0.9231
- Recall: 0.6000
- F1: 0.7273

Wichtig fuer die Interpretation: Der Split ist transparent als
`ki_gepruefter_silver_weak_gold_split_human_check_required` markiert. Das ist
eine belastbare Arbeitsgrundlage fuer Modellvergleich und Pipeline-Entscheidung,
aber keine finale menschliche Gold-Annotation.

Ein erster Airbnb-Trainingslauf wurde mit 1 CPU-Epoche ausgefuehrt:

```bash
cd Span-ASTE
TRANSFORMERS_CACHE=.cache/transformers \
TORCH_HOME=.cache/torch \
ALLENNLP_CACHE_ROOT=.cache/allennlp \
PYTHONPATH=aste \
.venv/bin/python aste/wrapper.py run_train \
  --path_train data_airbnb/gold_review_450_ki_geprueft/airbnb_gold_450_ki_geprueft_train.txt \
  --path_dev data_airbnb/gold_review_450_ki_geprueft/airbnb_gold_450_ki_geprueft_dev.txt \
  --save_dir outputs/airbnb_ki_geprueft_epoch1/seed_0 \
  --random_seed 0 \
  --num_epochs 1
```

Das Modell wurde nach
`Span-ASTE/outputs/airbnb_ki_geprueft_epoch1/seed_0/weights/model.tar.gz`
geschrieben. Der Testvergleich liegt in:

- `Span-ASTE/data_airbnb/gold_review_450_ki_geprueft/airbnb_epoch1_seed0_testvergleich.json`
- `Span-ASTE/data_airbnb/gold_review_450_ki_geprueft/pretrained_14res_test_pred.txt`
- `Span-ASTE/data_airbnb/gold_review_450_ki_geprueft/airbnb_epoch1_seed0_test_pred.txt`

Ergebnis des 1-Epochen-Airbnb-Modells:

```text
Precision: 1.0000
Recall:    0.0100
F1:        0.0198
```

Interpretation: Der 1-Epochen-Lauf ist nur ein technischer Nachweis, dass
Training, Modellarchivierung und Testauswertung lokal funktionieren. Das Modell
ist nach einer Epoche extrem konservativ und klar schlechter als das
vortrainierte `14res`-Modell. Ausserdem wird hier nicht direkt vom
`pretrained_14res`-Checkpoint weitertrainiert, sondern mit der Span-ASTE-Config
ein Airbnb-Modell auf Basis von `bert-base-uncased` trainiert.

## Triplets pro Listing aggregieren

Die flache Triplet-Datei kann zu Listing-Merkmalen aggregiert und mit
Listing-Stammdaten verknuepft werden:

```bash
python3 scripts/12_triplets_pro_listing_aggregieren.py
```

Ausgaben:

- `data/processed/airbnb_aste_listing_aggregation_1000_sample.csv`
- `data/processed/airbnb_aste_listing_aggregation_1000_sample_summary.json`

Die Tabelle enthaelt pro `listing_id` unter anderem:

- Anzahl Reviews, Saetze und Triplets mit ASTE-Ergebnis
- POS/NEU/NEG-Verteilung und Net-Sentiment-Score
- haeufigste Themen insgesamt sowie haeufigste positive/negative Themen
- haeufigste Aspekte insgesamt sowie haeufigste positive/negative Aspekte
- ausgewaehlte Listing-Merkmale wie Preis, Lage, Room Type und Ratings

Aktueller Stand der Sample-Aggregation: 1.050 nutzbare Triplets aus 4 Listings.
Das ist nur die Struktur fuer die spaetere Analyse; fuer belastbare Ergebnisse
muss danach der Full-Run ueber alle bereinigten englischen Saetze ausgefuehrt
und mit demselben Skript aggregiert werden.

## Full-Run vorbereiten

Die Eingabe fuer den Full-Run ueber alle bereinigten englischen Saetze wurde
erzeugt:

```bash
python3 scripts/02_span_aste_eingabe_erstellen.py \
  --limit 0 \
  --output Span-ASTE/data_airbnb/airbnb_aste_input_full.txt \
  --metadata Span-ASTE/data_airbnb/airbnb_aste_input_full_metadata.csv \
  --summary Span-ASTE/data_airbnb/airbnb_aste_input_full_summary.json
```

Ausgabe-Status:

- 1.124.837 Saetze geschrieben
- `Span-ASTE/data_airbnb/airbnb_aste_input_full.txt`
- `Span-ASTE/data_airbnb/airbnb_aste_input_full_metadata.csv`
- `Span-ASTE/data_airbnb/airbnb_aste_input_full_summary.json`

Der naechste rechenintensive Schritt ist die Prediction mit dem Hauptmodell
`pretrained_14res`:

```bash
cd Span-ASTE
TRANSFORMERS_CACHE=.cache/transformers \
TORCH_HOME=.cache/torch \
ALLENNLP_CACHE_ROOT=.cache/allennlp \
PYTHONPATH=aste \
.venv/bin/python -c "from wrapper import SpanModel; model=SpanModel(save_dir='pretrained/pretrained_14res', random_seed=0); model.predict('data_airbnb/airbnb_aste_input_full.txt', 'data_airbnb/airbnb_aste_output_full_pretrained_14res.txt')"
```

Danach werden die Full-Run-Triplets analog zum Sample geparst, qualitaetsgeprueft
und aggregiert:

```bash
python3 scripts/03_span_aste_ausgabe_parsen.py \
  --input Span-ASTE/data_airbnb/airbnb_aste_output_full_pretrained_14res.txt \
  --metadata Span-ASTE/data_airbnb/airbnb_aste_input_full_metadata.csv \
  --output Span-ASTE/data_airbnb/airbnb_aste_triplets_full_roh.csv \
  --summary Span-ASTE/data_airbnb/airbnb_aste_triplets_full_roh_summary.json

python3 scripts/04_triplets_qualitaetspruefung.py \
  --input Span-ASTE/data_airbnb/airbnb_aste_triplets_full_roh.csv \
  --output Span-ASTE/data_airbnb/airbnb_aste_triplets_full_qc.csv \
  --clean-output Span-ASTE/data_airbnb/airbnb_aste_triplets_full_clean.csv \
  --aspect-rules Span-ASTE/data_airbnb/airbnb_aste_aspekt_regeln_full.csv \
  --summary Span-ASTE/data_airbnb/airbnb_aste_triplets_full_qc_summary.json

python3 scripts/12_triplets_pro_listing_aggregieren.py \
  --input Span-ASTE/data_airbnb/airbnb_aste_triplets_full_qc.csv \
  --output data/processed/airbnb_aste_listing_aggregation_full.csv \
  --summary data/processed/airbnb_aste_listing_aggregation_full_summary.json
```

## Vergleich mit Rating, Preis und Lage

Fuer die 1.000er-Strukturprobe wurden Vergleichstabellen und SVG-Grafiken
erzeugt:

```bash
python3 scripts/16_listing_sentiment_mit_listings_vergleichen.py
```

Ausgaben in `data/analysis/listing_sentiment_1000_sample/`:

- `korrelationen_sentiment_listing_merkmale.csv`
- `gruppenvergleich_lage_roomtype.csv`
- `themen_haeufigkeiten.csv`
- `grafik_top_themen.svg`
- `grafik_sentiment_nach_lage.svg`
- `grafik_rating_vs_sentiment.svg`
- `listing_sentiment_vergleich_summary.json`

Nach der Full-Aggregation wird dasselbe Skript auf die volle Listing-Datei
angewendet:

```bash
python3 scripts/16_listing_sentiment_mit_listings_vergleichen.py \
  --input data/processed/airbnb_aste_listing_aggregation_full.csv \
  --output-dir data/analysis/listing_sentiment_full \
  --min-triplets 3
```
