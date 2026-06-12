# Span-ASTE Pilot-Annotation

Ziel: Aus den Modellvorhersagen echte Gold-Labels machen. Die Spalten
`model_*` sind nur Vorschlaege des pretrained Span-ASTE-Modells und duerfen
korrigiert, ignoriert oder erweitert werden.

## Was annotieren?

Annotiere ein Triplet, wenn im Satz ein Airbnb-relevanter Aspekt bewertet wird:

- `gold_aspect_text`: worueber gesprochen wird, z. B. `location`, `host`,
  `bed`, `bathroom`, `street`, `check-in`.
- `gold_opinion_text`: bewertender Ausdruck, z. B. `perfect`, `friendly`,
  `noisy`, `uncomfortable`.
- `gold_sentiment`: `POS`, `NEG` oder `NEU`.

Mehrere Triplets pro Satz sind erlaubt. Wenn ein Satz mehrere Gold-Triplets
hat, trage sie mit ` ; ` getrennt ein, jeweils in derselben Reihenfolge in den
Gold-Spalten.

Beispiel:

```text
gold_aspect_text: location ; street
gold_opinion_text: perfect ; noisy
gold_sentiment: POS ; NEG
```

## Was nicht annotieren?

- Rein faktische Aussagen ohne Bewertung: `The apartment has two bedrooms.`
- Aussagen ohne Airbnb-relevanten Aspekt: Smalltalk, Reisebegleitung,
  persoenliche Umstaende.
- Host- oder Personennamen als Aspekt, wenn eigentlich die Person/Kommunikation
  gemeint ist. Dann lieber `host` oder `communication` annotieren.

## Neutrale Faelle

`NEU` nur verwenden, wenn eine erkennbare Bewertung vorhanden ist, aber nicht
klar positiv oder negativ ist.

Beispiele:

- `The room was okay.` -> `room`, `okay`, `NEU`
- `The apartment has two bedrooms.` -> kein Triplet

## Token-Positionen

Die Token-Positionen koennen zunaechst leer bleiben. Wichtig ist, dass
`gold_aspect_text` und `gold_opinion_text` exakt im Satz vorkommen. Dann lassen
sich `gold_aspect_start`, `gold_aspect_end`, `gold_opinion_start` und
`gold_opinion_end` spaeter automatisch berechnen.

## Schwierige Faelle

Setze `difficult` auf `yes`, wenn der Satz mehrdeutig ist oder die Entscheidung
unsicher bleibt. Nutze `notes` fuer kurze Begruendungen.
