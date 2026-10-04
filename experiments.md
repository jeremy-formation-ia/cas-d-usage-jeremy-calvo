# Journal des expériences — Marketing bancaire

Traçage des runs de modélisation (règle d'or : reproductibilité, décision, comparabilité).
Tous les runs partagent le **même split stratifié** (80/20, `random_state=42`) et la même
**validation croisée stratifiée** (5 folds) sur le train. Le jeu de test reste scellé.

- **Dataset** : `data/bank-additional-full.csv` (sha256 `74adfc57…`)
- **Métrique de référence** : F1 macro
- **Modèles** : LogisticRegression, RandomForest, HistGradientBoosting (`class_weight='balanced'`)
- **Scénarios** : S1 (toutes) ⊃ S2 (−`duration`) ⊃ S3 (−sensibles) ⊃ S4 (−historique)

## Runs (validation croisée, moyenne sur 5 folds)

| Run | Scénario | Modèle | F1 macro | ROC-AUC | Recall « oui » | Précision « oui » | Verdict |
|---|---|---|---|---|---|---|---|
| exp_001 | S1 | LogisticRegression | 0,750 | 0,937 | 0,882 | 0,438 | Écarté — fuite (`duration`) |
| exp_002 | S1 | RandomForest | 0,730 | 0,942 | 0,412 | 0,668 | Écarté — fuite |
| exp_003 | S1 | HistGradientBoosting | **0,766** | **0,949** | 0,922 | 0,457 | Écarté — fuite |
| exp_004 | S2 | LogisticRegression | 0,675 | 0,790 | 0,623 | 0,354 | Référence technique |
| exp_005 | S2 | RandomForest | 0,657 | 0,773 | 0,280 | 0,554 | Écarté — recall faible |
| exp_006 | S2 | HistGradientBoosting | **0,691** | **0,799** | 0,621 | 0,382 | Retenu (variante) |
| exp_007 | S3 | LogisticRegression | 0,675 | 0,791 | 0,620 | 0,354 | Écarté |
| exp_008 | S3 | RandomForest | 0,627 | 0,714 | 0,407 | 0,310 | Écarté |
| exp_009 | S3 | HistGradientBoosting | **0,685** | **0,802** | 0,631 | 0,369 | Alternative éthique |
| exp_010 | S4 | LogisticRegression | 0,677 | 0,783 | 0,602 | 0,361 | Écarté |
| exp_011 | S4 | RandomForest | 0,655 | 0,723 | 0,501 | 0,341 | Écarté |
| exp_012 | S4 | HistGradientBoosting | **0,683** | **0,796** | 0,626 | 0,368 | **Retenu** |

## Décisions

- **Famille** : HistGradientBoosting (meilleur F1 macro / ROC-AUC sur les quatre scénarios).
- **Scénario** : S4 — sans `duration` (fuite), sans variables sensibles (éthique), sans historique de campagne. Performance quasi identique à S3/S2 pour un périmètre minimal.
- **Seuil** : 0,30 (priorité au rappel, ajusté en validation croisée sur le train).
- **Métriques sur le test scellé** (évalué une seule fois) : F1 macro 0,477, ROC-AUC 0,807, recall « oui » 0,851, précision « oui » 0,178.

## Reproduction

```bash
python -m src.modeling     # benchmark modèles × scénarios
python -m src.train        # entraîne et persiste le modèle retenu
```
