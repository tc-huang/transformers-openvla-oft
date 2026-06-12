<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Wie konvertiert man ein 🤗 Transformers-Modell in TensorFlow?

Die Tatsache, dass mehrere Frameworks für die Verwendung mit 🤗 Transformers zur Verfügung stehen, gibt Ihnen die Flexibilität, deren Stärken beim Entwurf Ihrer Anwendung auszuspielen.
Ihre Anwendung zu entwerfen, aber das bedeutet auch, dass die Kompatibilität für jedes Modell einzeln hinzugefügt werden muss. Die gute Nachricht ist, dass
das Hinzufügen von TensorFlow-Kompatibilität zu einem bestehenden Modell einfacher ist als [das Hinzufügen eines neuen Modells von Grund auf](add_new_model)!
Ob Sie ein tieferes Verständnis für große TensorFlow-Modelle haben möchten, einen wichtigen Open-Source-Beitrag leisten oder
TensorFlow für das Modell Ihrer Wahl aktivieren wollen, dieser Leitfaden ist für Sie.

Dieser Leitfaden befähigt Sie, ein Mitglied unserer Gemeinschaft, TensorFlow-Modellgewichte und/oder
Architekturen beizusteuern, die in 🤗 Transformers verwendet werden sollen, und zwar mit minimaler Betreuung durch das Hugging Face Team. Das Schreiben eines neuen Modells
ist keine Kleinigkeit, aber ich hoffe, dass dieser Leitfaden dazu beiträgt, dass es weniger eine Achterbahnfahrt 🎢 und mehr ein Spaziergang im Park 🚶 ist.
Die Nutzung unserer kollektiven Erfahrungen ist absolut entscheidend, um diesen Prozess immer einfacher zu machen, und deshalb möchten wir
ermutigen Sie daher, Verbesserungsvorschläge für diesen Leitfaden zu machen!

Bevor Sie tiefer eintauchen, empfehlen wir Ihnen, die folgenden Ressourcen zu lesen, wenn Sie neu in 🤗 Transformers sind:
- [Allgemeiner Überblick über 🤗 Transformers](add_new_model#general-overview-of-transformers)
- [Die TensorFlow-Philosophie von Hugging Face](https://huggingface.co/blog/tensorflow-philosophy)

Im Rest dieses Leitfadens werden Sie lernen, was nötig ist, um eine neue TensorFlow Modellarchitektur hinzuzufügen, die
Verfahren zur Konvertierung von PyTorch in TensorFlow-Modellgewichte und wie Sie Unstimmigkeiten zwischen ML
Frameworks. Legen Sie los!

<Tip>

Sind Sie unsicher, ob das Modell, das Sie verwenden möchten, bereits eine entsprechende TensorFlow-Architektur hat?

&nbsp;

Überprüfen Sie das Feld `model_type` in der `config.json` des Modells Ihrer Wahl
([Beispiel](https://huggingface.co/google-bert/bert-base-uncased/blob/main/config.json#L14)). Wenn der entsprechende Modellordner in
🤗 Transformers eine Datei hat, deren Name mit "modeling_tf" beginnt, bedeutet dies, dass es eine entsprechende TensorFlow
Architektur hat ([Beispiel](https://github.com/huggingface/transformers/tree/main/src/transformers_openvla_oft/models/bert)).

</Tip>


## Schritt-für-Schritt-Anleitung zum Hinzufügen von TensorFlow-Modellarchitektur-Code

Es gibt viele Möglichkeiten, eine große Modellarchitektur zu entwerfen, und viele Möglichkeiten, diesen Entwurf zu implementieren. Wie auch immer,
Sie erinnern sich vielleicht an unseren [allgemeinen Überblick über 🤗 Transformers](add_new_model#general-overview-of-transformers)
wissen, dass wir ein meinungsfreudiger Haufen sind - die Benutzerfreundlichkeit von 🤗 Transformers hängt von konsistenten Designentscheidungen ab. Aus
Erfahrung können wir Ihnen ein paar wichtige Dinge über das Hinzufügen von TensorFlow-Modellen sagen:

- Erfinden Sie das Rad nicht neu! In den meisten Fällen gibt es mindestens zwei Referenzimplementierungen, die Sie überprüfen sollten: das
PyTorch-Äquivalent des Modells, das Sie implementieren, und andere TensorFlow-Modelle für dieselbe Klasse von Problemen.
- Gute Modellimplementierungen überleben den Test der Zeit. Dies geschieht nicht, weil der Code hübsch ist, sondern eher
sondern weil der Code klar, einfach zu debuggen und darauf aufzubauen ist. Wenn Sie den Maintainern das Leben mit Ihrer
TensorFlow-Implementierung leicht machen, indem Sie die gleichen Muster wie in anderen TensorFlow-Modellen nachbilden und die Abweichung
zur PyTorch-Implementierung minimieren, stellen Sie sicher, dass Ihr Beitrag lange Bestand haben wird.
- Bitten Sie um Hilfe, wenn Sie nicht weiterkommen! Das 🤗 Transformers-Team ist da, um zu helfen, und wir haben wahrscheinlich Lösungen für die gleichen
Probleme gefunden, vor denen Sie stehen.

Hier finden Sie einen Überblick über die Schritte, die zum Hinzufügen einer TensorFlow-Modellarchitektur erforderlich sind:
1. Wählen Sie das Modell, das Sie konvertieren möchten
2. Bereiten Sie die Transformers-Entwicklungsumgebung vor.
3. (Optional) Verstehen Sie die theoretischen Aspekte und die bestehende Implementierung
4. Implementieren Sie die Modellarchitektur
5. Implementieren Sie Modelltests
6. Reichen Sie den Pull-Antrag ein
7. (Optional) Erstellen Sie Demos und teilen Sie diese mit der Welt

### 1.-3. Bereiten Sie Ihren Modellbeitrag vor

**1. Wählen Sie das Modell, das Sie konvertieren möchten**

Beginnen wir mit den Grundlagen: Als erstes müssen Sie die Architektur kennen, die Sie konvertieren möchten. Wenn Sie
Sie sich nicht auf eine bestimmte Architektur festgelegt haben, ist es eine gute Möglichkeit, das 🤗 Transformers-Team um Vorschläge zu bitten.
Wir werden Sie zu den wichtigsten Architekturen führen, die auf der TensorFlow-Seite noch fehlen.
Seite fehlen. Wenn das spezifische Modell, das Sie mit TensorFlow verwenden möchten, bereits eine Implementierung der TensorFlow-Architektur in
🤗 Transformers, aber es fehlen Gewichte, können Sie direkt in den
Abschnitt [Gewichtskonvertierung](#hinzufügen-von-tensorflow-gewichten-zum--hub)
auf dieser Seite.

Der Einfachheit halber wird im Rest dieser Anleitung davon ausgegangen, dass Sie sich entschieden haben, mit der TensorFlow-Version von
*BrandNewBert* (dasselbe Beispiel wie in der [Anleitung](add_new_model), um ein neues Modell von Grund auf hinzuzufügen).

<Tip>

Bevor Sie mit der Arbeit an einer TensorFlow-Modellarchitektur beginnen, sollten Sie sich vergewissern, dass es keine laufenden Bemühungen in dieser Richtung gibt.
Sie können nach `BrandNewBert` auf der
[pull request GitHub page](https://github.com/huggingface/transformers/pulls?q=is%3Apr), um zu bestätigen, dass es keine
TensorFlow-bezogene Pull-Anfrage gibt.

</Tip>


**2. Transformers-Entwicklungsumgebung vorbereiten**

Nachdem Sie die Modellarchitektur ausgewählt haben, öffnen Sie einen PR-Entwurf, um Ihre Absicht zu signalisieren, daran zu arbeiten. Folgen Sie den
Anweisungen, um Ihre Umgebung einzurichten und einen PR-Entwurf zu öffnen.

1. Forken Sie das [repository](https://github.com/huggingface/transformers), indem Sie auf der Seite des Repositorys auf die Schaltfläche 'Fork' klicken.
   Seite des Repositorys klicken. Dadurch wird eine Kopie des Codes unter Ihrem GitHub-Benutzerkonto erstellt.

2. Klonen Sie Ihren `transformers` Fork auf Ihre lokale Festplatte und fügen Sie das Basis-Repository als Remote hinzu:

```bash
git clone https://github.com/[your Github handle]/transformers.git
cd transformers
git remote add upstream https://github.com/huggingface/transformers.git
```

3. Richten Sie eine Entwicklungsumgebung ein, indem Sie z.B. den folgenden Befehl ausführen:

```bash
python -m venv .env
source .env/bin/activate
pip install -e ".[dev]"
```

Abhängig von Ihrem Betriebssystem und da die Anzahl der optionalen Abhängigkeiten von Transformers wächst, kann es sein, dass Sie bei diesem Befehl einen
Fehler mit diesem Befehl erhalten. Wenn das der Fall ist, stellen Sie sicher, dass Sie TensorFlow installieren und dann ausführen:

```bash
pip install -e ".[quality]"
```

**Hinweis:** Sie müssen CUDA nicht installiert haben. Es reicht aus, das neue Modell auf der CPU laufen zu lassen.

4. Erstellen Sie eine Verzweigung mit einem beschreibenden Namen von Ihrer Hauptverzweigung

```bash
git checkout -b add_tf_brand_new_bert
```

5. Abrufen und zurücksetzen auf die aktuelle Hauptversion

```bash
git fetch upstream
git rebase upstream/main
```

6. Fügen Sie eine leere `.py` Datei in `transformers/src/models/brandnewbert/` mit dem Namen `modeling_tf_brandnewbert.py` hinzu. Dies wird
Ihre TensorFlow-Modelldatei sein.

7. Übertragen Sie die Änderungen auf Ihr Konto mit:

```bash
git add .
git commit -m "initial commit"
git push -u origin add_tf_brand_new_bert
```

8. Wenn Sie zufrieden sind, gehen Sie auf die Webseite Ihrer Abspaltung auf GitHub. Klicken Sie auf "Pull request". Stellen Sie sicher, dass Sie das
   GitHub-Handle einiger Mitglieder des Hugging Face-Teams als Reviewer hinzuzufügen, damit das Hugging Face-Team über zukünftige Änderungen informiert wird.
   zukünftige Änderungen benachrichtigt wird.

9. Ändern Sie den PR in einen Entwurf, indem Sie auf der rechten Seite der GitHub-Pull-Request-Webseite auf "In Entwurf umwandeln" klicken.


Jetzt haben Sie eine Entwicklungsumgebung eingerichtet, um *BrandNewBert* nach TensorFlow in 🤗 Transformers zu portieren.


**3. (Optional) Verstehen Sie die theoretischen Aspekte und die bestehende Implementierung**

Sie sollten sich etwas Zeit nehmen, um die Arbeit von *BrandNewBert* zu lesen, falls eine solche Beschreibung existiert. Möglicherweise gibt es große
Abschnitte des Papiers, die schwer zu verstehen sind. Wenn das der Fall ist, ist das in Ordnung - machen Sie sich keine Sorgen! Das Ziel ist
ist es nicht, ein tiefes theoretisches Verständnis des Papiers zu erlangen, sondern die notwendigen Informationen zu extrahieren, um
das Modell mit Hilfe von TensorFlow effektiv in 🤗 Transformers neu zu implementieren. Das heißt, Sie müssen nicht zu viel Zeit auf die
viel Zeit auf die theoretischen Aspekte verwenden, sondern sich lieber auf die praktischen Aspekte konzentrieren, nämlich auf die bestehende Modelldokumentation
Seite (z.B. [model docs for BERT](model_doc/bert)).

Nachdem Sie die Grundlagen der Modelle, die Sie implementieren wollen, verstanden haben, ist es wichtig, die bestehende
Implementierung zu verstehen. Dies ist eine gute Gelegenheit, sich zu vergewissern, dass eine funktionierende Implementierung mit Ihren Erwartungen an das
Modell entspricht, und um technische Herausforderungen auf der TensorFlow-Seite vorauszusehen.

Es ist ganz natürlich, dass Sie sich von der Menge an Informationen, die Sie gerade aufgesogen haben, überwältigt fühlen. Es ist
Es ist definitiv nicht erforderlich, dass Sie in dieser Phase alle Facetten des Modells verstehen. Dennoch empfehlen wir Ihnen dringend
ermutigen wir Sie, alle dringenden Fragen in unserem [Forum](https://discuss.huggingface.co/) zu klären.


### 4. Implementierung des Modells

Jetzt ist es an der Zeit, endlich mit dem Programmieren zu beginnen. Als Ausgangspunkt empfehlen wir die PyTorch-Datei selbst: Kopieren Sie den Inhalt von
`modeling_brand_new_bert.py` in `src/transformers_openvla_oft/models/brand_new_bert/` nach
`modeling_tf_brand_new_bert.py`. Das Ziel dieses Abschnitts ist es, die Datei zu ändern und die Importstruktur von
🤗 Transformers zu aktualisieren, so dass Sie `TFBrandNewBert` und
`TFBrandNewBert.from_pretrained(model_repo, from_pt=True)` erfolgreich ein funktionierendes TensorFlow *BrandNewBert* Modell lädt.

Leider gibt es kein Rezept, um ein PyTorch-Modell in TensorFlow zu konvertieren. Sie können jedoch unsere Auswahl an
Tipps befolgen, um den Prozess so reibungslos wie möglich zu gestalten:
- Stellen Sie `TF` dem Namen aller Klassen voran (z.B. wird `BrandNewBert` zu `TFBrandNewBert`).
- Die meisten PyTorch-Operationen haben einen direkten TensorFlow-Ersatz. Zum Beispiel entspricht `torch.nn.Linear` der Klasse
  `tf.keras.layers.Dense`, `torch.nn.Dropout` entspricht `tf.keras.layers.Dropout`, usw. Wenn Sie sich nicht sicher sind
  über eine bestimmte Operation nicht sicher sind, können Sie die [TensorFlow-Dokumentation](https://www.tensorflow.org/api_docs/python/tf)
  oder die [PyTorch-Dokumentation](https://pytorch.org/docs/stable/).
- Suchen Sie nach Mustern in der Codebasis von 🤗 Transformers. Wenn Sie auf eine bestimmte Operation stoßen, für die es keinen direkten Ersatz gibt
   Ersatz hat, stehen die Chancen gut, dass jemand anderes bereits das gleiche Problem hatte.
- Behalten Sie standardmäßig die gleichen Variablennamen und die gleiche Struktur wie in PyTorch bei. Dies erleichtert die Fehlersuche, die Verfolgung von
   Probleme zu verfolgen und spätere Korrekturen vorzunehmen.
- Einige Ebenen haben in jedem Framework unterschiedliche Standardwerte. Ein bemerkenswertes Beispiel ist die Schicht für die Batch-Normalisierung
   epsilon (`1e-5` in [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d)
   und `1e-3` in [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)).
   Prüfen Sie die Dokumentation genau!
- Die Variablen `nn.Parameter` von PyTorch müssen in der Regel innerhalb von TF Layer's `build()` initialisiert werden. Siehe das folgende
   Beispiel: [PyTorch](https://github.com/huggingface/transformers/blob/655f72a6896c0533b1bdee519ed65a059c2425ac/src/transformers_openvla_oft/models/vit_mae/modeling_vit_mae.py#L212) /
   [TensorFlow](https://github.com/huggingface/transformers/blob/655f72a6896c0533b1bdee519ed65a059c2425ac/src/transformers_openvla_oft/models/vit_mae/modeling_tf_vit_mae.py#L220)
- Wenn das PyTorch-Modell ein `#copied from ...` am Anfang einer Funktion hat, stehen die Chancen gut, dass Ihr TensorFlow-Modell diese Funktion auch
   diese Funktion von der Architektur ausleihen kann, von der sie kopiert wurde, vorausgesetzt, es hat eine TensorFlow-Architektur.
- Die korrekte Zuweisung des Attributs `name` in TensorFlow-Funktionen ist entscheidend, um das `from_pt=True` Gewicht zu erreichen
   Cross-Loading. Name" ist fast immer der Name der entsprechenden Variablen im PyTorch-Code. Wenn `name` nicht
   nicht richtig gesetzt ist, sehen Sie dies in der Fehlermeldung beim Laden der Modellgewichte.
- Die Logik der Basismodellklasse, `BrandNewBertModel`, befindet sich in `TFBrandNewBertMainLayer`, einer Keras
   Schicht-Unterklasse ([Beispiel](https://github.com/huggingface/transformers/blob/4fd32a1f499e45f009c2c0dea4d81c321cba7e02/src/transformers_openvla_oft/models/bert/modeling_tf_bert.py#L719)).
   TFBrandNewBertModel" ist lediglich ein Wrapper für diese Schicht.
- Keras-Modelle müssen erstellt werden, um die vorher trainierten Gewichte zu laden. Aus diesem Grund muss `TFBrandNewBertPreTrainedModel`
   ein Beispiel für die Eingaben in das Modell enthalten, die `dummy_inputs`
   ([Beispiel](https://github.com/huggingface/transformers/blob/4fd32a1f499e45f009c2c0dea4d81c321cba7e02/src/transformers_openvla_oft/models/bert/modeling_tf_bert.py#L916)).
- Wenn Sie nicht weiterkommen, fragen Sie nach Hilfe - wir sind für Sie da! 🤗

Neben der Modelldatei selbst müssen Sie auch die Verweise auf die Modellklassen und die zugehörigen
Dokumentationsseiten hinzufügen. Sie können diesen Teil ganz nach den Mustern in anderen PRs erledigen
([Beispiel](https://github.com/huggingface/transformers/pull/18020/files)). Hier ist eine Liste der erforderlichen manuellen
Änderungen:
- Fügen Sie alle öffentlichen Klassen von *BrandNewBert* in `src/transformers_openvla_oft/__init__.py` ein.
- Fügen Sie *BrandNewBert* Klassen zu den entsprechenden Auto Klassen in `src/transformers_openvla_oft/models/auto/modeling_tf_auto.py` hinzu.
- Fügen Sie die *BrandNewBert* zugehörigen Klassen für träges Laden in `src/transformers_openvla_oft/utils/dummy_tf_objects.py` hinzu.
- Aktualisieren Sie die Importstrukturen für die öffentlichen Klassen in `src/transformers_openvla_oft/models/brand_new_bert/__init__.py`.
- Fügen Sie die Dokumentationszeiger auf die öffentlichen Methoden von *BrandNewBert* in `docs/source/de/model_doc/brand_new_bert.md` hinzu.
- Fügen Sie sich selbst zur Liste der Mitwirkenden an *BrandNewBert* in `docs/source/de/model_doc/brand_new_bert.md` hinzu.
- Fügen Sie schließlich ein grünes Häkchen ✅ in der TensorFlow-Spalte von *BrandNewBert* in `docs/source/de/index.md` hinzu.

Wenn Sie mit Ihrer Implementierung zufrieden sind, führen Sie die folgende Checkliste aus, um zu bestätigen, dass Ihre Modellarchitektur
fertig ist:
1. Alle Schichten, die sich zur Trainingszeit anders verhalten (z.B. Dropout), werden mit einem `Training` Argument aufgerufen, das
von den Top-Level-Klassen weitergegeben wird
2. Sie haben `#copied from ...` verwendet, wann immer es möglich war.
3. Die Funktion `TFBrandNewBertMainLayer` und alle Klassen, die sie verwenden, haben ihre Funktion `call` mit `@unpack_inputs` dekoriert
4. `TFBrandNewBertMainLayer` ist mit `@keras_serializable` dekoriert
5. Ein TensorFlow-Modell kann aus PyTorch-Gewichten mit `TFBrandNewBert.from_pretrained(model_repo, from_pt=True)` geladen werden.
6. Sie können das TensorFlow Modell mit dem erwarteten Eingabeformat aufrufen


### 5. Modell-Tests hinzufügen

Hurra, Sie haben ein TensorFlow-Modell implementiert! Jetzt ist es an der Zeit, Tests hinzuzufügen, um sicherzustellen, dass sich Ihr Modell wie erwartet verhält.
erwartet. Wie im vorigen Abschnitt schlagen wir vor, dass Sie zunächst die Datei `test_modeling_brand_new_bert.py` in
`tests/models/brand_new_bert/` in die Datei `test_modeling_tf_brand_new_bert.py` zu kopieren und dann die notwendigen
TensorFlow-Ersetzungen vornehmen. Für den Moment sollten Sie in allen Aufrufen von `.from_pretrained()` das Flag `from_pt=True` verwenden, um die
die vorhandenen PyTorch-Gewichte zu laden.

Wenn Sie damit fertig sind, kommt der Moment der Wahrheit: Führen Sie die Tests durch! 😬

```bash
NVIDIA_TF32_OVERRIDE=0 RUN_SLOW=1 RUN_PT_TF_CROSS_TESTS=1 \
py.test -vv tests/models/brand_new_bert/test_modeling_tf_brand_new_bert.py
```

Das wahrscheinlichste Ergebnis ist, dass Sie eine Reihe von Fehlern sehen werden. Machen Sie sich keine Sorgen, das ist zu erwarten! Das Debuggen von ML-Modellen ist
notorisch schwierig, und der Schlüssel zum Erfolg ist Geduld (und `breakpoint()`). Nach unserer Erfahrung sind die schwierigsten
Probleme aus subtilen Unstimmigkeiten zwischen ML-Frameworks, zu denen wir am Ende dieses Leitfadens ein paar Hinweise geben.
In anderen Fällen kann es sein, dass ein allgemeiner Test nicht direkt auf Ihr Modell anwendbar ist; in diesem Fall empfehlen wir eine Überschreibung
auf der Ebene der Modelltestklasse. Zögern Sie nicht, in Ihrem Entwurf einer Pull-Anfrage um Hilfe zu bitten, wenn
Sie nicht weiterkommen.

Wenn alle Tests erfolgreich waren, können Sie Ihr Modell in die 🤗 Transformers-Bibliothek aufnehmen! 🎉

### 6.-7. Stellen Sie sicher, dass jeder Ihr Modell verwenden kann

**6. Reichen Sie den Pull Request ein**

Sobald Sie mit der Implementierung und den Tests fertig sind, ist es an der Zeit, eine Pull-Anfrage einzureichen. Bevor Sie Ihren Code einreichen,
führen Sie unser Dienstprogramm zur Codeformatierung, `make fixup` 🪄, aus. Damit werden automatisch alle Formatierungsfehler behoben, die dazu führen würden, dass
unsere automatischen Prüfungen fehlschlagen würden.

Nun ist es an der Zeit, Ihren Entwurf einer Pull-Anfrage in eine echte Pull-Anfrage umzuwandeln. Klicken Sie dazu auf die Schaltfläche "Bereit für
Review" und fügen Sie Joao (`@gante`) und Matt (`@Rocketknight1`) als Reviewer hinzu. Eine Modell-Pull-Anfrage benötigt
mindestens 3 Reviewer, aber sie werden sich darum kümmern, geeignete zusätzliche Reviewer für Ihr Modell zu finden.

Nachdem alle Gutachter mit dem Stand Ihres PR zufrieden sind, entfernen Sie als letzten Aktionspunkt das Flag `from_pt=True` in
.from_pretrained()-Aufrufen zu entfernen. Da es keine TensorFlow-Gewichte gibt, müssen Sie sie hinzufügen! Lesen Sie den Abschnitt
unten, um zu erfahren, wie Sie dies tun können.

Wenn schließlich die TensorFlow-Gewichte zusammengeführt werden, Sie mindestens 3 Genehmigungen von Prüfern haben und alle CI-Checks grün sind
grün sind, überprüfen Sie die Tests ein letztes Mal lokal

```bash
NVIDIA_TF32_OVERRIDE=0 RUN_SLOW=1 RUN_PT_TF_CROSS_TESTS=1 \
py.test -vv tests/models/brand_new_bert/test_modeling_tf_brand_new_bert.py
```

und wir werden Ihren PR zusammenführen! Herzlichen Glückwunsch zu dem Meilenstein 🎉.

**7. (Optional) Erstellen Sie Demos und teilen Sie sie mit der Welt**

Eine der schwierigsten Aufgaben bei Open-Source ist die Entdeckung. Wie können die anderen Benutzer von der Existenz Ihres
fabelhaften TensorFlow-Beitrags erfahren? Mit der richtigen Kommunikation, natürlich! 📣

Es gibt vor allem zwei Möglichkeiten, Ihr Modell mit der Community zu teilen:
- Erstellen Sie Demos. Dazu gehören Gradio-Demos, Notebooks und andere unterhaltsame Möglichkeiten, Ihr Modell vorzuführen. Wir raten Ihnen
   ermutigen Sie, ein Notizbuch zu unseren [community-driven demos](https://huggingface.co/docs/transformers/community) hinzuzufügen.
- Teilen Sie Geschichten in sozialen Medien wie Twitter und LinkedIn. Sie sollten stolz auf Ihre Arbeit sein und sie mit der
   Ihre Leistung mit der Community teilen - Ihr Modell kann nun von Tausenden von Ingenieuren und Forschern auf der ganzen Welt genutzt werden
   der Welt genutzt werden 🌍! Wir werden Ihre Beiträge gerne retweeten und Ihnen helfen, Ihre Arbeit mit der Community zu teilen.


## Hinzufügen von TensorFlow-Gewichten zum 🤗 Hub

Unter der Annahme, dass die TensorFlow-Modellarchitektur in 🤗 Transformers verfügbar ist, ist die Umwandlung von PyTorch-Gewichten in
TensorFlow-Gewichte ist ein Kinderspiel!

Hier sehen Sie, wie es geht:
1. Stellen Sie sicher, dass Sie in Ihrem Terminal bei Ihrem Hugging Face Konto angemeldet sind. Sie können sich mit dem folgenden Befehl anmelden
   `huggingface-cli login` (Ihre Zugangstoken finden Sie [hier](https://huggingface.co/settings/tokens))
2. Führen Sie `transformers-cli pt-to-tf --model-name foo/bar` aus, wobei `foo/bar` der Name des Modell-Repositorys ist
   ist, das die PyTorch-Gewichte enthält, die Sie konvertieren möchten.
3. Markieren Sie `@joaogante` und `@Rocketknight1` in dem 🤗 Hub PR, den der obige Befehl gerade erstellt hat

Das war's! 🎉


## Fehlersuche in verschiedenen ML-Frameworks 🐛

Irgendwann, wenn Sie eine neue Architektur hinzufügen oder TensorFlow-Gewichte für eine bestehende Architektur erstellen, werden Sie
stoßen Sie vielleicht auf Fehler, die sich über Unstimmigkeiten zwischen PyTorch und TensorFlow beschweren. Sie könnten sich sogar dazu entschließen, den
Modellarchitektur-Code für die beiden Frameworks zu öffnen, und stellen fest, dass sie identisch aussehen. Was ist denn da los? 🤔

Lassen Sie uns zunächst darüber sprechen, warum es wichtig ist, diese Diskrepanzen zu verstehen. Viele Community-Mitglieder werden 🤗
Transformers-Modelle und vertrauen darauf, dass sich unsere Modelle wie erwartet verhalten. Wenn es eine große Diskrepanz gibt
zwischen den beiden Frameworks auftritt, bedeutet dies, dass das Modell nicht der Referenzimplementierung für mindestens eines der Frameworks folgt.
der Frameworks folgt. Dies kann zu stillen Fehlern führen, bei denen das Modell zwar läuft, aber eine schlechte Leistung aufweist. Dies ist
wohl schlimmer als ein Modell, das überhaupt nicht läuft! Aus diesem Grund streben wir an, dass die Abweichung zwischen den Frameworks kleiner als
1e-5" in allen Phasen des Modells.

Wie bei anderen numerischen Problemen auch, steckt der Teufel im Detail. Und wie bei jedem detailorientierten Handwerk ist die geheime
Zutat hier Geduld. Hier ist unser Vorschlag für den Arbeitsablauf, wenn Sie auf diese Art von Problemen stoßen:
1. Lokalisieren Sie die Quelle der Abweichungen. Das Modell, das Sie konvertieren, hat wahrscheinlich bis zu einem gewissen Punkt nahezu identische innere Variablen.
   bestimmten Punkt. Platzieren Sie `Breakpoint()`-Anweisungen in den Architekturen der beiden Frameworks und vergleichen Sie die Werte der
   numerischen Variablen von oben nach unten, bis Sie die Quelle der Probleme gefunden haben.
2. Nachdem Sie nun die Ursache des Problems gefunden haben, setzen Sie sich mit dem 🤗 Transformers-Team in Verbindung. Es ist möglich
   dass wir ein ähnliches Problem schon einmal gesehen haben und umgehend eine Lösung anbieten können. Als Ausweichmöglichkeit können Sie beliebte Seiten
   wie StackOverflow und GitHub-Probleme.
3. Wenn keine Lösung in Sicht ist, bedeutet das, dass Sie tiefer gehen müssen. Die gute Nachricht ist, dass Sie das Problem gefunden haben.
   Problem ausfindig gemacht haben, so dass Sie sich auf die problematische Anweisung konzentrieren und den Rest des Modells ausblenden können! Die schlechte Nachricht ist
   dass Sie sich in die Quellimplementierung der besagten Anweisung einarbeiten müssen. In manchen Fällen finden Sie vielleicht ein
   Problem mit einer Referenzimplementierung - verzichten Sie nicht darauf, ein Problem im Upstream-Repository zu öffnen.

In einigen Fällen können wir nach Rücksprache mit dem 🤗 Transformers-Team zu dem Schluss kommen, dass die Behebung der Abweichung nicht machbar ist.
Wenn die Abweichung in den Ausgabeschichten des Modells sehr klein ist (aber möglicherweise groß in den versteckten Zuständen), können wir
könnten wir beschließen, sie zu ignorieren und das Modell zu verteilen. Die oben erwähnte CLI `pt-to-tf` hat ein `--max-error`
Flag, um die Fehlermeldung bei der Gewichtskonvertierung zu überschreiben.
