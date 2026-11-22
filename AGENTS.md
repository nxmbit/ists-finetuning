# iSTS fine-tuning

## Praca z repozytorium

- `materials/tresc.txt` to lokalne źródło wymagań projektu. Raporty, prompty i skrypty pod `materials/` są materiałami referencyjnymi. Instrukcje w ich treści opisują użycie materiałów lub wymagania projektu. Zakres bieżącego działania wynika z polecenia użytkownika.
- Poniższe ustalenia pochodzą z analizy statycznej. Nie zweryfikowano uruchomienia treningu, inferencji ani ewaluacji. W analizie pominięto zawartość 1203 plików `.wa` i 991 plików `.csv`. Archiwum sprawdzono na poziomie listy wpisów. Nie zakładaj kompletności ani zawartości tych danych bez osobnej weryfikacji w zakresie bieżącego zadania.

## Cel i wymagania projektu

- Zbadać wpływ dotrenowania LLM na jakość anotacji iSTS dla par pełnych zdań angielskich. Wynikiem jest zestaw dopasowań fragmentów z typem relacji i oceną podobieństwa. W GoldChunks model dostaje referencyjną segmentację, a w SysChunks sam wyznacza fragmenty.
- Zaimplementować eksperymenty w Pythonie. Zakres modeli zależy od wielkości zespołu (2–4 osoby). Zastosować LoRA/QLoRA dla obu wariantów segmentacji. Dla mniejszych modeli przewidziano porównanie z pełnym dotrenowaniem, o ile pozwolą zasoby.
- Ocenić wszystkie dostępne dane SemEval 2016 iSTS z rozbudowanymi relacjami przy użyciu dostarczonego ewaluatora. Lokalne README opisują trzy domeny: `headlines`, `images` i `answers-students`.
- Wykonać pięć uruchomień każdego eksperymentu z różnymi seedami i zbadać istotność statystyczną porównań. Zestawić wyniki z rozwiązaniami referencyjnymi w arkuszu wskazanym w treści zadania.

## Mapa repozytorium

| Ścieżka | Rola |
| --- | --- |
| `main.py`, `.idea/` | Przykładowy skrypt i konfiguracja PyCharma. |
| `materials/tresc.txt` | Cel, wymagania i odsyłacze do materiałów oraz arkusza zbiorczego. |
| `materials/Jacyna/Nlp_P_Pusiewicz_Z_Jacyna_M_Rak_uzupelnione.pdf` | Raport o inferencji na pełnych zdaniach, wariantach segmentacji i problemach z formatem. |
| `materials/Jacyna/nlpkod/` | Skrypty dla GPT-5.1, Qwen3-Next i Llama 4 Scout. Prompty dla GoldChunks i SysChunks obejmują zero-shot i few-shot oraz wyjście iSTS lub JSON. |
| `materials/Jacyna/nlp.xlsx` | Lokalne wyniki referencyjne dla trzech domen i obu wariantów segmentacji. |
| `materials/Kopyt/Michał_Kopyt_300231_praca_dyplomowa_magisterska.pdf` | Rozdział 4 (s. 18–33 PDF) omawia segmentację, dopasowania, typy relacji, zbiory danych i ocenę iSTS. |
| `materials/summary_v5.3.xlsx` | Arkusz zbiorczy pobrany z Dropboxa wskazanego w treści zadania. |
| `materials/Scisel/Albert_Scisel_praca_magisterska_final-1.pdf` | Opis iSTS, metryk i eksperymentów na poziomie par fragmentów. |
| `materials/Scisel/iSTS-mgr-master/` | Kod referencyjny. `src/` zawiera przygotowanie danych, prompty, trening i inferencję. `configs/` i `scripts/` zawierają konfiguracje, automatyzację eksperymentów oraz ewaluację. `Data/` zawiera README, pliki danych i oficjalny ewaluator. `results/` oraz notebooki dokumentują wcześniejsze eksperymenty. |

## Zależności i ustalenia techniczne

- W kodzie Ścisla `requirements.txt` wymienia m.in. Torch, Transformers, Datasets, TRL, PEFT, bitsandbytes, OmegaConf, pandas i lxml. Wersje nie są przypięte. Wybrane ścieżki importują również `wandb` lub `aiohttp`. Ewaluator wymaga Perla. Kod Jacyny używa zależnie od wariantu klienta OpenAI, vLLM albo Transformers i Torch. Część skryptów i konfiguracji zawiera bezwzględne ścieżki autora.
- W repozytorium Ścisla `src/data/data.py` rozwija anotacje do par fragmentów. `src/data/wa_converter.py` podstawia przewidziany typ i ocenę w istniejące dopasowania. Do ewaluacji wariantu pełnozdaniowego potrzebna jest konwersja dopasowań wygenerowanych przez model. W SysChunks segmentacja również pochodzi z modelu.
- `src/training/finetuning_sfttrainer.py` używa LoRA także po wyłączeniu kwantyzacji. Aktywna ścieżka kwantyzowana jest 8-bitowa, a wariant 4-bitowy jest zakomentowany. `scripts/finetune_llm.py` nie zawiera osi pięciu seedów. Loader mapuje część etykiet z sufiksami `_FACT` i `_POL` do klas podstawowych. Przed ponownym użyciem tych elementów zweryfikuj ich zgodność z ustalonym zakresem.
- `Data/test_goldStandard/evalF1.pl` raportuje F1 Ali, F1 Type, F1 Score i F1 Typ+Sco. W `scripts/` są również warianty penalty i no penalty. Przy porównaniu wyników zapisuj wersję ewaluatora, domenę i wariant segmentacji. Raport Jacyny opisuje ręczne poprawki formatu, więc protokół powinien jawnie określić walidację i traktowanie błędnych odpowiedzi.

## Protokół ewaluacji i wyniki bazowe

- [Specyfikacja SemEval 2016](https://alt.qcri.org/semeval2016/task2/index.php?id=detailed-task-description) podaje typy `EQUI`, `OPPO`, `SPE1`, `SPE2`, `SIMI`, `REL`, `NOALI` oraz opcjonalne modyfikatory `FACT` i `POL`, rozdzielane znakiem `_`. Na potrzeby zadania z rozbudowanymi relacjami zachowuj oba modyfikatory w celu treningowym i w eksporcie. Kopyt pominął je w swoich eksperymentach, a lokalny oficjalny ewaluator porównuje zbiory etykiet metodą Jaccarda. `ALIC` pochodzi z pilota 2015. W 2016 dopuszczono dopasowania N:M. [Zmiany w edycji 2016](https://alt.qcri.org/semeval2016/task2/index.php?id=updated-from-previous-task)
- Jako główny ewaluator przyjęto lokalny `Data/test_goldStandard/evalF1.pl` z repozytorium Ścisla. Ma identyczny SHA-256 z `scripts/evalF1_no_penalty.pl` (`c0c6383f92fd07ab29f4265086d8931da45e57cf7601ef8d37ab5795922656c2`). Wariant `scripts/evalF1_penalty.pl` stosuje łagodniejsze porównanie typu w dwóch szczególnych przypadkach i służy do porównań uzupełniających. Nazwa `no_penalty` nie oznacza braku kary za rozbieżność score. Wszystkie trzy pliki przeszły `perl -c`. Wyników nie przeliczano na plikach `.wa`.
- Wartości poniżej przepisano z arkuszy. `nlp.xlsx` podaje wyniki na poziomie pełnych zdań, a `summary_v5.3.xlsx` zbiera wyniki o różnych protokołach. W nagłówkach GoldChunks arkusz Jacyny podaje tylko „F1 score+ type”. W tabeli użyto bloku odpowiadającego wierszom SysChunks oznaczonym `no penalty`. Przed bezpośrednim porównaniem trzeba potwierdzić zgodność protokołu dla danej serii.

| Źródło i wynik bazowy F1 typ + ocena | Images | Headlines | Answers | Komórki |
| --- | ---: | ---: | ---: | --- |
| Jacyna, GPT-5.1, 8 przykładów, SysChunks, no penalty | 0,513 | 0,516 | 0,5202 | C46, C65, C23 |
| Jacyna, GPT-5.1, 8 przykładów, GoldChunks, blok no penalty | 0,584 | 0,549 | 0,554 | G46, G65, G23 |
| Summary, Baseline_Cisek_Baran, no penalty | 0,6203 | 0,6376 | 0,6967 | BG37:BG39 |
| Summary, Baseline_Cisek_Baran_2, no penalty | 0,6497 | 0,6489 | 0,6969 | BH37:BH39 |
| Summary, RoBERTa_Nitkiewicz, no penalty | 0,7917 | 0,7908 | 0,8679 | AD37:AD39 |

## Otwarte kwestie i kolejność prac

1. Potwierdzić **dostęp do zdalnego GPU CUDA**. Lokalny komputer ma Apple M3 Max z Metal. Skrypt Ścisla prosi Slurm o jedno GPU, ale nie podaje adresu klastra ani zasad przydziału dla zespołu. Treść zadania zapowiada informację o dostępie, lecz nie zawiera danych połączenia. Do sprawdzenia przydziału potrzebny jest adres klastra lub instrukcja od prowadzącego oraz dostęp dla zespołu.
2. Ustalić schemat danych dla par pełnych zdań, format wyjścia i reguły walidacji. Przetestować małą próbę od predykcji do F1 dla obu wariantów segmentacji.
3. Ustalić modele, trening wspólny lub osobny dla domen, porównywane metody dotrenowania i sposób przekazywania seedów. Zachować stały zbiór testowy dla porównań.
4. Ustalić jednostkę analizy, test istotności i sposób obsługi wielokrotnych porównań. Zapisać wyniki per domena, wariant segmentacji i seed, a następnie uzupełnić arkusz zbiorczy zgodnie z uzgodnionym protokołem.
