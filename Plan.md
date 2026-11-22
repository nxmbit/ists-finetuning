% PLAN
% 1. Dostęp do GPU
% 2. Wybór modeli
% 3. Wybór metod dotrenowania
% 4. Wybór formatu danych treningowych i predykcji
% 5. Wybór sposobu obsługi niepoprawnych odpowiedzi
% 6. Wybór metody analizy statystycznej
% 7. Wybór protokołu porównania z wynikami referencyjnymi
% 8. Uruchomienie eksperymentów
% 9. Analiza wyników
% 10. Raportowanie wyników

% modele:
Gemma 4 31B

Potworzyc experyment Jacyny korzystajac z ich kodu.
 zmierzyc ich baseline bez dotrenowania
 Przygotowanie danych treningowych --- mamy, ale musimy (mamy outputy, )
 musimy zdefinowac inputy, oraz system prompt
 Dotrenowac dwa adaptery LoRa i QLoRa
 
 Zobaczy czy w baseline nie bedzie problemow z outputami. typu, ze model bedzie sam sie poprawial, albo format psuł
 Dataset ma juz skrypty ewaluacyjne na podstawie danych testowych. 
 Wspomniec o tym jak dzielony jest dataset ( w jakich proporcjach sa dane train, test, val)

 trenowanie lory :
 Robimy supervised tuning ( z nadzorem)
opisac jakie hiperpamatery uzyjemy, oraz jak bedziemy je wybierać 
(w przypadku gemmy wziac z dokumentacji)

Dodac jakis opis o datasetach przykald:
Liczba przykładów: 1599
Liczba atrybutów: 11
Zmienne wejściowe: cechy fizykochemiczne wina, m.in. kwasowość, zawartość cukru resztkowego, chlor-
ków, dwutlenku siarki, gęstość, pH, siarczany oraz zawartość alkoholu.
Klasy:
• 3 — 10 przykładów,
• 4 — 53 przykłady,
• 5 — 681 przykładów,
• 6 — 638 przykładów,
• 7 — 199 przykładów,
• 8 — 18 przykładów.
Zbiór nie posiada wydzielonego zestawu testowego.
Zbiór stanowi zadanie klasyfikacji wieloklasowej o niezbalansowanym rozkładzie klas. Większość przykładów
należy do klas środkowych, natomiast klasy skrajne są rzadkie, co zwiększa trudność klasyfikacji. Dzięki
temu zbiór dobrze uzupełnia dwa pozostałe zbiory binarne i pozwala sprawdzić, jak badane modyfikacje
lasu losowego zachowują się w bardziej złożonym przypadku klasyfikacji.


dodac wzmianke o miarach jakosci na podstawie excela i skryptow ewaluacyjnych

porowanianie baseline, oraz dotrenowanych modeli LoRa i QLoRa

5 seedow * 3 modele
dla mniejszych modeli, full fine tuning