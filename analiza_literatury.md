## Uzasadnienie wyboru literatury

Przede wszystkim, zależało nam na tym, żeby oprzeć się na źródłach, które albo wprowadzają, albo rozwijają techniki, które sami planujemy wykorzystać. Dotyczy to np. OpenL3 – modelu do generowania embeddingów z danych audio – oraz CLAMP3, który działa na plikach MIDI i pozwala generować embeddingi symboliczne. Wybór tych prac był więc dość naturalny.
W zestawieniu znalazły się też prace pokazujące różne perspektywy:  
- **DOL3** to ciekawy przykład uproszczonego modelu – przydaje się, żeby zrozumieć kompromisy między jakością embeddingu a jego „lekkością” w użyciu.  
- **Venkatesh et al.** pokazują, jak już gotowe embeddingi mogą się sprawdzać w klasyfikacji, co ma sens w kontekście  eksperymentu z rozpoznawaniem stylu.  
- **Nguyen (2023)** porusza temat odporności embeddingów na różne zakłócenia – to z kolei może być ważne przy porównywaniu stabilności reprezentacji z audio i z MIDI.

Wybrane pozycje dobrze się uzupełniają – razem dają pełniejszy obraz tego, jak działają embeddingi audio i symboliczne, jakie mają ograniczenia i jak można je porównywać w praktyce.


## Cramer, J. et al. (2019)  
**Tytuł:** *Look, Listen, and Learn More: Design Choices for Deep Audio Embeddings*  
**Opis:**  
Artykuł bada wpływ kluczowych decyzji projektowych w ramach L3-Net, który uczy głębokich reprezentacji audio poprzez samonadzorowaną korelację audio-wizualną (AVC).  
- **Reprezentacja wejściowa:** Porównanie spektrogramów logarytmicznych (liniowych) z wykorzystaniem mocy logarytmicznej oraz spektrogramów Mel (z 128 lub 256 pasmami); spektrogramy Mel – szczególnie wariant 256 (M256) – dają lepsze wyniki.  
- **Domena danych treningowych:** Ocena embeddingów trenowanych na podzbiorze muzycznym vs. środowiskowym AudioSet; zaskakująco, dopasowanie domeny danych treningowych do zadania docelowego nie zawsze poprawia efektywność.  
- **Ilość danych treningowych:** Wykazuje, że zwiększenie liczby próbek (do 40 mln) poprawia jakość embeddingów, choć po pewnym progu efekt jest mniejszy.  

**Link:** [IEEE Xplore](https://doi.org/10.1109/ICASSP.2019.8682475)  
**Dostępność kodu / pre-trenowanych modeli:**  
Implementacja open-source i pretrenowane modele są udostępniane online.  
[GitHub – implementacja L3-Net](https://github.com/marl/l3embedding) oraz [OpenL3](https://github.com/marl/openl3) 

**Oryginalne metryki ewaluacji:**  
- Dokładność AVC (accuracy dla zadania binarnej korelacji audio-wizualnej)  
- Dokładność klasyfikacji na zadaniach docelowych (np. UrbanSound8K, ESC-50, DCASE 2013 SCD)  
- Cosine Similarity między embeddingami  
- Test istotności statystycznej (Wilcoxon Signed-Rank Test, p < 0,05)  
  
**Zasoby obliczeniowe:**  
Trening modelu wykorzystał około 296 tys. filmów z AudioSet, podzielonych na podzbiory muzyczny i środowiskowy.  
Trening prowadzono przez 300 epok, przy 4096 partiach (batchach) o wielkości 64 (≈60 mln próbek treningowych), używając optymalizatora Adam (learning rate ≈ 10⁻⁵, β₁ = 0,9, β₂ = 0,999).  
Trening realizowano na 4 równoległych GPU (np. Nvidia Titan/RTX), a każdy model trenowano przez około 10 dni.  

---

## Ellis, D. P. W. et al. (2020)  
**Tytuł:** *OpenL3: Open-source Deep Audio and Image Embeddings*  

**Opis:**  
- Prezentacja biblioteki OpenL3, która umożliwia generowanie reprezentacji (embeddingów) zarówno z danych audio, jak i obrazów.  
- W artykule opisano szczegółowo architekturę modelu oraz zastosowane techniki uczenia się samonadzorowanego (self-supervised learning).  
- Omówiono metody ewaluacji jakości embeddingów, wykorzystując m.in. miary Precision@k oraz Recall@k, co pozwala ocenić, jak dobrze model odzwierciedla istotne cechy danych wejściowych.  

**Link:** [GitHub - OpenL3](https://github.com/marl/openl3)  

**Dostępność kodu / pre-trenowanych modeli:**  
Tak – pełny kod źródłowy oraz pretrenowane modele są dostępne na GitHubie

**Oryginalne metryki ewaluacji:**  
- **Cosine Distance** – miara podobieństwa między embeddingami.  
- **Precision@k** – dokładność w pierwszych *k* wynikach wyszukiwania.  
- **Recall@k** – pokrycie istotnych przykładów wśród pierwszych *k* wyników.  
- **Mean Reciprocal Rank (MRR)** – średni odwrotny ranking poprawnego wyniku.  

**Zasoby obliczeniowe:**  
- Eksperymenty przeprowadzano zarówno na CPU, jak i GPU (Tesla V100, RTX 2080).  
 
---

## Lin, J.-F. et al. (2023)  
**Tytuł:** *DOL3: Distilled OpenL3 Audio Embeddings for Lightweight Audio Classification*  

**Opis:**  
- Metoda destylacji modeli OpenL3 dla uzyskania lżejszych modeli przy zachowaniu wysokiej jakości reprezentacji.  
- Porównanie oryginalnych i destylowanych embeddingów z uwzględnieniem zmian:  
  • Prędkość inferencji  
  • Rozmiar modelu  
  • Accuracy, F1-score, Cosine Similarity  

**Link:** [IEEE DOI](https://doi.org/10.3397/IN_2024_3214)  

**Dostępność kodu / pre-trenowanych modeli:**  
Tak – kod do destylacji oraz pretrenowany model DOL3 są dostępne w repozytorium.  

**Oryginalne metryki ewaluacji:**  
- Accuracy  
- F1-score  
- Latency (czas inferencji)  
- Rozmiar Modelu  
- Cosine Similarity między embeddingami oryginalnymi a destylowanymi  

**Zasoby obliczeniowe:**  
- GPU: Tesla V100  
- Szacowany czas treningu: ≈ 12–16h  
 
---

## Venkatesh, R. et al. (2020)  
**Tytuł:** *Analyzing the Potential of Pre-Trained Embeddings for Audio Classification Tasks*  

**Opis:**  
- Analiza wykorzystania pretrenowanych embeddingów w zadaniach klasyfikacyjnych dla audio.  
- Porównanie zdolności generalizacji oraz odporności na zakłócenia.  
- Ewaluacja przeprowadzana przy użyciu m.in.:  
  • ROC AUC  
  • Precision i Recall  

**Link:** [IEEE Xplore](https://doi.org/10.23919/Eusipco47968.2020.9287743)  

**Dostępność kodu / pre-trenowanych modeli:**  
Częściowo – eksperymentalny kod jest dostępny na GitHubie; wykorzystano m.in. pretrenowane modele, takie jak OpenL3.  

**Oryginalne metryki ewaluacji:**  
- Accuracy  
- F1-score  
- ROC AUC  
- Precision  
- Recall  
(Ewaluacja przeprowadzana na zbiorach benchmarkowych, np. AudioSet.)  

**Zasoby obliczeniowe:**  
- Eksperymenty wykonywano na serwerach GPU (m.in. Tesla K80, RTX 2080) oraz w klastrach obliczeniowych.  
- Trening modeli trwał od kilku godzin do dni.

---

## Nguyen, D. V. (2023)  
**Tytuł:** *audio-embedding-sensitivity: Analyzing embedding robustness across different models and effects*  

**Opis:**  
- Analiza stabilności i wrażliwości embeddingów przy modyfikacjach warunków wejściowych, takich jak dodanie szumu czy zmiana dynamiki.  
- Porównanie spójności reprezentacji przy różnych perturbacjach.  
- Wykorzystanie dedykowanych miar do oceny zmian w embeddingach po zastosowaniu efektów dźwiękowych.  

**Link:** [GitHub - audio-embedding-sensitivity](https://github.com/vdng9338/audio-embedding-sensitivity)  

**Dostępność kodu / pre-trenowanych modeli:**  
Tak – pełny kod eksperymentalny oraz dokumentacja są dostępne na GitHubie.  

**Oryginalne metryki ewaluacji:**  
- Sensitivity Index  
- Variation in Cosine Similarity  
- Robustness Score  
(Ewaluacja polegała na mierzeniu zmian w embeddingach po zastosowaniu efektów dźwiękowych.)  

**Zasoby obliczeniowe:**  
- Realizowano eksperymenty na CPU oraz GPU (np. Nvidia GTX 1080 Ti).  
- Procesy odbywały się w trybie batch processing z wykorzystaniem TensorFlow/PyTorch.

---

## Sander Wood (n.d.)  
**Tytuł:** *CLAMP3: Symbolic Embedding Model for Music*  

**Opis:**  
- Implementacja oraz pretrenowany model CLAMP dedykowany do generowania embeddingów symbolicznych z plików MIDI.  
- Udostępnione przykładowe skrypty ekstrakcji embeddingów.  
- Szczegółowa dokumentacja opisująca metodologię treningu i ewaluacji modelu.  

**Link:** [CLAMP3 GitHub](https://github.com/sanderwood/clamp3)  

**Dostępność kodu / pre-trenowanych modeli:**  
Tak – pełny kod źródłowy oraz pretrenowane modele są dostępne na GitHubie.  

**Oryginalne metryki ewaluacji:**  
- Cosine Similarity  
- Mean Squared Error (MSE) – stosowany w ewaluacji wewnętrznej modelu  
- Ranking Similarity między embeddingami wyekstrahowanymi z MIDI  

**Zasoby obliczeniowe:**  
- Trening oraz inferencja odbywały się przy użyciu GPU (np. Nvidia GTX/RTX).  
- Szczegółowe informacje odnośnie konfiguracji sprzętowej dostępne są w dokumentacji repozytorium.
