# GitHub и Zenodo для этой статьи: использовать уже существующий Zenodo DOI

## Главное исправление

Для этой статьи уже создан Zenodo record:

```text
https://doi.org/10.5281/zenodo.18135575
```

Поэтому **не нужно создавать новый независимый Zenodo DOI**. Нужно сделать GitHub-репозиторий и связать его с уже существующим архивом. Если нужно обновить архив, используйте в Zenodo кнопку **New version** в существующем record, а не новый отдельный record.

## Что загрузить в GitHub

Загрузить нужно весь корень этого пакета:

```text
README.md
LICENSE
CITATION.cff
.zenodo.json
requirements.txt
run_demo.py
src/
results/
paper/
docs/
.github/
.gitignore
```

Не загружать временные папки:

```text
__pycache__/
.ipynb_checkpoints/
.venv/
.cache/
```

Они исключены в `.gitignore`.

## Рекомендуемое имя репозитория

```text
ecg-walsh-hadamard-sasida
```

Короткое описание GitHub:

```text
Reference implementation for interpretable ECG anomaly scoring in sequency-ordered Walsh-Hadamard coordinates.
```

## Как загрузить через браузер GitHub

1. Откройте GitHub.
2. Создайте новый публичный репозиторий `ecg-walsh-hadamard-sasida`.
3. Не создавайте автоматически README, LICENSE и `.gitignore`, потому что они уже есть в этом пакете.
4. Нажмите **Add file → Upload files**.
5. Перетащите все файлы и папки из этого пакета.
6. Commit message:

```text
Initial GitHub package for ECG Walsh-Hadamard SaSiDa manuscript
```

7. Нажмите **Commit changes**.

## Как загрузить через Git

```bash
git init
git add .
git commit -m "Initial GitHub package for ECG Walsh-Hadamard SaSiDa manuscript"
git branch -M main
git remote add origin https://github.com/sergoep/ecg-walsh-hadamard-sasida.git
git push -u origin main
```

## Проверка после загрузки

В репозитории должно быть видно:

```text
README.md
run_demo.py
src/ecg_walsh/core.py
src/ecg_walsh/synthetic.py
results/summary_metrics.csv
results/figures/fig_contribution_profile.png
CITATION.cff
LICENSE
.zenodo.json
```

Проверьте, что локально или в GitHub Codespaces работает:

```bash
pip install -r requirements.txt
python run_demo.py
```

Главные проверки должны быть порядка:

```text
max_invariance_error       ~ 10^-13
exact_decomposition_error  ~ 10^-14
orthogonality_error        ~ 10^-16
```

## Что делать с Zenodo, если DOI уже есть

Откройте существующий record:

```text
https://zenodo.org/records/18135575
```

Дальше есть два варианта.

### Вариант A: ничего не менять в Zenodo

Если в Zenodo уже лежит корректный архив, а GitHub нужен только как открытый репозиторий, то достаточно:

1. Загрузить этот пакет в GitHub.
2. В README GitHub оставить DOI:

```text
https://doi.org/10.5281/zenodo.18135575
```

3. В статье в Code Availability указать этот DOI.

### Вариант B: обновить Zenodo тем же проектом

Если нужно, чтобы Zenodo содержал именно этот новый GitHub-пакет:

1. Откройте существующий Zenodo record.
2. Нажмите **New version**.
3. Загрузите новый ZIP-пакет GitHub-репозитория.
4. В metadata оставьте тот же title и authors.
5. В version укажите, например:

```text
v1.0.1
```

6. В description добавьте:

```text
This version adds the GitHub-ready reproducibility package, updated README, CITATION.cff, .zenodo.json metadata, generated verification outputs, and the final SaSiDa manuscript source.
```

7. Нажмите **Publish**.

Важно: в этом случае Zenodo создаст новую версию внутри того же проекта. Для статьи лучше цитировать DOI версии, которая точно соответствует загруженному коду. Если хотите универсальную ссылку на все версии, оставьте concept DOI:

```text
https://doi.org/10.5281/zenodo.18135575
```

## Что написать в статье в Code Availability

```text
The reference implementation and reproducibility package are archived on Zenodo at https://doi.org/10.5281/zenodo.18135575 and are available from the associated GitHub repository. The archived release contains the code used for the fast verification run, including orthogonal-invariance and exact-decomposition checks, generated figures, run configuration, and summary metrics.
```

## Что не писать

Не писать:

```text
The synthetic experiment proves clinical performance.
```

Правильно писать:

```text
The synthetic run is used only as a computational verification of the proposed operator and its exact identities. It is not interpreted as clinical validation or as evidence of predictive superiority.
```

## Commit после обновления DOI

Так как DOI уже известен, `README.md` и `CITATION.cff` уже обновлены. После загрузки в GitHub сделайте commit:

```bash
git add README.md CITATION.cff .zenodo.json docs/GITHUB_ZENODO_STEPS_RU.md
git commit -m "Add existing Zenodo DOI and updated archive instructions"
git push
```
