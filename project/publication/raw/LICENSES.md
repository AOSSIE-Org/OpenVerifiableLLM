# Source-specific licensing and attribution

## Wikipedia

Credit the English Wikipedia contributors. The complete raw dump is redistributed
without changing its bytes. The [Wikimedia dump licensing guide](https://dumps.wikimedia.org/legal.html)
describes text as generally available under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode)
and [GFDL](https://www.gnu.org/licenses/fdl-1.3.html), with exceptions. Some text is
CC-only; imported material can have additional attribution requirements, and
fair-use material is not generally relicensed. Original notices remain in the raw
markup. This archive does not relicense third-party material or include media files.

For each XML `<page><id>PAGE_ID</id>` the corresponding page and contributor history
are at `https://en.wikipedia.org/w/index.php?curid=PAGE_ID` and
`https://en.wikipedia.org/w/index.php?curid=PAGE_ID&action=history`.
For `<revision><id>REVISION_ID</id>`, the source revision is at
`https://en.wikipedia.org/w/index.php?oldid=REVISION_ID`.
Substitute the recorded decimal ID. These links preserve attribution to the page's
contributors and allow inspection of its history and any additional notices.

Prepared article text, if published, is a modified form: markup, templates,
references and selected tags are removed by the committed extraction recipe.
Distribute those modifications under CC BY-SA 4.0 with source attribution and this
change notice. Keep the raw source and original notices obtainable alongside them.
The [Wikimedia Terms of Use, section 7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en#7._Licensing_of_Content)
govern attribution and reuse; the source-specific terms control over this summary.

## OpenAssistant

The original `conversation/LICENSE` and `conversation/README.md` accompany the
unchanged [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
train and validation Parquet files. The pinned source declares Apache License 2.0.
Retain its original license, attribution and dataset limitations when reusing it.
Any selected/normalized conversation records are separately documented modifications.

## Project code and evidence

Project code is governed by the source repository's GNU GPL version 3. That code
license is not a replacement for the data licenses above. No model-weight license
or verified-training claim is assigned by this raw source archive.
