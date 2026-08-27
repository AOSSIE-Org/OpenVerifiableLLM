import bz2

# To make this tampered I deleted e of online

xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<mediawiki>
  <page>
    <revision>
      <text>
        Hello <ref>citation</ref> world.
        This is [[Python|programming language]]
        {{Wikipedia }}is a free onlin encyclopedia.
      </text>
    </revision>
  </page>
</mediawiki>
"""

with bz2.open("experiments/data_subset/tampered_sample_wiki.xml.bz2", "wt", encoding="utf-8") as f:
    f.write(xml_content)