# Data layout

The repository does not redistribute benchmark data, images, audio, model
weights, or private credentials. Create the following paths or override them
through `.env`:

```text
data/
  Mem-Gallery/
  H2HMEM/
  VoxPolyBench/
  h2h_source/
  dense_captions_h2hmem.json
  dense_captions_memgallery.json
  memgallery_legacy_facts/
```

`h2h_source` contains the five QA-free normalized dialogue streams named
`multi-party_dialogue1.json` through `multi-party_dialogue5.json`.
`memgallery_legacy_facts` supplies the QA-free visual-set metadata used by the
frozen contextual fact writer. All live paths can be absolute.

