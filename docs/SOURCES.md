# Data provenance and attribution

MaleCNS connectome v1.0, confidence threshold 0.5, downloaded 2026-09-13 from the official public bulk distribution:

- [Official dataset and bulk download documentation](https://male-cns.janelia.org/download/)
- [Annotations](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather)
- [Synaptic weights](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather)
- [Neurotransmitter predictions](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather)

Credit: FlyEM / HHMI Janelia, University of Cambridge Department of Zoology, MRC Laboratory of Molecular Biology, and Google Research. Dataset licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The graph here is a selected, normalized and signed derivative. Neuron identities, soma coordinates and synapse counts are retained. No endorsement implied. Source file sizes and SHA-256 digests are in `data/source_manifest.json`.

The model is our engineered rate reservoir, not the authors' biological model. GABA and glutamate are assigned negative signs; other predictions, including unclear and modulatory transmitters, are assigned positive signs. This is a coarse modeling assumption; receptor-specific effects are not modeled. All incoming counts are normalized per postsynaptic neuron. Soma positions are normalized for display, with a gentle camera rotation; displayed lines are measured connections, not neuron morphology or axon reconstructions.

Background reading used for approach selection: [Shiu et al. computational Drosophila model source](https://github.com/philshiu/Drosophila_brain_model). Its simulation code and FlyWire connectivity were not used in this project. The present work uses MaleCNS directly.

Visual assets, game, editing and synthesized audio were generated procedurally from original project code. No stock footage or generated fictional gameplay is used. The interface's online Google Fonts are optional; browser system fonts work offline. Video text uses system DejaVu Sans.
