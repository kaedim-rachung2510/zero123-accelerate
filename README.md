## Zero-shot One Image to 3D Object on Accelerate

[The original demo](https://zero123.cs.columbia.edu/), which previously required 22GB of VRAM, can now be run in Colab using a Tesla T4.
This means you can explore large-scale diffusion modeling for 3D object reconstruction and novel view generation on less expensive hardware.

The underlying large-scale diffusion model is sharded and inference is done with [Accelerate](https://huggingface.co/docs/accelerate/) for an overall reduced memory footprint.

[Colab notebook](https://colab.research.google.com/drive/1iNpZqSlu8SMaDMVXLxp8a6jtwPB7LsUJ)
