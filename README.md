# Getting started

This project enables the training of Flux1-dev (and any other models that use T5-XXL for prompt tokenization) on NSFW content and offers other improvements to tokenization by extending the T5 model's tokenizer with new vocabulary, and adjusting its embedding size accordingly.

If you want to get straight into it, pre-patched models are now available for [download on HuggingFace](https://huggingface.co/Kaoru8/T5XXL-Unchained). You can download one of those and the `tokenizer.json` file and skip to step 4.

If you already have the original `t5xxl_fp16.safetensors` model downloaded and want to save on bandwidth, here are the steps to patch it yourself:
### 1. Download and extract this repository

### 2. Install the required libraries

```
pip install torch transformers safetensors
```
### 3. Convert the vanilla T5-XXL model to the new architecture

Code has been tested on and confirmed working on the following model as a base for conversion:

[t5xxl_fp16.safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors) - 9.79 GB, SHA256 6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635

After you have that, you can convert it to use the new token embedding size by opening `convert.py` in a text editor, scrolling to the bottom, editing the `convertModel()` function to point to it (and optionally enable F8 quantization), then run the script. If you're currently running something else that uses up a lot of VRAM, shut it down temporarily as it will massively slow down the conversion process.
### 4. Patching ComfyUI to support inference with the new tokenizer and model

> [!CAUTION]
> This is a quick and dirty patch that will make ComfyUI work with the new T5 model and tokenizer, but will also break support for the vanilla ones. It is a temporary measure that lets you get the new things working immediately, while giving the developers time to implement proper support for the new tokenizer in a manner that works best for them.

- Download and setup [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

- Make a backup copy of `ComfyUI/comfy/text_encoders/t5_config_xxl.json`, then copy over the `t5_config_xxl.json` from this repository in its place.

- Make a backup copy of `ComfyUI/comfy/text_encoders/t5_tokenizer/tokenizer.json`, then copy over the `tokenizer.json` from this repository in its place.

Your ComfyUI install should now be able to load the new T5 model and use the matching tokenizer with any CLIP loader node, and inference should work.

**Keep in mind that the model released on HuggingFace (or the one you converted yourself) is "raw"** - meaning it was modified to work with the new tokenizer and embedding size, but it still hasn't actually been trained to do so. Without actually training the new T5 model and Flux on the new tokens, and if just using it out of the box as-is, expect the following:

- No capability to generate any of the concepts behind newly added tokens (NSFW or otherwise)
- Prompt adherence for pre-existing tokens from the vanilla tokenizer should be mostly unaffected, but a few words might have lower adherence
- You will get small border artifacts on about 10-15% of generated images, more on these below

All of these issues gradually resolve themselves with training, so let's get to that.
### 5. Patching Kohya's scripts to support training the new T5 model on the new tokenizer

> [!CAUTION]
> Same warning as above - patching things in this manner will break support for training the vanilla T5 model until more elegant official support for the new tokenizer can be implemented.

- Download the [sd3 branch of Kohya's Stable Diffusion scripts](https://github.com/kohya-ss/sd-scripts/tree/sd3)

- Make backup copies of `library/strategy_flux.py` and `library/flux_utils.py`, then copy over the `strategy_flux.py` and `flux_utils.py` from this repository in their place.

- Copy over the `tokenizer.json` from this repository to the `tests` directory.
### 6. Train the model

You can now point Kohya's scripts to the new T5 model path, and train in the same manner as usual. A couple of notes:

- Make sure that the `t5xxl` parameter is pointing to one of the new `T5XXL-Unchained` model variants instead of the vanilla ones

- Make sure that you're training both the UNet and T5, so use the `train_t5xxl=True` flag. Training just the UNet is of no use - both models need to be trained on the new tokens and embedding size in tandem to learn and adapt to them.

- The larger tokenizer and corresponding larger embeddings result in a slightly larger model size compared to vanilla T5. So if you're training on a low VRAM GPU and were using the `blocks_to_swap` argument to make training work for you before, and training is significantly slower now, you may need to increase the value by 1 to get the same training speeds as before.

- If you test generation with the raw model, and LORAs early on in the training process, you may see some minor artifacts on the edges of some of the generated images - fraying/fading, solid color borders, or mosaic patterns. This is normal and expected - the model has to gradually adjust to a significantly increased embedding size, most of which was initialized with random values. The artifacts will gradually dissipate and should eventually completely disappear after fine-tuning for a while. Worst case scenario, you may have to crop out about ~15 pixels from some edges of some images (or slightly scale them up until the artifacts are out of frame) early on, and you can probably automate that process with existing ComyUI workflows.

- Again, because the model is adapting to a new embedding size and new tokens it's never seen before, give it at least a solid 5-10k steps worth of training before making any conclusions about output quality, effectiveness of decensoring, new concept convergence, and prompt adherence. The model has a lot of old behaviors to un-learn, and a lot of new ones to learn. I guarantee that you'll eventually be pleasantly surprised. It just works - and not in the Todd Howard kind of way.
# About the new tokenizer

The original tokenizer had a vocabulary size of 32,100, while the newly uncensored one has been extended to a vocabulary size of 69,300. Aside from effectively uncensoring the model, this results in significantly more direct 1:1 word -> token/concept mappings (and therefore convergence speed during training and prompt adherence during inference) for the following:

- NSFW terms and anatomy
- Danbooru tags
- First and last names of people and characters
- Ethnicities and nationalities

I also considered extending it with more general vocabulary from a word frequency list, but that not only significantly blew up the tokenizer/embedding size further, but would also result in significantly lowered prompt adherence and output quality for general terms - the model would have to learn entirely new token mappings for a lot of pre-existing concepts that it already knew from the old tokenization schema, effectively forgeting them unless it could be trained on a huge high-quality dataset.

I think that this was the best possible compromise that simultaneously uncensors the model and will significantly improve its performance on the types of words listed above, while having no or negligable impact on current performance.

If you want to directly test the differences and improvements to tokenization with your own prompts, and intuitively understand why this works to both effectively uncensor the model as well as improve performance, you can do so as follows:

1. Download the vanilla T5-base model and tokenizer from [here](https://huggingface.co/google-t5/t5-base/tree/main). You only need the `config.json`, `generation_config.json`, `model.safetensors` and `tokenizer.json` files.
2. Put all of the files in a single folder, make a copy of the folder, then replace the `tokenizer.json` file in the copied folder with the one from this repository.
3. Open the `testTokenizer.py` file from this repository in a text editor, set the folder paths to the ones you just created, edit the prompt list to include whatever you want to test, then run the file in a terminal.

Testing very verbose prompts with lots of NSFW terms in them will be particularly illuminating.
# Donations

Code wants to be free, models want to be free, and you owe me nothing.

That said, if you end up finding value in this project and have some spare cash lying around, I would greatly appreciate a donation. Every little bit helps, and will go a long way toward my "get a less crappy GPU" savings fund - doing proof-of-concept training runs for this project on a 12GB VRAM GPU at 8-9 s/it was downright painful.

- Bitcoin: bc1q2n9x666c2uke6gj72rrrcvfe0jw9q0wmrqpfa5
- Ethereum: 0x0aa10Ee10C8717a2fb98b8eDB3C6Dde2C21512b0
- Credit/Debit card or Cash App: [Buy Me A Coffee](https://buymeacoffee.com/kaoru8)

Go forth, create, and enjoy yourselves. Just remember the words of Ben Parker - with great power comes great responsibility. Use the models, but please don't abuse them. I leave the distinction and the decision of where to draw that particular line up to your own judgement.