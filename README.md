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

- If you test generation with the raw model, and LORAs early on in the training process, you may see some minor artifacts on the edges of some of the generated images - fraying/fading, solid color borders, or mosaic patterns. This is normal and expected - the model has to gradually adjust to a significantly increased embedding size, most of which was initialized with random values. The artifacts will gradually dissipate and should eventually completely disappear after fine-tuning for a while. Worst case scenario, you may have to crop out about ~15 pixels from some edges of some images (or slightly scale them up until the artifacts are out of frame) early on, and you can probably automate that process with existing ComfyUI workflows.

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

Testing very verbose prompts with lots of NSFW terms, Danbooru tags, or names in them will be particularly illuminating.

If you want more specifics on the sources and composition of the newly added vocabulary, it was compiled as follows:

1) The [List of Dirty, Naughty, Obscene and Otherwise Bad Words](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) that Google explicitly removed from their training set for the vanilla T5 + tokenizer was used as a base of NSFW terms. It was further extended with a list of canonical forms of terms from [an obscenity wordlist on Kaggle](https://www.kaggle.com/datasets/mathurinache/the-obscenity-list). The two lists were combined, de-duplicated, and cleaned up lightly (ie. removing phrases and keeping single words). The resulting list of 507 new NSFW terms was added to the tokenizer.
2) A large dataset of first and last names and associated statistics was obtained from [a dataset of names derived from Facebook data](https://pypi.org/project/names-dataset/). The most commonly used first names (rank <= 1500) and last names (rank <= 500) across 19 different countries (Australia, Brazil, Canada, China, France, Germany, India, Indonesia, Ireland, Italy, Japan, Mexico, Philippines, Korea, Russia, Spain, Thailand, United Kingdom, United States of America) were selected, and the resulting list was de-duplicated to a total of 10,779 unique first and last names. 934 of those already existed in the vanilla tokenizer, and the other 9,845 new ones were added to it.
3) A list of 274 nationalities and ethnicities was manually compiled from multiple Wikipedia articles. 66 were pre-existing in the vanilla tokenizer, and 208 new ones were added to it.
4) A list of all Danbooru tags (1,235,499) was collected using their official API. Tags of type `Copyright` were skipped. Tags of type `Artist` were selected by criteria `post_count >= 500`. Tags of type `Character` were selected by criteria `post_count >= 250`.  Tags of type `General` and `Meta` were selected by criteria `post_count >= 100`. `Character` type tags were cleaned and normalized to their natural-language equivalents, ie. the single tag "artoria\_pendragon_(fate)" would be converted to two separate tokens ["Artoria", "Pendragon"]. All other types of tags were added as-is. In total, this added 5,994 new character names (mostly Japanese and English) and other 20,646 Danbooru tags to the tokenizer.

The addition of so many Danbooru tags may be controversial, given how it's a captioning/prompting style that not everyone uses, and how many new tokens it accounts for in the extended tokenizer. However, I stand by the decision to include them, as I think they offer a lot of benefits and control over output even if they're not your primary prompting style, and I will attempt to explain my rationale here.

The main power and benefit of using Danbooru tags is that they can express very specific and/or very complex concepts with a single tag, and if we add them to the tokenizer, we can do the same with single tokens. Consider the following simplest example of this:

Your prompt is "a woman wearing bike shorts". The vanilla tokenizer doesn't see "bike shorts", as a single unified concept, it sees it as two distinct ones - "bike" and "shorts". So overwhelmingly, the models will generate a woman wearing some type of shorts, and a bicycle somewhere in the image. But we don't want the bicycle at all and didn't prompt for it (intiuitively - technically, we absolutely did as far as the tokenizer is concerned). So, what happens when you prompt with the Danbooru tag, ie. "a woman wearing bike_shorts" instead? Well, with the vanilla tokenizer, the same exact thing - it tokenizes it as ["bike", "\_", "shorts"], still seeing it as two (or rather, 3 now) distinct things. With the new tokenizer, however, the prompt using the Danbooru tag would be tokenized as ["a", "woman", "wearing", "bike_shorts"], where "bike_shorts" is a single distinct token that represents a single distinct concept - a specific type of shorts, with no bicycle necessarily involved. Accordingly, the models would have a far lower probability of generating a bicycle in the output unless explicitly prompted for it, ie. by adding "standing next to a bike" to the prompt.

As a direct result of this more direct and specific 1:1 concept to token mapping, models trained on the new tokens should converge on those concepts more easily than the vanilla model and tokenizer could during training time, and have improved prompt adherence by using Danbooru tags as powerful output control knobs during inference - even if most of your prompt is still written as natural language.

While the addition of so many new tokens definitely is an increase in complexity in technical terms (increased embedding size), it also serves to effectively decrease complexity and add specificity during training and inference for both the models themselves and the end user.

# Next steps

While we now have a fully uncensored and extended T5 + tokenizer combo capable of learning NSFW concepts as well as being improved in other ways, the fact remains that the current raw converted models are still untrained on all the new tokens and underlying concepts. We can start training LORAs on them, but the process is still way slower than it could be, because we're doing it from scratch with each new LORA.

Ideally, now that the required tools are out, someone would do a full fine-tune of the vanilla Flux and the new T5 on a very large dataset of both SFW and NSFW images whose captions provide good overall coverage of the majority of the new tokens (as well as old ones), and release them to the public as new foundational models.

These new foundational models would then provide uncensored generation out-of-the-box, have improved prompt adherence and concept understanding compared to vanilla Flux and the raw models, and make future personal LORA training using that as a base much faster, as the model would already have a good understanding of the new tokens and could just zero-in on the specifics of the LORA instead of having to learn the new tokens entirely from scratch every time.

In case there is someone with the compute power to throw at the task reading this, here are some tips and guidelines for how to approach the dataset creation and captioning process to maximize training effectiveness and the final quality and usefulness of the trained model:

> [!NOTE]
> If you just want to train personal LORAs on small datasets rather than train large foundational models, the following section doesn't apply to you - just compose and caption your datasets as you usually would, and it will work fine.

- Use a very large dataset - in a perfect world, at least 100-200k image-caption pairs would be a good start, but do whatever you can with the resources you have

- The dataset shouldn't be composed entirely of NSFW images. On the contrary - NSFW content should be a smaller portion of it. While simply uncensoring the base model was the initial motivation for this project, the reality is that NSFW terms and concepts only account for a small percentage of the newly added tokens, and that the new tokenizer has much more potential that should be properly tapped into to result in a more powerful and flexible model than vanilla Flux currently is even in entirely SFW domains. Aim for maybe 20-30% NSFW content, tops. That should be more than enough to completely remove all censorship from the models, teach them all possible NSFW concepts in a wide variety of permutations, while also not overfitting them to NSFW content and underfitting them to everything else.

- Within both SFW and NSFW subsets of the dataset, try to have a 50:50 split between real-world photos and artwork/illustration. This will help the model generalize better, have good coverage and variety in both domains, and allow for good blends between the two.

- When captioning, use a mix of both natural-language captions (such as those generated by Florence2 and JoyCaption) and Danbooru tags (like those generated by WD14) on every image. Again, this will make the models generalize better, and result in better prompt adherence with either style of prompting. It will also provide better coverage of the entire new token space and hopefully help the models converge on concepts faster.

- The Danbooru tag portion of the caption should just be space-separated (using commas would just be extra tokens), and randomly shuffled for each image - avoid sorting tags in alphabetic or any other fixed order - again, this will help the models generalize better.

- Swap around the order of captions. Use "natural-language + Danbooru" on about half of the dataset, then "Danbooru + natural-language" on the other half. Yet again, should help the models generalize better and not learn to rely on specific tokens being at a certain position in the prompt to work well.

Ultimately, you're the one throwing GPU power and money at the issue, so I leave the final decision of dataset size and composition up to you - but I highly recommend that you at least follow the captioning guidelines for the best results.
# Donations

Code wants to be free, models want to be free, and you owe me nothing.

That said, if you end up finding value in this project and have some spare cash lying around, I would greatly appreciate a donation. Every little bit helps, and will go a long way toward my "get a less crappy GPU" savings fund - doing proof-of-concept training runs for this project on a 12GB VRAM GPU at 8-9 s/it was downright painful.

- Bitcoin: bc1q2n9x666c2uke6gj72rrrcvfe0jw9q0wmrqpfa5
- Ethereum: 0x0aa10Ee10C8717a2fb98b8eDB3C6Dde2C21512b0
- Credit/Debit card or Cash App: [Buy Me A Coffee](https://buymeacoffee.com/kaoru8)

Go forth, create, and enjoy yourselves. Just remember the words of Ben Parker - with great power comes great responsibility. Use the models, but please don't abuse them. I leave the distinction and the decision of where to draw that particular line up to your own judgement.