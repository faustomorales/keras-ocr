Fine-tuning detector
====================

I currently do not have a tutorial for fine-tuning the text detector because
I am not aware of a real-world dataset with the required character-level bounding
box annotations. For now, all I have is training using the synthetic data generator (see
the first half of the end-to-end training example). There are two ways we could deal with this.

- We could implement the pseudo-ground truth label generation as described in the CRAFT paper. This is probably the best solution long-term.
- We could find a dataset with the required labeling granularity.

Please reach out on the repository if you want to help tackle one of these items.