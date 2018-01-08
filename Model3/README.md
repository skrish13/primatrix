- gave a 0.25 weightage in the sampler to the background class
- gave 0.5 to slightly majority
- running for 2 epochs instead basically
- using weighted loss with 1/x as ratio (lol)
- no pre-training (it becomes too tough to know which worked?)

Things thought of but didnt go through:
- remove humans (maybe later)
- undersample background more
- undersample humans more 

Inference:
- Its not weighted (need to prop analyze), so increasing accuracy on minority classes is useless as in no of examples youll increase it less
- but overfitting on majority class is fine but you need to get better scores so focus on not that much major and not very minor classes,


#### 

removed fucking sampling shit, gets
