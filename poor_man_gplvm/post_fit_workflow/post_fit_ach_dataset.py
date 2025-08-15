'''
For Yiyao Zhang's ACH dataset

need:
- transition events
    - continuous and fragmented chunks
        - further subdivide by sleep stage
    - previously tried detecting ACh bouts; however, it can be more or less noisy depending on the session, making detection criteria sensitive to sessions; thus 
    it's easier to detect the model-derived transition
- feature
    - ach, pop fr, consec pv difference, 
- peri transition feature average and shuffle
- tuning
    - jump probability vs sleep stage
- representational analysis 
    - (here the observation is the latent during continuous states tend to be similar within a NREM interval, but can be different across NREM intervals)
    - mean latent posterior 
- 

'''

