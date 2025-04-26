ATTACK_MODE = 'fgsm'
EPSILON = 0.15
NUM_STEPS = 30
Clean = True

#if test in clean label attack, set the untargeted to True
if Clean:
    UNTARGETED = True
    TARGETED_LABEL = 1
else: 
    UNTARGETED = False
    TARGETED_LABEL = 8
    
EPSILON_STEP = EPSILON / NUM_STEPS
NUM_SAMPLES = 100
ROUND_TO_ATTACK = 10
