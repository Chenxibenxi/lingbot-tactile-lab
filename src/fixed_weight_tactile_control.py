"""Input-only ablation: both conditions always select the same true-trained weights."""
from terminal_h50_contract import tactile_arm
CONDITIONS=('live','history_zero')
def weight_arm(condition):
    if condition not in CONDITIONS:raise ValueError('registered input-only condition required')
    return 'true'
def payload(current,condition):
    weight_arm(condition)
    return tactile_arm(current,'true' if condition=='live' else 'zero')
